import math
import gmsh
from gmshModel.Model import SimpleCubicCell, RandomInclusionRVE
from msh2dat_3D import rve_msh2dat
import os
import shutil
import numpy as np

class RVEGenerator:
    """
    Fábrica unificada para geração de Elementos de Volume Representativos (RVEs) 3D.
    Suporta homogeneização elástica e plástica para FAST, Abaqus e Jive.
    """
    def __init__(self, L=1.0, vf_percent=10, ordem_elemento=1, mesh_size=0.10):
        self.L = L
        self.vf = vf_percent / 100.0
        self.ordem_elemento = ordem_elemento
        self.rve_model = None
        self.r = 0.0 
        self.nInc = 0 
        self.mesh_size=mesh_size
        self.EMat, self.nuMat = 70000, 0.33
        self.EInc, self.nuInc = 400000.0, 0.2

    def _configurar_malha(self):
        # 1. Tamanho global desejado para o elemento (h)
        tamanho_elemento = self.L * self.mesh_size
        perimetro = 2.0 * math.pi * self.r
        elementos_circ = max(2*int(perimetro / tamanho_elemento), 10)
        if self.ordem_elemento == 2:
            elementos_circ = max(2*int(perimetro / tamanho_elemento), 8)

        return {
            "threads": None,
            "refinementOptions": {
                "maxMeshSize": tamanho_elemento,
                "inclusionRefinement": True,
                "interInclusionRefinement": False,
                "elementsPerCircumference": elementos_circ,
                "elementsBetweenInclusions": 3,
                "inclusionRefinementWidth": 3,
                "transitionElements": "auto",
                "aspectRatio": 1.5
            }
        }

        """return {
            "threads": None,
            "refinementOptions": {
                "maxMeshSize": tamanho_elemento,
                "inclusionRefinement": False,
                "interInclusionRefinement": False,
                "elementsBetweenInclusions": 0,
                "inclusionRefinementWidth": 0,
                "transitionElements": "auto",
                "aspectRatio": 1.5
            }
        }"""

    def gerar_esfera_central(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", self.ordem_elemento)
        
        # Correção matemática para quando a fração volumétrica for ZERO
        if self.vf == 0.0:
            # Se for 0, gera um cubo sem inclusões (Apenas Matriz)
            self.r = 0.0001  # Um raio ínfimo só para o Gmsh não dar crash (ou você pode pular a criação da esfera)
            self.nInc = 0
            # NOTA: Para RVEs 100% matriz, seria melhor não usar a inclusão na geometria, 
            # mas vamos manter o fluxo genérico por enquanto com um raio quase nulo.
        else:
            self.r = ((3.0 * self.vf * (self.L**3)) / (4.0 * math.pi))**(1.0/3.0)
            self.nInc = 1
        
        init_params = {
            "numberCells": [1, 1, 1],
            "radius": self.r,
            "size": [self.L, self.L, self.L],
            "inclusionType": "Sphere",
            "origin": [0, 0, 0],
            "periodicityFlags": [1, 1, 1],
            "domainGroup": "matrix",
            "inclusionGroup": "inclusion",
            "gmshConfigChanges": {
                "General.Terminal": 0,
                "Mesh.CharacteristicLengthExtendFromBoundary": 0,
                "Mesh.MshFileVersion": 2.2

            }
        }
        
        self.rve_model = SimpleCubicCell(**init_params)
        self._construir_geometria_e_malha()
        print(f"RVE Central gerado. Raio: {self.r:.4f}")

    def gerar_esferas_aleatorias(self, r_fixo=0.15, min_dist=0.10):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", self.ordem_elemento)
        
        self.r = r_fixo
        vol_inc = (4.0 * math.pi * self.r**3.0) / 3.0
        vol_tot = self.L**3.0
        self.nInc = round(vol_tot * self.vf / vol_inc)
        
        init_params = {
            "inclusionSets": [self.r, self.nInc],
            "inclusionType": "Sphere",
            "size": [self.L, self.L, self.L],
            "origin": [0, 0, 0],
            "periodicityFlags": [1, 1, 1],
            "domainGroup": "matrix",
            "inclusionGroup": "inclusion",
            "gmshConfigChanges": {
                "General.Terminal": 0,
                "Mesh.CharacteristicLengthExtendFromBoundary": 0,
                "Geometry.MatchMeshTolerance": 1.e-07,
                "Mesh.MshFileVersion": 2.2 
            }
        }
        
        self.rve_model = RandomInclusionRVE(**init_params)
        
        modeling_params = {
            "placementOptions": {
                "maxAttempts": 10000,
                "minRelDistBnd": min_dist,
                "minRelDistInc": min_dist,
            }
        }
        
        self.rve_model.defineGeometricObjects(**modeling_params)
        self.rve_model.addGeometricObjectsToGmshModel()
        self._construir_geometria_e_malha()
        print(f"RVE Aleatório gerado. {self.nInc} partículas de raio {self.r:.4f}")

    def _construir_geometria_e_malha(self):
        if isinstance(self.rve_model, SimpleCubicCell):
            self.rve_model.defineGeometricObjects()
            self.rve_model.addGeometricObjectsToGmshModel()
            
        self.rve_model.defineBooleanOperations()
        self.rve_model.performBooleanOperationsForGmshModel()
        self.rve_model.definePhysicalGroups()

        if 'boundary' in self.rve_model.groups:
            del self.rve_model.groups['boundary']
            self.rve_model.physicalGroups.pop()

        if isinstance(self.rve_model, SimpleCubicCell):
            matrix_tag = self.rve_model.groups['domain'][0][1]
            inc_tags = [tag for dim, tag in self.rve_model.groups['inclusions']]
            gmsh.model.addPhysicalGroup(3, [matrix_tag], name="matrix")
            gmsh.model.addPhysicalGroup(3, inc_tags, name="inclusion")
        else:
            self.rve_model.addPhysicalGroupsToGmshModel()

        self.rve_model.setupPeriodicity()
        self.rve_model.createMesh(**self._configurar_malha())
        #self.rve_model.visualizeMesh()

    def exportar_para_fast(self, nome_ficheiro):
        self.rve_model.saveMesh(nome_ficheiro)
        print(f"[FAST] Ficheiro guardado: {nome_ficheiro}")

    def exportar_para_jive(self, nome_ficheiro):
        try:
            self.rve_model.gmshAPI.removePhysicalName('matrix')
            self.rve_model.gmshAPI.removePhysicalName('inclusion')
        except:
            pass 
        self.rve_model.saveMesh(nome_ficheiro)
        print(f"[JIVE] Ficheiro guardado (sem nomes físicos): {nome_ficheiro}")

    def exportar_para_abaqus(self, nome_msh_temp, nome_inp_final):
        self.exportar_para_fast(nome_msh_temp)
        try:
            gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
            gmsh.write(nome_inp_final)
            with open(nome_inp_final, 'a') as f:
                f.write("\n** MATERIAIS INJETADOS VIA PYTHON\n")
                f.write("*MATERIAL, NAME=MatMaterial\n*ELASTIC\n")
                f.write(f"{self.EMat}, {self.nuMat}\n")
                f.write("*MATERIAL, NAME=IncMaterial\n*ELASTIC\n")
                f.write(f"{self.EInc}, {self.nuInc}\n")
                f.write("*SOLID SECTION, ELSET=matrix, MATERIAL=MatMaterial\n")
                f.write("*SOLID SECTION, ELSET=inclusion, MATERIAL=IncMaterial\n")
            print(f"[ABAQUS] Ficheiro pronto: {nome_inp_final}")
        except Exception as e:
            print(f"Erro ao exportar para Abaqus: {e}")

    def fechar(self):
        if self.rve_model:
            self.rve_model.close()
            
    def contar_nos(self):
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        return len(node_tags)
        
    def contar_elementos(self):
        _, element_tags, _ = gmsh.model.mesh.getElements(3)
        return sum(len(tags) for tags in element_tags)
    def verificar_qualidade_malha(self):
        """
        Verifica se existem elementos invertidos (Jacobiano <= 0) na malha 3D.
        Retorna True se a malha estiver saudável, e False se estiver corrompida.
        """
        
        # Obtém todos os elementos 3D (tetraedros) da malha atual
        tipos_elem, tags_elem, _ = gmsh.model.mesh.getElements(dim=3)
        
        elementos_ruins = 0
        
        for i, tipo in enumerate(tipos_elem):
            # Calcula o Jacobiano Mínimo ("minJ") para cada elemento deste tipo
            qualidades = gmsh.model.mesh.getElementQualities(tags_elem[i], qualityName="minSJ")
            
            # Conta quantos elementos têm Jacobiano menor ou igual a zero
            ruins = np.sum(np.array(qualidades) <= 0.0)
            elementos_ruins += ruins
            
        if elementos_ruins > 0:
            print(f"\n ALERTA CRÍTICO: Foram encontrados {elementos_ruins} elementos com volume zero ou invertidos!")
            print("   O Abaqus e o FAST VÃO FALHAR com esta malha. A exportação será ignorada.")
            return False
        else:
            print("\n✅ Verificação de malha: Todos os elementos 3D estão saudáveis (Jacobiano positivo).")
            return True
    def gerar_fibra_continua(self, eixo=[0, 0, 1]):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", self.ordem_elemento)
        self.r = math.sqrt((self.vf * (self.L**3)) / (math.pi * self.L))
        init_params = {
            "numberCells": [1, 1, 1],
            "radius": self.r,
            "size": [self.L, self.L, self.L],
            "inclusionType": "Cylinder",
            "inclusionAxis": eixo,
            "origin": [0, 0, 0],
            "periodicityFlags": [1, 1, 1],
            "domainGroup": "matrix",
            "inclusionGroup": "inclusion"
        }
        self.rve_model = SimpleCubicCell(**init_params)
        self._construir_geometria_e_malha()
        print(f"RVE de Fibra gerado. Raio: {self.r:.4f}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" FÁBRICA DE RVES - BATCH AUTOMÁTICO")
    print("="*50)

    # Note: vf=0 pode causar singularidade na geometria. Foi adicionado um patch lá em cima.
    fracoes_volume = [5, 10, 20, 30]  
    fracoes_volume = [30]  
    ordens_elem = [1]  
    malhas = [0.3, 0.2, 0.15, 0.10, 0.075, 0.05]
    malhas = [0.3, 0.2, 0.15, 0.10]
    malhas = [0.045, 0.04,0.035]
    malhas = [0.25]

    pasta_saida = "processar"
    os.makedirs(pasta_saida, exist_ok=True)

    E_modulos = [78e3,395e3]
    nu_poissons = [0.33, 0.2]
    deformacao_macro = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]

    for vf in fracoes_volume:
        for mesh_size in malhas:
            for ordem_elem in ordens_elem:

                print(f"\n>>> Gerando: VF={vf}% | mesh={mesh_size} | ordem={ordem_elem}")
                
                # Nome inicial sem nós/elementos (usado temporariamente)
                prefixo_temp = f"TEMP_RVE_vf{vf}_m{mesh_size}_ord{ordem_elem}"

                try:
                    meu_rve = RVEGenerator(vf_percent=vf, ordem_elemento=ordem_elem, mesh_size=mesh_size)
                    meu_rve.gerar_esfera_central()
                    #meu_rve.gerar_esferas_aleatorias(r_fixo=0.15, min_dist=0.10)
                    num_nos = meu_rve.contar_nos()
                    num_elem = meu_rve.contar_elementos()
                    qualidade_malha=meu_rve.verificar_qualidade_malha()
                    if not qualidade_malha:
                        print(f"Malha: Vf{vf} mesh={mesh_size} com problema ")
                        continue
                    # =============== A MÁGICA ESTÁ AQUI ===============
                    # Agora que temos a contagem, montamos o prefixo DEFINITIVO
                    prefixo_final = f"RVE_vf{vf}_m{mesh_size}_ord{ordem_elem}_N{num_nos}_E{num_elem}"
                    # ==================================================

                    # 1. Exporta para FAST
                    nome_fast_msh = os.path.join(pasta_saida, f"{prefixo_final}_FAST.msh")
                    meu_rve.exportar_para_fast(nome_fast_msh)

                    # 2. Exporta para Abaqus (Usando um temp puro para a conversão interna)
                    nome_abaqus = os.path.join(pasta_saida, f"{prefixo_final}_Abaqus.inp")
                    temp_msh_abaqus = os.path.join(pasta_saida, f"temp_{prefixo_final}.msh")
                    #meu_rve.exportar_para_abaqus(nome_msh_temp=temp_msh_abaqus, nome_inp_final=nome_abaqus)
                    # Limpa o lixo temporário do abaqus
                    if os.path.exists(temp_msh_abaqus): os.remove(temp_msh_abaqus)

                    # 3. Exporta para Jive
                    nome_jive = os.path.join(pasta_saida, f"{prefixo_final}_Jive.msh")
                    #meu_rve.exportar_para_jive(nome_jive)

                    # 4. Chama a caixa preta (msh2dat) que salva o .dat na raiz
                    rve_msh2dat(nome_fast_msh, E_modulos, nu_poissons, deformacao_macro)

                    # 5. Move o .dat da raiz para a pasta 'processar'
                    nome_dat_raiz = f"{prefixo_final}_FAST.dat"
                    nome_dat_destino = os.path.join(pasta_saida, nome_dat_raiz)
                    if os.path.exists(nome_dat_raiz):
                        shutil.move(nome_dat_raiz, nome_dat_destino)

                    print(f"Malha consolidada! Nós: {num_nos} | Elementos: {num_elem}")

                except Exception as e:
                    print(f"Erro no caso VF={vf}, mesh={mesh_size}: {e}")
                finally:
                    if 'meu_rve' in locals():
                        meu_rve.fechar()

    print("\n" + "="*50)
    print(" TODOS OS CASOS FORAM PROCESSADOS!")
    print(f" Arquivos finais (nomeados com Nós/Elementos) guardados em: {pasta_saida}/")
    print("="*50)