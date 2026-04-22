import math
import gmsh
from gmshModel.Model import SimpleCubicCell, RandomInclusionRVE
from dissertacao.script.msh2dat_3D import rve_msh2dat  # Importa o seu conversor/otimizador
import os


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
        self.r = 0.0 # Raio calculado posteriormente
        self.nInc = 0 # Número de inclusões
        self.mesh_size=mesh_size=0.10
        # Propriedades padrão para o Abaqus
        self.EMat, self.nuMat = 2230.0, 0.3
        self.EInc, self.nuInc = 70000.0, 0.2

    def _configurar_malha(self):
        """Define os parâmetros da malha de forma inteligente baseada na ordem do elemento."""
        elementos_circ = 20 if self.ordem_elemento == 1 else 12
        elementos_entre = 2 if self.ordem_elemento == 1 else 1
        
        return {
            "threads": None,
            "refinementOptions": {
                "maxMeshSize": self.L * self.mesh_size,
                "inclusionRefinement": False,
                "interInclusionRefinement": False,
                "elementsBetweenInclusions": elementos_entre,
                "transitionElements": "auto",
                "aspectRatio": 1.5
            }
        }

    def gerar_esfera_central(self):
        """Gera um RVE com uma única partícula perfeitamente centrada."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", self.ordem_elemento)
        
        self.r = ((3.0 * self.vf * (self.L**3)) / (4.0 * math.pi))**(1.0/3.0)
        self.nInc = 1
        
        init_params = {
            "numberCells": [1, 1, 1],
            "radius": self.r,
            "size": [self.L, self.L, self.L],
            "inclusionType": "Sphere",
            "origin": [self.L/2, self.L/2, self.L/2],
            "periodicityFlags": [1, 1, 1],
            "domainGroup": "matrix",
            "inclusionGroup": "inclusion", # Singular para o FAST
            "gmshConfigChanges": {
                "General.Terminal": 0,
                "Mesh.CharacteristicLengthExtendFromBoundary": 0
            }
        }
        
        self.rve_model = SimpleCubicCell(**init_params)
        self._construir_geometria_e_malha()
        print(f"RVE Central gerado. Raio: {self.r:.4f}")

    def gerar_esferas_aleatorias(self, r_fixo=0.15, min_dist=0.10):
        """Gera um RVE com múltiplas partículas aleatórias usando o algoritmo RSA."""
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
            "inclusionGroup": "inclusion", # Singular para o FAST
            "gmshConfigChanges": {
                "General.Terminal": 0,
                "Mesh.CharacteristicLengthExtendFromBoundary": 0,
                "Geometry.MatchMeshTolerance": 1.e-07,
                "Mesh.MshFileVersion": 2.2 # Crucial para compatibilidade
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
        """Método interno para automatizar a burocracia do Gmsh."""
        if isinstance(self.rve_model, SimpleCubicCell):
            self.rve_model.defineGeometricObjects()
            self.rve_model.addGeometricObjectsToGmshModel()
            
        self.rve_model.defineBooleanOperations()
        self.rve_model.performBooleanOperationsForGmshModel()
        self.rve_model.definePhysicalGroups()

        # Limpeza de contornos
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
        self.rve_model.visualizeMesh()
    # ==========================================================
    # MÓDULOS DE EXPORTAÇÃO
    # ==========================================================
    def exportar_para_fast(self, nome_ficheiro):
        """Guarda o .msh preservando os nomes físicos (exigência do msh2dat do FAST)."""
        self.rve_model.saveMesh(nome_ficheiro)
        print(f"[FAST] Ficheiro guardado: {nome_ficheiro}")

    def exportar_para_jive(self, nome_ficheiro):
        """Remove os nomes físicos e guarda o .msh (exigência do script go.sh do Jive)."""
        # Remove os nomes apenas na memória antes de guardar
        try:
            self.rve_model.gmshAPI.removePhysicalName('matrix')
            self.rve_model.gmshAPI.removePhysicalName('inclusion')
        except:
            pass # Caso já tenham sido removidos
        
        self.rve_model.saveMesh(nome_ficheiro)
        print(f"[JIVE] Ficheiro guardado (sem nomes físicos): {nome_ficheiro}")

    def exportar_para_abaqus(self, nome_msh_temp, nome_inp_final):
        """Guarda um .msh temporário, converte para .inp e injeta os materiais."""
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
        """Limpa a memória e encerra o motor do Gmsh."""
        if self.rve_model:
            self.rve_model.close()

    def gerar_fibra_continua(self, eixo=[0, 0, 1]):
        """Gera um RVE 3D com uma fibra cilíndrica contínua."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", self.ordem_elemento)
        
        # Para um cilindro que cruza o cubo (altura = L), o volume é pi * r^2 * L
        self.r = math.sqrt((self.vf * (self.L**3)) / (math.pi * self.L))
        
        init_params = {
            "numberCells": [1, 1, 1],
            "radius": self.r,
            "size": [self.L, self.L, self.L],
            "inclusionType": "Cylinder",
            "inclusionAxis": eixo,
            "origin": [self.L/2, self.L/2, 0], # Base do cilindro
            "periodicityFlags": [1, 1, 1],
            "domainGroup": "matrix",
            "inclusionGroup": "inclusion" # Mantido no singular para o FAST
        }
        
        from gmshModel.Model import SimpleCubicCell
        self.rve_model = SimpleCubicCell(**init_params)
        self._construir_geometria_e_malha()
        print(f"RVE de Fibra gerado. Raio: {self.r:.4f}")


if __name__ == "__main__":

    print("\n" + "="*50)
    print(" FÁBRICA DE RVES - MULTI-SOFTWARE (FAST, ABAQUS, JIVE)")
    print("="*50)

    # 1. Configuração do RVE
    fracoes_volume = [2.5, 5, 10, 20, 25, 30, 40, 50]     # %
    fracao_volume = fracoes_volume[4]     # %
    
    ordens_elem = [1,2]         # 1 = TET4 (Rápido), 2 = TET10 (Pesado)
    ordem_elem = ordens_elem[0]         # 1 = TET4 (Rápido), 2 = TET10 (Pesado)
    
    mesh=[0.05, 0.075,0.1,0.15,0.2,0.25]
    mesh_size = mesh[3]
    raio_particula = 0.15  # Raio fixo para o algoritmo RSA
    #prefixo = f"RVE_Aleatorio_{fracao_volume}pct"
    prefixo = f"RVE_centrado_{fracao_volume}_mesh({mesh_size})_tipo_({ordem_elem})"

    # 2. Geração da Geometria Base
    meu_rve = RVEGenerator(vf_percent=fracao_volume, ordem_elemento=ordem_elem, mesh_size=mesh_size)
    #meu_rve.gerar_esferas_aleatorias(r_fixo=raio_particula)
    meu_rve.gerar_esfera_central()

    # ====================================================================
    # EXPORTAÇÃO 1: ABAQUS (Cria o .inp com malha, materiais e ELSETs)
    # ====================================================================
    nome_abaqus = f"{prefixo}_Abaqus.inp"
    # Ele usa um .msh temporário por trás dos panos para gerar o .inp
    #meu_rve.exportar_para_abaqus(nome_msh_temp="temp_abaqus.msh", nome_inp_final=nome_abaqus)
    if os.path.exists("temp_abaqus.msh"):
        os.remove("temp_abaqus.msh") # Limpa a sujeira

    # ====================================================================
    # EXPORTAÇÃO 2: FAST (Cria o .msh com nomes físicos para o conversor)
    # ====================================================================
    nome_fast_msh = f"{prefixo}_FAST.msh"
    meu_rve.exportar_para_fast(nome_fast_msh)

    # ====================================================================
    # EXPORTAÇÃO 3: JIVE (Tem que ser o último! Apaga nomes físicos)
    # ====================================================================
    nome_jive = f"{prefixo}_Jive.msh"
    #meu_rve.exportar_para_jive(nome_jive)

    # Fecha a API do Gmsh para liberar memória
    meu_rve.fechar()

    # ====================================================================
    # EXPORTAÇÃO 4: OTIMIZAÇÃO DO FAST (.msh -> .dat com PBCs e Sloan)
    # ====================================================================
    print("\n--- INICIANDO TRADUÇÃO E OTIMIZAÇÃO PARA O FAST ---")
    
    # Propriedades: [Matriz, Inclusão]
    E_modulos = [70000.0, 400000.0]   
    nu_poissons = [0.33, 0.2]
    
    # Deformação Macro a aplicar (ex: Tração uniaxial em X)
    deformacao_macro = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0] 

    # Chama a função que lê o MSH do FAST, aplica equações e otimiza os nós
    rve_msh2dat(nome_fast_msh, E_modulos, nu_poissons, deformacao_macro)
    
    print("\n" + "="*50)
    print(" PROCESSO CONCLUÍDO COM SUCESSO!")
    print(" O arquivo .dat está pronto para ser executado no fast.exe")
    print(" O arquivo .inp está pronto para ser executado no Abaqus")
    print("="*50)