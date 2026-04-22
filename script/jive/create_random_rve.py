#!/usr/bin/python3

#------------------------------------------------------------------------
#    Simple GmshModel Python API functionalities 
#
#    Reference: GmshModel documentations
#    https://gmshmodel.readthedocs.io/en/latest/index.html
#    Accessed: July 2025
#
#    Author: Luiz Ant. T. Mororo, luiz.mororo@ifce.edu.br
#    Date:   July 2025
#------------------------------------------------------------------------

import sys
import math
import gmsh
import gmshModel

# ==========================================
# INPUT PARAMETERS
# ==========================================
r = 0.005/2.  # particle size (radius)
volFracs = [10,20,30] # volume fraction [%]
meshPer = 7.5 
nInc_alvo = 5 # number of inclusions alvo

gmAPI = gmshModel.Model

for volFrac in volFracs:
    
    # 1. Matemática robusta para partículas esféricas
    volInc = (4.0 / 3.0) * math.pi * (r**3)
    vol_inclusoes_total = nInc_alvo * volInc
    vol_cubo_necessario = vol_inclusoes_total / (volFrac / 100.0)
    a = vol_cubo_necessario ** (1.0 / 3.0)
    nInc = nInc_alvo

    print('\n' + '='*60)
    print(f' INICIANDO GERAÇÃO: Vf = {volFrac}% | a = {a:.6f}')
    print(f' Partículas = {nInc} | Volume Total = {vol_cubo_necessario:.5e}')
    print('='*60)

    initParameters={                                                           
        "inclusionSets": [r, nInc],                                  
        "inclusionType": "Sphere",                                             
        "size": [a, a, a],                                                     
        "origin": [0, 0, 0],                                                   
        "periodicityFlags": [1, 1, 1],                                         
        "domainGroup": "matrix",                                               
        "inclusionGroup": "inclusions",                                        
        "gmshConfigChanges": {
            "General.Terminal": 0,                             
            "Mesh.CharacteristicLengthExtendFromBoundary": 0,
            "Mesh.SecondOrderLinear": 1,
            "Mesh.ElementOrder": 2,
            "Geometry.MatchMeshTolerance": 1.e-06, # Tolerância apertada para booleanos
            "Mesh.MshFileVersion": 2.2             # Formato JIVE/FAST
        }
    }

    modelingParameters = {                                                     
        "placementOptions": {
            "maxAttempts": 10000,      # Aumentado para dar mais chance ao gerador
            "minRelDistBnd": 0.05,     # Reduzido para permitir aproximação da borda
            "minRelDistInc": 0.05,     # Reduzido para permitir aglomeração
        }
    }

    meshingParameters={                                                        
        "threads": None,                                                       
        "refinementOptions": {
            "maxMeshSize": a*meshPer/100.,                            
            "inclusionRefinement": True,                     
            "interInclusionRefinement": False,                
            "elementsPerCircumference": 20,  # Reduzido de 30 para evitar colisões de facetas
            "elementsBetweenInclusions": 4,                  
            "inclusionRefinementWidth": 4,                   
            "transitionElements": "auto",                    
            "aspectRatio": 1.5                               
        }
    }

    # ==========================================
    # LOOP DE ROBUSTEZ (TENTATIVA E ERRO AUTOMÁTICO)
    # ==========================================
    MAX_TENTATIVAS = 20
    tentativa = 0
    sucesso = False

    while tentativa < MAX_TENTATIVAS:
        tentativa += 1
        print(f"\n--- Tentativa {tentativa}/{MAX_TENTATIVAS} ---")
        
        testRVE = gmAPI.RandomInclusionRVE(**initParameters)
        
        try:
            # 2. Definição e Posicionamento Geométrico
            testRVE.defineGeometricObjects(**modelingParameters)
            testRVE.addGeometricObjectsToGmshModel()
            
            # --- INTERCEÇÃO RÁPIDA (FAIL-FAST) ---
            # Sincroniza o motor CAD e conta os volumes gerados ANTES de cortar
            gmsh.model.occ.synchronize()
            qtd_volumes_3d = len(gmsh.model.occ.getEntities(3))
            esferas_colocadas = qtd_volumes_3d - 1 # Subtrai 1 do cubo da matriz
            
            if esferas_colocadas < nInc:
                print(f"[AVISO] O gerador só conseguiu espaço para {esferas_colocadas}/{nInc} partículas.")
                print("Reiniciando a semente aleatória...")
                testRVE.close()
                continue # Volta ao início do 'while' sem processar a malha
                
            print(f"[OK] {nInc}/{nInc} partículas inseridas. Iniciando cortes booleanos...")
            
            # 3. Cortes e Operações Booleanas
            testRVE.defineBooleanOperations()                                          
            testRVE.performBooleanOperationsForGmshModel()                             
            testRVE.definePhysicalGroups()                                             
            
            del testRVE.groups['boundary']
            testRVE.physicalGroups.pop()
            
            testRVE.addPhysicalGroupsToGmshModel()                                     
            testRVE.gmshAPI.removePhysicalName('matrix')
            testRVE.gmshAPI.removePhysicalName('inclusions')
            testRVE.setupPeriodicity()                                                 
            
            # 4. Geração da Malha
            print("Geometria validada. A gerar malha de Elementos Finitos...")
            testRVE.createMesh(**meshingParameters)
            
            # 5. Guardar o ficheiro
            nome_arquivo = f'rve_vf{int(volFrac)}.msh2'
            testRVE.saveMesh(nome_arquivo)
            print(f"[SUCESSO] RVE gerado e guardado como: {nome_arquivo}")
            
            testRVE.close()
            sucesso = True
            break # Sai do ciclo 'while' pois já conseguiu gerar perfeitamente
            
        except Exception as e:
            # Captura o erro "Invalid boundary mesh (overlapping facets)"
            print(f"[FALHA NA MALHA] Erro do motor Gmsh: {e}")
            print("Geometria sobreposta detetada durante o corte. Reiniciando modelo...")
            testRVE.close()

    if not sucesso:
        print(f"\n[ERRO CRÍTICO] Falha ao gerar Vf={volFrac}% após {MAX_TENTATIVAS} tentativas.")
        print("Tente reduzir ainda mais a fração volumétrica ou as distâncias relativas (minRelDistInc).")