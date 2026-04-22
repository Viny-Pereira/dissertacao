import os
import glob
import re
import numpy as np
import pandas as pd

# ==========================================
# 1. MODELOS ANALÍTICOS (TEORIA)
# ==========================================
class ModelosAnaliticos:
    @staticmethod
    def voigt(Em, Ec, num, nuc, Vc):
        """Limite Superior: Pressupõe deformação uniforme."""
        E = Em + (Ec - Em) * Vc
        nu = num + (nuc - num) * Vc
        G = E / (2 * (1 + nu))
        return E, nu, G

    @staticmethod
    def reuss(Em, Ec, num, nuc, Vc):
        """Limite Inferior: Pressupõe tensão uniforme."""
        E = (Em * Ec) / (Em * Vc + Ec * (1 - Vc))
        nu = (num * nuc) / (num * Vc + nuc * (1 - Vc))
        G = E / (2 * (1 + nu))
        return E, nu, G

    @staticmethod
    def mori_tanaka(Em, Ec, num, nuc, Vc):
        """Modelo Mori-Tanaka para inclusões esféricas."""
        Km = Em / (3 * (1 - 2 * num))
        Kc = Ec / (3 * (1 - 2 * nuc))
        Gm = Em / (2 * (1 + num))
        Gc = Ec / (2 * (1 + nuc))
        
        Vm = 1.0 - Vc
        fm = Gm * (9 * Km + 8 * Gm) / (6 * (Km + 2 * Gm))
        
        K = Km + Vc / (1 / (Kc - Km) + Vm / (Km + 4 * Gm / 3))
        G = Gm + Vc / (1 / (Gc - Gm) + Vm / (Gm + fm))
        
        E = (9 * K * G) / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        return E, nu, G

# ==========================================
# 2. EXTRAÇÃO NUMÉRICA (INVERSÃO DA MATRIZ)
# ==========================================
def analisar_matriz_cm_3d(CM):
    """Inverte a matriz constitutiva 6x6 para extrair a Compliance [S]."""
    S = np.linalg.inv(CM) 
    
    # Módulos de Young (1/S_ii)
    E11, E22, E33 = 1.0/S[0, 0], 1.0/S[1, 1], 1.0/S[2, 2]
    
    # Coeficientes de Poisson (-S_ij / S_ii)
    nu12, nu13, nu23 = -S[0, 1]/S[0, 0], -S[0, 2]/S[0, 0], -S[1, 2]/S[1, 1]
    
    # Módulos de Cisalhamento (1/S_jj)
    G23, G13, G12 = 1.0/S[3, 3], 1.0/S[4, 4], 1.0/S[5, 5]
    
    return E11, E22, E33, G12, G13, G23, nu12, nu13, nu23

# ==========================================
# 3. MOTOR DO CRAWLER E GERAÇÃO DE CSV
# ==========================================
def minera_e_compara(pasta_raiz):
    print("="*65)
    print(" MINERAÇÃO E COMPARAÇÃO MICROMECÂNICA (JIVE/FAST)")
    print("="*65)
    
    # --> AJUSTE AS PROPRIEDADES DA SUA MATRIZ AQUI (Valores base do .pro)
    Em_val = 2462.58
    num_val = 0.38
    nuc_val = 0.49 # Poisson da inclusão do seu .pro
    # <--
    
    padrao_busca = os.path.join(pasta_raiz, "E_*", "vf_*", "MicroPBC.out")
    arquivos = glob.glob(padrao_busca)
    
    if not arquivos:
        print("[ERRO] Nenhum ficheiro MicroPBC.out encontrado.")
        return
        
    dados_totais = []
    
    for caminho in arquivos:
        # Extrair Ec e vf a partir do nome das pastas
        match = re.search(r'E_(\d+(?:\.\d+)?)[\\/]vf_(\d+(?:\.\d+)?)', caminho)
        if not match:
            continue
            
        Ec_val = float(match.group(1))
        vf_percent = float(match.group(2))
        Vc_val = vf_percent / 100.0 # Converter % para decimal
        
        # Ler arquivo
        with open(caminho, 'r') as f:
            conteudo = f.read()
            
        if "stiff:" in conteudo:
            bloco_stiff = conteudo.split("stiff:")[1]
            padrao_numero = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?'
            stiff_vals = re.findall(padrao_numero, bloco_stiff)
            
            if len(stiff_vals) >= 36:
                # Constrói a matriz numpy 6x6
                CM = np.array([float(v) for v in stiff_vals[:36]]).reshape((6, 6))
                
                # 1. Numérico
                E11, E22, E33, G12, G13, G23, nu12, nu13, nu23 = analisar_matriz_cm_3d(CM)
                E_num_avg = (E11 + E22 + E33) / 3.0
                anisotropia_E = max(abs(E11 - E22), abs(E11 - E33), abs(E22 - E33)) / E11 * 100
                
                # 2. Analítico
                E_v, nu_v, G_v = ModelosAnaliticos.voigt(Em_val, Ec_val, num_val, nuc_val, Vc_val)
                E_r, nu_r, G_r = ModelosAnaliticos.reuss(Em_val, Ec_val, num_val, nuc_val, Vc_val)
                E_mt, nu_mt, G_mt = ModelosAnaliticos.mori_tanaka(Em_val, Ec_val, num_val, nuc_val, Vc_val)
                
                # Adicionar à lista de dados
                dados_totais.append({
                    'Vc_fracao': Vc_val,
                    'Ec_Inclusao': Ec_val,
                    'E_Voigt': E_v,
                    'E_Reuss': E_r,
                    'E_MoriTanaka': E_mt,
                    'E11_Numerico': E11,
                    'E22_Numerico': E22,
                    'E33_Numerico': E33,
                    'E_Medio_Numerico': E_num_avg,
                    'G12_Num': G12,
                    'G13_Num': G13,
                    'G23_Num': G23,
                    'nu12_Num': nu12,
                    'nu13_Num': nu13,
                    'nu23_Num': nu23,
                    'Anisotropia_E(%)': anisotropia_E
                })
                print(f"[OK] Dados extraídos e processados: Ec = {Ec_val} | vf = {vf_percent}%")

    # 4. Gerar o ficheiro CSV
    if dados_totais:
        df = pd.DataFrame(dados_totais)
        
        # Ordenar por Inclusão e depois por Fração Volumétrica
        df = df.sort_values(by=['Ec_Inclusao', 'Vc_fracao']).reset_index(drop=True)
        
        nome_csv = "Comparativo_Homogeneizacao_RVE.csv"
        df.to_csv(nome_csv, index=False)
        print("\n" + "="*65)
        print(f"[SUCESSO] Ficheiro gerado: {nome_csv}")
        print("Abra o ficheiro no Excel para gerar os gráficos de calibração!")
        print("="*65)

if __name__ == "__main__":
    path = "/home/viny/Documentos/workspace/JiveApps/fe2/analise_linear_AL"
    minera_e_compara(path)