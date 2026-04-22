import os
import glob
import time
import csv
import numpy as np
import shutil

from dissertacao.script.multiscale import run_fast

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
PASTA = "processar"
FAST_EXE = r"C:\Users\Viny Pereira\Desktop\workspace\dissertacao\FASTv2.5.1\fast.exe"

os.makedirs(PASTA, exist_ok=True)

# Pega o caminho absoluto da pasta processar e do diretório atual
pasta_trabalho = os.path.abspath(PASTA)
pasta_original = os.getcwd()

# ==========================================================
# FUNÇÃO: Extrair parâmetros do nome do arquivo
# ==========================================================
def extrair_parametros(nome):
    """
    Extrai vf, mesh, ordem, num_nos e num_elementos de nomes como:
    RVE_vf10_m0.05_ord1_N1405_E8302_FAST
    """
    try:
        partes = nome.split("_")
        vf = float(partes[1].replace("vf", ""))
        mesh = float(partes[2].replace("m", ""))
        ordem = int(partes[3].replace("ord", ""))
        
        num_nos = None
        num_elementos = None
        
        # Procura as partes que indicam Nós (N) e Elementos (E)
        for p in partes:
            if p.startswith("N") and p[1:].isdigit():
                num_nos = int(p[1:])
            elif p.startswith("E") and p[1:].isdigit():
                num_elementos = int(p[1:])
                
        return vf, mesh, ordem, num_nos, num_elementos
    except Exception as e:
        print(f"Erro ao extrair parâmetros de '{nome}': {e}")
        return None, None, None, None, None

# ==========================================================
# ETAPA 2: RODAR FAST
# ==========================================================
print("\n" + "="*60)
print(" ETAPA 2: PROCESSANDO .DAT NO FAST")
print("="*60)

dat_files = glob.glob(os.path.join(PASTA, "*.dat"))
dat_files.sort()

resultados = []

for dat_file in dat_files:

    base_name = os.path.splitext(os.path.basename(dat_file))[0]
    print(f"\nProcessando: {base_name}")

    # Extrai TUDO direto do nome!
    vf, mesh, ordem, num_nos, num_elementos = extrair_parametros(base_name)

    try:
        # Descobre a "casa" do FAST
        fast_dir = os.path.dirname(FAST_EXE)
        
        # Arquivos de origem e destino
        dat_original = os.path.join(pasta_trabalho, base_name + ".dat")
        dat_temp = os.path.join(fast_dir, base_name + ".dat")

        # 1. Copia o .dat temporariamente para a pasta do FAST
        shutil.copy2(dat_original, dat_temp)

        t_ini = time.time()

        # 2. Roda o solver
        cm_matrix, sigma_vec = run_fast(FAST_EXE, base_name, timeout=12000)

        tempo_exec = time.time() - t_ini

        # 3. Limpa a sujeira
        if os.path.exists(dat_temp):
            os.remove(dat_temp)

        # 4. Traz o resultado (.pos) para a pasta 'processar'
        pos_temp = os.path.join(fast_dir, base_name + ".pos")
        pos_final = os.path.join(pasta_trabalho, base_name + ".pos")
        if os.path.exists(pos_temp):
            shutil.move(pos_temp, pos_final)

        # ==========================
        # PROPRIEDADES EFETIVAS
        # ==========================
        S = np.linalg.inv(cm_matrix)

        E1 = 1.0 / S[0, 0]
        E2 = 1.0 / S[1, 1]
        E3 = 1.0 / S[2, 2]

        nu12 = -S[0, 1] / S[0, 0]
        nu13 = -S[0, 2] / S[0, 0]
        nu23 = -S[1, 2] / S[1, 1]

        G12 = 1.0 / S[3, 3]
        G13 = 1.0 / S[4, 4]
        G23 = 1.0 / S[5, 5]

        # MPa → GPa
        conv = 1.0 / 1000.0
        E1 *= conv; E2 *= conv; E3 *= conv
        G12 *= conv; G13 *= conv; G23 *= conv

        resultados.append({
            "Malha": base_name,
            "VF": vf,
            "mesh": mesh,
            "ordem": ordem,
            "num_nos": num_nos,
            "num_elementos": num_elementos,
            "E1_GPa": E1,
            "E2_GPa": E2,
            "E3_GPa": E3,
            "G12_GPa": G12,
            "G13_GPa": G13,
            "G23_GPa": G23,
            "nu12": nu12,
            "nu13": nu13,
            "nu23": nu23,
            "Sigma_XX": sigma_vec[0] if sigma_vec is not None else None,
            "Tempo_s": tempo_exec
        })

        print(f"✔ E1 = {E1:.4f} GPa | Tempo = {tempo_exec:.2f}s | Nós: {num_nos} | Elementos: {num_elementos}")

    except Exception as e:
        print(f"Erro no FAST para a malha {base_name}: {e}")
        if 'dat_temp' in locals() and os.path.exists(dat_temp):
            os.remove(dat_temp)

        resultados.append({
            "Malha": base_name, "VF": vf, "mesh": mesh, "ordem": ordem,
            "num_nos": num_nos, "num_elementos": num_elementos,
            "E1_GPa": "ERRO", "E2_GPa": "ERRO", "E3_GPa": "ERRO",
            "G12_GPa": "ERRO", "G13_GPa": "ERRO", "G23_GPa": "ERRO",
            "nu12": "ERRO", "nu13": "ERRO", "nu23": "ERRO",
            "Sigma_XX": "ERRO", "Tempo_s": "ERRO"
        })

# ==========================================================
# ETAPA 3: SALVAR CSV
# ==========================================================
print("\n" + "="*60)
print(" ETAPA 3: SALVANDO CSV")
print("="*60)

csv_path = os.path.join(PASTA, "resultados_convergencia.csv")

colunas = [
    "Malha", "VF", "mesh", "ordem", "num_nos", "num_elementos",
    "E1_GPa","E2_GPa","E3_GPa", "G12_GPa","G13_GPa","G23_GPa",
    "nu12","nu13","nu23", "Sigma_XX", "Tempo_s"
]

with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=colunas)
    writer.writeheader()
    writer.writerows(resultados)

print(f"\n✔ CSV salvo em: {csv_path}")

print("\n" + "="*60)
print(" PROCESSO COMPLETO FINALIZADO!")
print("="*60)