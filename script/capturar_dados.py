import os, glob, time, csv, shutil, subprocess, re
import numpy as np

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================
FAST_EXE = "/home/viny/Documentos/workspace/fast/FASTv2.4.6/fast"
os.environ.update({k: "15" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]})

# ==========================================================
# FUNÇÃO: Processar Micromecânica Linear e Não-Linear
# ==========================================================
def processar_micromecanica(dat_file):
    base_name = os.path.splitext(os.path.basename(dat_file))[0]
    fast_dir = os.path.dirname(FAST_EXE)
    dat_temp = os.path.join(fast_dir, f"{base_name}.dat")
    
    print(f"\n➤ Processando: {base_name}.dat")
    shutil.copy2(dat_file, dat_temp)
    
    # 1. Executa o solver silenciosamente e captura todo o texto de saída
    t_ini = time.time()
    proc = subprocess.run(
        [FAST_EXE], cwd=fast_dir, input=f"{base_name}\n", 
        capture_output=True, text=True
    )
    tempo = time.time() - t_ini
    
    # 2. Limpeza e resgate do .pos
    if os.path.exists(dat_temp): os.remove(dat_temp)
    for arq in glob.glob(os.path.join(fast_dir, f"{base_name}.*")):
        shutil.move(arq, os.path.join(os.path.dirname(dat_file), os.path.basename(arq)))

    # Dicionário base (se der erro, preenche com ERRO)
    res = {
        "Arquivo": base_name, "E1_GPa": "ERRO", "E2_GPa": "ERRO", "E3_GPa": "ERRO", 
        "G12_GPa": "ERRO", "G13_GPa": "ERRO", "G23_GPa": "ERRO",
        "nu12": "ERRO", "nu13": "ERRO", "nu23": "ERRO", "Sigma_XX": "ERRO", "Tempo_s": round(tempo, 2)
    }

    if proc.returncode != 0:
        print(f"  ❌ Erro do FAST: {proc.stderr.strip()}")
        return res

    # Prepara o texto trocando vírgulas por pontos
    out = proc.stdout.replace(',', '.')

    # 3. Extração Linear: Matriz [CM]
    if "[CM]" in out:
        bloco_cm = out.split("[CM]")[1]
        # Pega todos os números reais após a tag [CM]
        floats = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", bloco_cm)]
        try:
            if len(floats) >= 36: # Problema 3D
                S = np.linalg.inv(np.array(floats[:36]).reshape(6,6))
                res.update({
                    "E1_GPa": (1/S[0,0])/1000, "E2_GPa": (1/S[1,1])/1000, "E3_GPa": (1/S[2,2])/1000,
                    "G12_GPa": (1/S[3,3])/1000, "G13_GPa": (1/S[4,4])/1000, "G23_GPa": (1/S[5,5])/1000,
                    "nu12": -S[0,1]/S[0,0], "nu13": -S[0,2]/S[0,0], "nu23": -S[1,2]/S[1,1]
                })
            elif len(floats) >= 9: # Problema 2D (Estado Plano)
                S = np.linalg.inv(np.array(floats[:9]).reshape(3,3))
                res.update({
                    "E1_GPa": (1/S[0,0])/1000, "E2_GPa": (1/S[1,1])/1000, "E3_GPa": "N/A",
                    "G12_GPa": (1/S[2,2])/1000, "G13_GPa": "N/A", "G23_GPa": "N/A",
                    "nu12": -S[0,1]/S[0,0], "nu13": "N/A", "nu23": "N/A"
                })
            print("  ✔ Matriz Constitutiva [CM] extraída e invertida.")
        except Exception as e:
            print(f"  ❌ Erro matemático na matriz: {e}")

    # 4. Extração Não-Linear: Tensões Médias {AvrStr}
    if "{AvrStr}" in out:
        bloco_str = out.split("{AvrStr}")[1]
        floats = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", bloco_str)]
        if len(floats) >= 1:
            res["Sigma_XX"] = floats[0]
            print("  ✔ Tensões médias não-lineares ({AvrStr}) extraídas.")

    return res

# ==========================================================
# BLOCO PRINCIPAL (Execução e CSV)
# ==========================================================
if __name__ == '__main__':
    pasta_alvo = input("\nDigite o caminho da pasta com os .dat (Enter = atual): ").strip() or os.getcwd()
    dat_files = sorted(glob.glob(os.path.join(pasta_alvo, "*.dat")))
    
    if not dat_files:
        print(f"❌ Nenhum .dat encontrado em {pasta_alvo}")
        exit()

    print(f"\n🚀 PROCESSANDO {len(dat_files)} ARQUIVOS...")
    resultados = [processar_micromecanica(df) for df in dat_files]

    # Salva o CSV
    csv_path = os.path.join(pasta_alvo, "micromecanica_resultados.csv")
    colunas = ["Arquivo", "E1_GPa", "E2_GPa", "E3_GPa", "G12_GPa", "G13_GPa", "G23_GPa", "nu12", "nu13", "nu23", "Sigma_XX", "Tempo_s"]
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n✅ Concluído! Resultados salvos em: {csv_path}\n")