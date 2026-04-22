import os
import shutil
import subprocess
import glob

# ==========================================
# 1. CONFIGURAÇÕES DA ANÁLISE
# ==========================================
modulos_Ei = [380,385,390,395,400,405,410]          # Valores para o Módulo da Inclusão (Ei)
fracoes_vf = [10, 20, 30]   # Frações volumétricas das malha
nome_template = "template.pro" # O ficheiro com as etiquetas {{}}
nome_pro_real = "rve.pro"  # O ficheiro que o go.sh vai ler (MUDE se o seu go.sh ler outro nome!)

# ==========================================
# 2. MOTOR DE AUTOMAÇÃO
# ==========================================
for Ei in modulos_Ei:
    for vf in fracoes_vf:
        
        # ---------------------------------------------------------
        # ATENÇÃO: Defina aqui o padrão de nome das suas malhas!
        # Se os seus arquivos são rve_10.msh, rve_20.msh, mantenha assim:
        nome_malha = f"RVE_vf{vf}_m0.1_ord2.msh" 
        # ---------------------------------------------------------
        
        print("\n" + "="*55)
        print(f" INICIANDO ANÁLISE: Inclusão (Ei) = {Ei} | VF = {vf}%")
        print(f" Lendo malha: {nome_malha}")
        print("="*55)

        # A) Ler o template base
        if not os.path.exists(nome_template):
            print(f"[ERRO CRÍTICO] O ficheiro {nome_template} não foi encontrado na pasta!")
            break

        with open(nome_template, 'r') as f:
            conteudo = f.read()

        # B) Injetar os valores reais substituindo as etiquetas
        conteudo = conteudo.replace("{{MODULO_INCLUSAO}}e+3", str(Ei))
        conteudo = conteudo.replace("{{NOME_MALHA}}", nome_malha)

        # C) Escrever o arquivo .pro temporário para o Jive ler
        with open(nome_pro_real, 'w') as f:
            f.write(conteudo)

        # D) Executar o solver Jive através do seu script go.sh
        print("[Sistema] Executando ./go.sh ... (Isto pode demorar alguns minutos)")
        try:
            # Roda o comando e bloqueia o Python até a análise do Jive terminar
            subprocess.run(["./go.sh"], check=True)
        except subprocess.CalledProcessError:
            print(f"\n[ALERTA] O solver falhou ou foi interrompido para Ei={Ei}, vf={vf}.")
            print("Pulo para a próxima análise...\n")
            continue 

        # E) Criar a hierarquia de pastas exata que você pediu (Ex: E_400/vf_10)
        pasta_destino = os.path.join(f"E_{Ei}", f"vf_{vf}")
        os.makedirs(pasta_destino, exist_ok=True)

        # F) Encontrar todos os ficheiros de saída (.out e .log)
        arquivos_gerados = glob.glob("*.out") + glob.glob("*.log")

        if not arquivos_gerados:
            print("[AVISO] Nenhum ficheiro .out ou .log foi gerado. Verifique se o Jive rodou corretamente.")
        else:
            # G) Mover os resultados (MicroPBC.out, .log, etc) para a pasta organizada
            for arq in arquivos_gerados:
                caminho_origem = arq
                caminho_destino = os.path.join(pasta_destino, arq)
                
                # Se for re-executar, apaga o antigo antes de mover o novo
                if os.path.exists(caminho_destino):
                    os.remove(caminho_destino)
                    
                shutil.move(caminho_origem, caminho_destino)

            print(f"[OK] Análise concluída! {len(arquivos_gerados)} ficheiros movidos para: {pasta_destino}")

print("\n" + "="*55)
print("[SUCESSO TOTAL] O lote completo de simulações foi finalizado!")
print("="*55 + "\n")