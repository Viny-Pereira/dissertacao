import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extrair_dados_fast(nome_arquivo):
    """
    Lê o ficheiro .pos do FAST e extrai a curva Tensão-Deformação Macroscópica.
    """
    with open(nome_arquivo, 'r') as f:
        conteudo = f.read()

    # 1. Encontrar a deformação macroscópica base (eixo X)
    match_macro = re.search(r'%MICRO\.MODEL\.MACRO\.STRAINS\n([eE\+\-\.\d\s]+)', conteudo)
    if not match_macro:
        print("Erro: Bloco %MICRO.MODEL.MACRO.STRAINS não encontrado.")
        return None
    
    # Pega o primeiro valor numérico (eps_xx)
    valores_macro = match_macro.group(1).split()
    eps_base_xx = float(valores_macro[0])

    # 2. Dividir o arquivo em blocos de Passos (Steps)
    # Ignora o índice 0 pois é o cabeçalho antes do passo 1
    passos_texto = conteudo.split('%RESULT.CASE.STEP\n')[1:] 
    
    dados = []
    # Adicionamos a origem para o gráfico começar do zero (0,0)
    dados.append({'Passo': 0, 'Deformacao_XX': 0.0, 'Tensao_XX_MPa': 0.0})

    for i, texto_passo in enumerate(passos_texto):
        passo_num = i + 1
        
        # A) Pegar o Fator do Passo (Step Factor)
        match_factor = re.search(r'%RESULT\.CASE\.STEP\.FACTOR\n([eE\+\-\.\d]+)', texto_passo)
        if match_factor:
            fator = float(match_factor.group(1))
        else:
            fator = 1.0
            
        deformacao_atual = eps_base_xx * fator
        
        # B) Pegar as tensões nodais para fazer a média
        # Procura o bloco NODAL.SCALAR.DATA e pega tudo até à próxima tag ou fim do ficheiro
        padrao_dados = r'%RESULT\.CASE\.STEP\.NODAL\.SCALAR\.DATA\n(\d+)\n(.*?)(?=\n\n|\Z|%RESULT)'
        match_dados = re.search(padrao_dados, texto_passo, re.DOTALL)
        
        if match_dados:
            linhas_nos = match_dados.group(2).strip().split('\n')
            
            soma_tensao = 0.0
            contador = 0
            
            for linha in linhas_nos:
                colunas = linha.split()
                # A coluna 0 é o ID do nó, a coluna 1 é STRESS_XX
                if len(colunas) >= 2:
                    soma_tensao += float(colunas[1]) 
                    contador += 1
            
            # Cálculo da média da tensão no RVE
            tensao_media_pa = soma_tensao / contador if contador > 0 else 0.0
            tensao_media_mpa = tensao_media_pa / 1e6 # Converter Pascal para MegaPascal
            
            dados.append({
                'Passo': passo_num,
                'Deformacao_XX': deformacao_atual,
                'Tensao_XX_MPa': tensao_media_mpa
            })

    return pd.DataFrame(dados)

def gerar_relatorio_rve(nome_arquivo):
    if not os.path.exists(nome_arquivo):
        print(f"Erro: O ficheiro '{nome_arquivo}' não foi encontrado na pasta.")
        return

    print(f"Processando arquivo: {nome_arquivo}...")
    df = extrair_dados_fast(nome_arquivo)
    
    if df is not None and not df.empty:
        # Exportar CSV
        nome_csv = nome_arquivo.replace('.pos', '.csv')
        df.to_csv(nome_csv, index=False)
        print(f"[OK] Dados exportados com sucesso para: {nome_csv}")
        
        # Gerar o Gráfico
        plt.figure(figsize=(9, 6))
        plt.plot(df['Deformacao_XX'], df['Tensao_XX_MPa'], marker='o', 
                 linestyle='-', color='darkred', linewidth=2, label='Homogeneização FAST')
        
        # Estilização profissional para a dissertação
        plt.title('Curva Tensão-Deformação Macroscópica do RVE', fontsize=14, fontweight='bold')
        plt.xlabel(r'Deformação Macroscópica $\epsilon_{xx}$ (m/m)', fontsize=12)
        plt.ylabel(r'Tensão Equivalente $\Sigma_{xx}$ (MPa)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=11)
        
        # Ajustar limites do gráfico para ficar limpo
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        # Salvar imagem
        nome_grafico = nome_arquivo.replace('.pos', '.png')
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico salvo com sucesso como: {nome_grafico}")
        
        # Mostrar na tela
        plt.show()

# ==========================================
# UTILIZAÇÃO DO SCRIPT
# ==========================================
# Coloque o nome do seu arquivo aqui em baixo:
nome_do_arquivo = r"C:\Users\Viny Pereira\Desktop\workspace\dissertacao\FASTv2.5.0\MieheKochEx3.pos"

gerar_relatorio_rve(nome_do_arquivo)