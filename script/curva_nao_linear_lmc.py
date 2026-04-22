import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extrair_dados_fast(nome_ficheiro):
    """
    Algoritmo universal de extração para o FAST (lcm).
    Deteta automaticamente se os resultados nodais são 2D (3 tensores) ou 3D (6 tensores).
    """
    if not os.path.exists(nome_ficheiro):
        print(f"Erro: O ficheiro '{nome_ficheiro}' não foi encontrado.")
        return None
        
    with open(nome_ficheiro, 'r') as f:
        conteudo = f.read()

    # Separa o ficheiro em blocos de Passos (Steps)
    passos_texto = conteudo.split('%RESULT.CASE.STEP\n')[1:] 
    dados = []

    for i, texto_passo in enumerate(passos_texto):
        passo_num = i + 1
        
        # 1. Fator de Carga (LF)
        match_lf = re.search(r'%RESULT\.CASE\.STEP\.FACTOR\n([eE\+\-\.\d]+)', texto_passo)
        lf = float(match_lf.group(1)) if match_lf else 0.0
        
        # 2. Deformação (Deslocamento X do Nó 2)
        match_disp = re.search(r'^\s*2\s+([eE\+\-\.\d]+)', texto_passo, re.MULTILINE)
        def_xx = float(match_disp.group(1)) if match_disp else 0.0
        
        # 3. Extração Dinâmica de Tensões
        padrao_dados = r'%RESULT\.CASE\.STEP\.ELEMENT\.NODAL\.SCALAR\.DATA\n\d+\n\d+\n(.*?)(?=\n\n|\Z|%RESULT)'
        match_dados = re.search(padrao_dados, texto_passo, re.DOTALL)
        
        linha_dados = {'Passo': passo_num, 'LF': lf, 'Deformacao_XX': def_xx}
        
        if match_dados:
            linhas_nos = match_dados.group(1).strip().split('\n')
            linhas_nos = [l for l in linhas_nos if l.strip()] # Limpa linhas vazias
            
            if linhas_nos:
                # O segredo: contar as colunas da primeira linha para saber a dimensão
                num_cols = len(linhas_nos[0].split())
                
                if num_cols == 3:
                    chaves = ['Tensao_XX', 'Tensao_YY', 'Tensao_XY']
                elif num_cols == 6:
                    chaves = ['Tensao_XX', 'Tensao_YY', 'Tensao_ZZ', 'Tensao_XY', 'Tensao_XZ', 'Tensao_YZ']
                else:
                    chaves = [f'Tensao_C{k}' for k in range(num_cols)]
                    
                somas = {chave: 0.0 for chave in chaves}
                contador = 0
                
                # Somatório para cálculo da média dos nós do elemento
                for linha in linhas_nos:
                    colunas = linha.split()
                    if len(colunas) == num_cols:
                        for k, chave in enumerate(chaves):
                            somas[chave] += float(colunas[k])
                        contador += 1
                
                # Regista as médias no dicionário do passo atual
                if contador > 0:
                    for chave in chaves:
                        linha_dados[chave] = somas[chave] / contador
                        
        dados.append(linha_dados)

    if not dados:
        return None

    # Adicionar o passo 0 (origem) dinamicamente com base nas chaves detetadas
    chaves_tensao = [k for k in dados[0].keys() if k.startswith('Tensao_')]
    linha_zero = {'Passo': 0, 'LF': 0.0, 'Deformacao_XX': 0.0}
    for chave in chaves_tensao:
        linha_zero[chave] = 0.0
    dados.insert(0, linha_zero)

    return pd.DataFrame(dados)

def gerar_relatorio_fast(nome_ficheiro):
    print(f"A analisar o ficheiro estrutural: {nome_ficheiro}...")
    df = extrair_dados_fast(nome_ficheiro)
    
    if df is not None and not df.empty:
        nome_csv = nome_ficheiro.replace('.pos', '_Resultados.csv')
        df.to_csv(nome_csv, index=False)
        print(f"[OK] Dados exportados com sucesso para: {nome_csv}")
        
        plt.figure(figsize=(10, 6))
        
        # Plot Principal (Tensão de Tração)
        plt.plot(df['Deformacao_XX'], df['Tensao_XX'], marker='s', 
                 linestyle='-', color='navy', linewidth=2.5, label='$\Sigma_{xx}$ (Tração)')
        """
        # Se for 3D, adiciona as tensões transversais para auditoria técnica
        if 'Tensao_YY' in df.columns and 'Tensao_ZZ' in df.columns:
            plt.plot(df['Deformacao_XX'], df['Tensao_YY'], marker='^', markersize=5,
                     linestyle='--', color='darkorange', alpha=0.7, label='$\Sigma_{yy}$ (Lateral Y)')
            plt.plot(df['Deformacao_XX'], df['Tensao_ZZ'], marker='v', markersize=5,
                     linestyle='-.', color='forestgreen', alpha=0.7, label='$\Sigma_{zz}$ (Lateral Z)')
        """
        # Linha de referência física
        plt.axhline(y=478.05, color='gray', linestyle=':', alpha=0.8, label='Escoamento ($S_y$)')

        dimensao_str = "3D (BRICK)" if 'Tensao_ZZ' in df.columns else "2D (Q4/Q8)"
        plt.title(f'Validação Estrutural {dimensao_str}: Ensaio Uniaxial', fontsize=14, fontweight='bold')
        plt.xlabel('Deformação Macroscópica $\epsilon_{xx}$ (m/m)', fontsize=12)
        plt.ylabel('Tensão Equivalente (MPa)', fontsize=12) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.xlim(left=0)
        
        nome_grafico = nome_ficheiro.replace('.pos', '_Grafico.png')
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico desenhado: {nome_grafico}\n")
        plt.show()

# ==========================================
# PAINEL DE CONTROLO
# ==========================================
FICHEIRO_POS = "Aluminio_brick8.pos" # O script ajusta-se automaticamente a 2D ou 3D

gerar_relatorio_fast(FICHEIRO_POS)