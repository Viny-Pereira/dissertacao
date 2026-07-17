import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def plot_robust_comparison_combined(excel_path):
    """
    Lê uma planilha Excel com abas 'dense' e 'sparse' e plota a evolução 
    do Erro Relativo Robusto para os conjuntos 'Unloading' e 'GP' simultaneamente.
    """
    # Configuração de estilo acadêmico (Computer Modern / STIX)
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12, # Fonte levemente menor para caber 4 itens
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Leitura dos dados
    df_dense = pd.read_excel(excel_path, sheet_name='dense')
    df_sparse = pd.read_excel(excel_path, sheet_name='sparse')

    # Extração das colunas de eixo X independentes
    pts_dense = df_dense['NumMatPnt']
    pts_sparse = df_sparse['NumMatPnt']
    
    # Extração das colunas de eixo Y (Convertendo para %)
    dense_unloading = df_dense['Robust-Unloading'] * 100
    dense_gp = df_dense['Robust-GP'] * 100
    
    sparse_unloading = df_sparse['Robust-Unloading'] * 100
    sparse_gp = df_sparse['Robust-GP'] * 100

    # Criação da figura
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Definindo a paleta de cores (Azul para Dense, Vermelho para Sparse)
    cor_dense = '#5DA5DA'
    cor_sparse = '#F15854'

    # 1. Curvas de Unloading (Linhas Sólidas, Marcadores Preenchidos)
    ax.plot(pts_dense, dense_unloading, 
            marker='s', markersize=6, linestyle='-', linewidth=1.8, 
            color=cor_dense, label='Dense (Unloading)')
    
    ax.plot(pts_sparse, sparse_unloading, 
            marker='o', markersize=6, linestyle='-', linewidth=1.8, 
            color=cor_sparse, label='Sparse (Unloading)')

    # 2. Curvas de GP (Linhas Tracejadas, Marcadores Vazios)
    ax.plot(pts_dense, dense_gp, 
            marker='s', markersize=6, linestyle='--', linewidth=1.8, 
            color=cor_dense, markerfacecolor='white', label='Dense (GP)')
    
    ax.plot(pts_sparse, sparse_gp, 
            marker='o', markersize=6, linestyle='--', linewidth=1.8, 
            color=cor_sparse, markerfacecolor='white', label='Sparse (GP)')

    # Formatação dos eixos
    ax.set_xlabel('Number of Fictitious Material Points')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Generalization Performance: Dense vs. Sparse Architectures')

    # Ajuste dos "ticks" para mostrar apenas números inteiros no eixo X
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(which='major', length=5, direction='in')
    ax.tick_params(which='minor', length=3, direction='in')

    # Grid limpo
    ax.grid(True, which='major', linestyle=':', alpha=0.6)

    # Legenda e Layout (2 colunas para organizar melhor)
    ax.legend(frameon=True, edgecolor='black', loc='upper right', ncol=2)
    fig.tight_layout()

    # Mostra a figura
    plt.show()

if __name__ == "__main__":
    caminho_planilha = "comparativo_dados_rede__.xlsx" 
    plot_robust_comparison_combined(caminho_planilha)