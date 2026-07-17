import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def plot_metric_comparison_combined(excel_path, metrica_alvo='L1Error'):
    """
    Lê uma planilha Excel com abas 'dense' e 'sparse' e plota a evolução 
    da métrica escolhida (ex: 'L1Error' ou 'MSE') para 'Unloading' e 'GP'.
    """
    # Configuração de estilo acadêmico (STIX)
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10, 
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Leitura dos dados
    df_dense = pd.read_excel(excel_path, sheet_name='dense')
    df_sparse = pd.read_excel(excel_path, sheet_name='sparse')

    # Extração das colunas de eixo X independentes
    pts_dense = df_dense['NumMatPnt']
    pts_sparse = df_sparse['NumMatPnt']
    
    # Construção dos nomes das colunas dinamicamente
    col_unloading = f'{metrica_alvo}-Unloading'
    col_gp = f'{metrica_alvo}-GP'

    # Extração das colunas de eixo Y (Sem multiplicar por 100, pois não é percentual)
    dense_unloading = df_dense[col_unloading]
    dense_gp = df_dense[col_gp]
    
    sparse_unloading = df_sparse[col_unloading]
    sparse_gp = df_sparse[col_gp]

    # Criação da figura
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Definindo a paleta de cores 
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
    
    # Ajusta o label do eixo Y dependendo da métrica escolhida
    if metrica_alvo == 'MSE':
        ax.set_ylabel(f'{metrica_alvo} [MPa²]') 
    else:
        ax.set_ylabel(f'{metrica_alvo} [MPa]')

    ax.set_title(f'Generalization Performance: {metrica_alvo}')

    # Ajuste dos "ticks" para mostrar apenas números inteiros no eixo X
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(which='major', length=5, direction='in')
    ax.tick_params(which='minor', length=3, direction='in')

    # Grid limpo
    ax.grid(True, which='major', linestyle=':', alpha=0.6)

    # Legenda e Layout
    ax.legend(frameon=True, edgecolor='black', loc='upper right', ncol=2)
    fig.tight_layout()

    # Mostra a figura
    plt.show()

if __name__ == "__main__":
    caminho_planilha = "comparativo_dados_rede__.xlsx" 
    # Você pode testar mudando 'L1Error' para 'MSE' ou 'L1Loss'
    plot_metric_comparison_combined(caminho_planilha, metrica_alvo='L1Error')