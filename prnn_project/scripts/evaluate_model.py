import matplotlib.pyplot as plt
import numpy as np

def aplicar_media_movel(dados, janela=50):
    """Aplica média móvel simples (SMA)."""
    if janela < 2:
        return dados
    return np.convolve(dados, np.ones(janela)/janela, mode='valid')

def plot_comparacao_espectro(lista_pontos, tipo_erro='RMSE', sigma_y=500.0, suavizar=False, janela=50, topologia='sparse'):
    """
    Plota a convergência comparando múltiplos números de pontos materiais.
    """
    # Configuração global de fontes (Padrão artigo científico/dissertação)
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11,          
        "axes.labelsize": 12,     
        "xtick.labelsize": 10,    
        "ytick.labelsize": 10,    
        "legend.fontsize": 10
    })

    # Formato Widescreen (10 x 5.5) ideal para apresentações e relatórios
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    # Paletas de cores distintas para cada quantidade de pontos
    paletas = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]

    for i, pts in enumerate(lista_pontos):
        # Mapeia dinamicamente o nome do arquivo conforme a topologia escolhida
        sufixo = topologia.lower()
        csv_path = f"trained_models/historico_{pts}_{sufixo}.csv"
        
        try:
            dados = np.loadtxt(csv_path, delimiter=',', comments='#')
            train_data = np.sqrt(dados[:, 0])
            val_data = np.sqrt(dados[:, 1])
            
            if tipo_erro == 'RELATIVO':
                train_data = (train_data / sigma_y) * 100
                val_data = (val_data / sigma_y) * 100
            
            paleta = paletas[i % len(paletas)]
            
            if suavizar:
                t_plot = aplicar_media_movel(train_data, janela)
                v_plot = aplicar_media_movel(val_data, janela)
                x_axis = np.arange(len(t_plot))
                
                # Linhas mais grossas para destaque visual
                ax.plot(x_axis, t_plot, color=paleta(0.8), linestyle='-', 
                        linewidth=2.5, label=f'{pts} pts ({topologia.upper()}) - Train')
                ax.plot(x_axis, v_plot, color=paleta(0.4), linestyle='--', 
                        linewidth=2.5, label=f'{pts} pts ({topologia.upper()}) - Val')
            else:
                ax.plot(train_data, color=paleta(0.8), linestyle='-', 
                        linewidth=1.5, alpha=0.6, label=f'{pts} pts ({topologia.upper()}) - Train')
                ax.plot(val_data, color=paleta(0.4), linestyle='--', 
                        linewidth=1.5, label=f'{pts} pts ({topologia.upper()}) - Val')
            
        except FileNotFoundError:
            print(f"Aviso: Arquivo '{csv_path}' não encontrado. Verifique se o treino foi executado.")

    # Configurações do Eixo e Escala Logarítmica
    ax.set_yscale('log')
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("RMSE (MPa)" if tipo_erro == 'RMSE' else "Relative Error (%)", fontweight='bold')
    
    # ATENÇÃO: Em escala logarítmica, o limite inferior (ymin) DEVE ser estritamente maior que 0 (ex: 1e-2 ou 1e-1)
    # Retiramos o limite '-5' para evitar o travamento do Matplotlib.
    #ax.set_ylim(bottom=None, top=1000) 
    
    ax.set_title(f"Convergence Analysis: Multiscale Material Points ({topologia.upper()})", fontsize=14, pad=15, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    
    # Legenda organizada em colunas no topo
    ax.legend(ncol=3, loc='upper right', frameon=True, edgecolor='black')
    
    # Salva a figura em alta resolução para a dissertação/slides
    output_filename = f"trained_models/comparacao_convergencia_{topologia}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Gráfico comparativo salvo com sucesso em: {output_filename}")
    
    plt.show()

# --- Exemplos de Uso ---
if __name__ == "__main__":
    # Exemplo comparando 1, 3 e 5 pontos com suavização (janela de 50 épocas)
    plot_comparacao_espectro(
        lista_pontos=[1, 3, 5], 
        tipo_erro='RMSE', 
        suavizar=True, 
        janela=50, 
        topologia='dense' # Pode trocar para 'dense' se preferir
    )