import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados
arquivo_excel = 'dados.xlsx'
df_t10 = pd.read_excel(arquivo_excel, sheet_name='T10').sort_values(by='num_nos')
df_t4 = pd.read_excel(arquivo_excel, sheet_name='T4').sort_values(by='num_nos')

# Configurações estéticas
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 2. Criar a figura e o Eixo Y Principal (Módulo)
fig, ax1 = plt.subplots(figsize=(10, 8))

# Plotando o Módulo (E_avg) no eixo da ESQUERDA (Linhas Contínuas)
linha1 = ax1.plot(df_t4['num_nos'], df_t4['E_avg'], color='#d9534f', linestyle='-', marker='s', linewidth=2, label='T4 - Módulo')
linha2 = ax1.plot(df_t10['num_nos'], df_t10['E_avg'], color='#5bc0de', linestyle='-', marker='o', linewidth=2, label='T10 - Módulo')

ax1.set_xlabel('Número de Nós', fontweight='bold')
ax1.set_ylabel('Módulo Efetivo - $E_{avg}$ (GPa)', fontweight='bold')
ax1.set_xscale('log') # Escala Log no eixo X ajuda a ver malhas muito grandes
ax1.grid(True, linestyle='--', alpha=0.6)

# 3. Criar o Eixo Y Secundário (Tempo)
ax2 = ax1.twinx()

# Plotando o Tempo (Tempo_s) no eixo da DIREITA (Linhas Tracejadas)
linha3 = ax2.plot(df_t4['num_nos'], df_t4['Tempo_s'], color='#d9534f', linestyle='--', marker='^', linewidth=1.5, alpha=0.8, label='T4 - Tempo')
linha4 = ax2.plot(df_t10['num_nos'], df_t10['Tempo_s'], color='#5bc0de', linestyle='--', marker='^', linewidth=1.5, alpha=0.8, label='T10 - Tempo')

ax2.set_ylabel('Tempo Computacional (s)', fontweight='bold')
ax2.set_yscale('log') # Escala Log no Y direito para o tempo não esmagar a curva do T4

# 4. Juntar as legendas dos dois eixos numa única caixa
linhas = linha1 + linha2 + linha3 + linha4
labels = [l.get_label() for l in linhas]
ax1.legend(linhas, labels, loc='upper center', framealpha=1.0, edgecolor='black')

plt.title('Convergência do Módulo vs Custo Computacional (T4 vs T10)', pad=15)
plt.tight_layout()

# 5. Salvar e Mostrar
nome_imagem = 'Grafico_Convergencia_Eixo_Duplo.png'
plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
print(f"Gráfico salvo como: {nome_imagem}")

plt.show()