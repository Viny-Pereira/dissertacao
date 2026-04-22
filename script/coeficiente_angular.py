import pandas as pd
import numpy as np

# 1. Carrega os dados e ordena pelo número de nós
df_t10_completo = pd.read_excel('dados.xlsx', sheet_name='T10').sort_values(by='num_nos')
df_t4_completo = pd.read_excel('dados.xlsx', sheet_name='T4').sort_values(by='num_nos')

# ====================================================================
# MODIFICAÇÃO: Remove a última linha (a malha com maior número de nós)
# O comando .iloc[:-1] pega da primeira linha até a penúltima
# ====================================================================
df_t10 = df_t10_completo.iloc[-2:] 
df_t4 = df_t4_completo.iloc[-2:] 

print(f"Malha final removida. Analisando {len(df_t4)} malhas para T4 e {len(df_t10)} malhas para T10.\n")

# ====================================================================
# OPÇÃO 1: Coeficiente Angular no Gráfico Log-Log (Complexidade)
# ====================================================================
coef_log_t4, _ = np.polyfit(np.log10(df_t4['num_nos']), np.log10(df_t4['Tempo_s']), 1)
coef_log_t10, _ = np.polyfit(np.log10(df_t10['num_nos']), np.log10(df_t10['Tempo_s']), 1)

print(f"--- Complexidade Computacional (Inclinação Log-Log) ---")
print(f"T4 : Tempo cresce a uma taxa de O(N^{coef_log_t4:.2f})")
print(f"T10: Tempo cresce a uma taxa de O(N^{coef_log_t10:.2f})\n")

# ====================================================================
# OPÇÃO 2: Custo Linear Médio (Regressão Linear de todos os pontos)
# ====================================================================
coef_lin_t4, _ = np.polyfit(df_t4['num_nos'], df_t4['Tempo_s'], 1)
coef_lin_t10, _ = np.polyfit(df_t10['num_nos'], df_t10['Tempo_s'], 1)

print(f"--- Custo Linear Médio (Regressão com todos os pontos) ---")
print(f"T4 : {coef_lin_t4:.6f} segundos/nó")
print(f"T10: {coef_lin_t10:.6f} segundos/nó\n")

# ====================================================================
# OPÇÃO 3: Taxa de Crescimento Normal (Método dos Dois Pontos: Último - Primeiro)
# Delta Y / Delta X
# ====================================================================
# Calcula para o T4
delta_tempo_t4 = df_t4['Tempo_s'].iloc[-1] - df_t4['Tempo_s'].iloc[0]
delta_nos_t4 = df_t4['num_nos'].iloc[-1] - df_t4['num_nos'].iloc[0]
taxa_dois_pontos_t4 = delta_tempo_t4 / delta_nos_t4

# Calcula para o T10
delta_tempo_t10 = df_t10['Tempo_s'].iloc[-1] - df_t10['Tempo_s'].iloc[0]
delta_nos_t10 = df_t10['num_nos'].iloc[-1] - df_t10['num_nos'].iloc[0]
taxa_dois_pontos_t10 = delta_tempo_t10 / delta_nos_t10

print(f"--- Taxa de Crescimento Simples (Apenas Primeiro e Último Ponto) ---")
print(f"T4 : {taxa_dois_pontos_t4:.6f} segundos/nó")
print(f"T10: {taxa_dois_pontos_t10:.6f} segundos/nó")