import pandas as pd
import numpy as np

arquivo_excel = "dados_chawla1998.xlsx"

# Lê, limpa e ordena
df = pd.read_excel(arquivo_excel, sheet_name='vf=0').dropna(subset=['Strain', 'Stress']).sort_values(by='Strain')

# 1. Filtra apenas as duas colunas e GUARDA a matriz gerada numa variável
matriz_dados = df[['Strain', 'Stress']].to_numpy()

# 2. Imprime usando repr() para manter a formatação de código Python
print("Matriz pronta a copiar:")
print(repr(matriz_dados))