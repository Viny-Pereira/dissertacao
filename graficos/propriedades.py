import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# ==========================================
# 1. DEFINIÇÃO DOS MODELOS CONSTITUTIVOS
# ==========================================
def modelo_bilinear(eps, E, E_t, eps_y):
    sig_y = E * eps_y 
    return np.piecewise(eps, [eps < eps_y, eps >= eps_y],
                        [lambda x: E * x,
                         lambda x: sig_y + E_t * (x - eps_y)])

def modelo_voce(eps_p, sig_y0, Q_inf, b):
    return sig_y0 + Q_inf * (1 - np.exp(-b * eps_p))

# ==========================================
# 2. LEITURA E CONVERSÃO DOS DADOS
# ==========================================
try:
    arquivo_excel = "dados_chawla1998.xlsx"
    df = pd.read_excel(arquivo_excel, sheet_name='vf=0').dropna(subset=['True_Strain2', 'True_Stress2']).sort_values(by='True_Strain2')
    
    eps_true = df["True_Strain2"].values - df["True_Strain2"].values[0]
    sig_true = df["True_Stress2"].values - df["True_Stress2"].values[0]

    eps_eng = np.exp(eps_true) - 1
    sig_eng = sig_true / np.exp(eps_true) 

    eps_exp = eps_eng
    sig_exp = sig_eng

except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o nome ou diretório do arquivo.")

# ==========================================
# 3. REGRESSÃO BILINEAR (ENGENHARIA)
# ==========================================
chute_inicial = [50000, 1000, 0.005] 
limites = ([100, 0, 0], [np.inf, np.inf, np.inf])

popt, _ = curve_fit(modelo_bilinear, eps_exp, sig_exp, p0=chute_inicial, bounds=limites)

E_calc = popt[0]
Et_calc = popt[1]
epsy_calc = popt[2]
sigy_calc = E_calc * epsy_calc

H_calc = (E_calc * Et_calc) / (E_calc - Et_calc) if E_calc != Et_calc else np.inf 

# ==========================================
# 4. REGRESSÃO DE VOCE (TENSÃO VERDADEIRA)
# ==========================================
eps_p_true = eps_true - (sig_true / E_calc)

sig_y0_true = sig_true[np.searchsorted(eps_eng, epsy_calc)] 
mascara_plast = sig_true > sig_y0_true

eps_p_fit = eps_p_true[mascara_plast]
sig_true_fit = sig_true[mascara_plast]

### CORREÇÃO 1: Zerar a deformação plástica inicial para o ajuste matemático ser perfeito ###
eps_p_fit = eps_p_fit - eps_p_fit[0]

def ajuste_voce_fit(eps_p, Q_inf, b):
    return modelo_voce(eps_p, sig_y0_true, Q_inf, b)

chute_voce = [100.0, 10.0]
limites_voce = ([0, 0], [np.inf, np.inf])

popt_voce, _ = curve_fit(ajuste_voce_fit, eps_p_fit, sig_true_fit, p0=chute_voce, bounds=limites_voce)
Q_calc = popt_voce[0]
b_calc = popt_voce[1]

# ==========================================
# 5. EXIBIÇÃO DE RESULTADOS
# ==========================================
print("\n" + "="*50)
print(" RESULTADOS DA REGRESSÃO BILINEAR (ISOLINHARD)")
print("="*50)
print(f"Módulo de Young (E)  [MPa]       : {E_calc:10.2f}")
print(f"Módulo Tangente (Et) [MPa]       : {Et_calc:10.2f}")
print(f"Tensão de Escoamento (Sy) [MPa]  : {sigy_calc:10.2f}")
print(f"Módulo de Encruamento (H) [MPa]  : {H_calc:10.2f}")

print("\n" + "="*50)
print(" RESULTADOS DA REGRESSÃO DE VOCE (NÃO-LINEAR)")
print("="*50)
print(f"Tensão Escoamento (Sy_0)   : {sig_y0_true:10.2f} MPa")
print(f"Parâmetro Saturação (Q_inf): {Q_calc:10.2f} MPa")
print(f"Taxa de Endurecimento (b)  : {b_calc:10.2f}")
print("="*50)

# ==========================================
# 6. GERAÇÃO DAS CURVAS PARA PLOTAGEM
# ==========================================
# Curva Bilinear (Engenharia)
sig_fit_bilinear = modelo_bilinear(eps_exp, E_calc, Et_calc, epsy_calc)

### CORREÇÃO 2: Construir a curva de Voce completa (Elástica + Plástica) ###

# Parte 1: Reta Elástica (desde 0 até a Tensão de Escoamento Verdadeira)
eps_true_elastico = np.linspace(0, sig_y0_true / E_calc, 50)
sig_true_elastico = E_calc * eps_true_elastico

# Parte 2: Curva Plástica de Voce
eps_p_teorico = np.linspace(0, np.max(eps_p_true), 200)
sig_true_plastico = modelo_voce(eps_p_teorico, sig_y0_true, Q_calc, b_calc)
# Reconstruir Deformação Verdadeira Total (Elástica + Plástica)
eps_true_plastico = eps_p_teorico + (sig_true_plastico / E_calc)

# Juntar a linha elástica com a linha plástica
eps_true_voce_completo = np.concatenate((eps_true_elastico, eps_true_plastico))
sig_true_voce_completo = np.concatenate((sig_true_elastico, sig_true_plastico))

# Converter de volta para Engenharia de uma só vez para plotar no gráfico final
eps_eng_voce = np.exp(eps_true_voce_completo) - 1
sig_eng_voce = sig_true_voce_completo / np.exp(eps_true_voce_completo)

# ==========================================
# 7. PLOTAGEM DO GRÁFICO
# ==========================================
plt.figure(figsize=(10, 7))
plt.plot(eps_exp, sig_exp, 'k.', alpha=0.3, label='Experimento (Engenharia)')

plt.plot(eps_exp, sig_fit_bilinear, 'b--', linewidth=2, label='Ajuste Bilinear')
plt.plot(eps_eng_voce, sig_eng_voce, 'r-', linewidth=2.5, label='Ajuste Voce (Exponencial)')

# Marcações
plt.plot(epsy_calc, sigy_calc, 'bo', markersize=6)
plt.axvline(x=epsy_calc, color='k', linestyle=':', alpha=0.4)

# Caixas de texto explicativas
caixa_bilinear = f"Modelo Bilinear:\n$E$ = {E_calc:.0f} MPa\n$\sigma_y$ = {sigy_calc:.1f} MPa\n$H$ = {H_calc:.0f} MPa"
caixa_voce = f"Modelo Voce:\n$Q_\infty$ = {Q_calc:.1f} MPa\n$b$ = {b_calc:.1f}"

plt.text(0.01, 0.85, caixa_bilinear, transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='blue', alpha=0.8))
plt.text(0.01, 0.65, caixa_voce, transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

plt.title('Comparação de Modelos de Encruamento Macroscópico', fontsize=14, fontweight='bold')
plt.xlabel('Engineering Strain (m/m)', fontsize=12)
plt.ylabel('Engineering Stress (MPa)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


# ==========================================
# 8. EXTRAÇÃO DAS FUNÇÕES DE YIELD (STRINGS PARA O SOLVER)
# ==========================================
# O FAST/Jive geralmente usa 'eqpe' ou 'alpha' como variável de deformação plástica.
# Vamos usar 'eqpe' como exemplo padrão.
var_plastica = "eqpe" 

print("\n" + "="*65)
print(" STRINGS PARA FICHEIRO DE CONFIGURAÇÃO (.pro / FAST / JIVE)")
print("="*65)

# 1. Função Yield Linear (Isotrópica)
# Formato: sig_y0 + H * eqpe
string_yield_linear = f"{sigy_calc:.4f} + ({H_calc:.4f} * {var_plastica})"
print("Modelo de Encruamento Linear (ISOLINHARD):")
print(f"YieldFunction = \"{string_yield_linear}\"")
print("-" * 65)

# 2. Função Yield Exponencial (Voce)
# Formato: sig_y0 + Q * (1 - exp(-b * eqpe))
string_yield_voce = f"{sig_y0_true:.4f} + {Q_calc:.4f} * (1.0 - exp(-{b_calc:.4f} * {var_plastica}))"
print("Modelo de Encruamento Exponencial (VOCE):")
print(f"YieldFunction = \"{string_yield_voce}\"")
print("="*65 + "\n")