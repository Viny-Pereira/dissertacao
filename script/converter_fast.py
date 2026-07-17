from msh2dat_3D import rve_msh2dat

name = "rve_vf30-malha-distribuida.msh"

# Propriedades
E_modulos = [78e3, 395e3]
nu_poissons = [0.33, 0.2]

# Condições de Contorno Periódicas (Macro)
deformacao_macro = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]

# Chama o conversor
print(f"Iniciando conversão da malha: {name}")
rve_msh2dat(name, E_modulos, nu_poissons, deformacao_macro)
print("Conversão finalizada com sucesso!")