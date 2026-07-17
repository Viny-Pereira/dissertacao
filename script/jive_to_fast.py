from msh2dat_3D import rve_msh2dat
import os

def jive_to_fast(
    arquivo_msh,
    E_modulos,
    nu_poissons,
    deformacao_macro,
):
    """
    Converte um arquivo .msh para o formato .dat utilizado pelo FAST.

    Parameters
    ----------
    arquivo_msh : str
        Caminho do arquivo .msh.
    E_modulos : list
        [E_matriz, E_inclusao]
    nu_poissons : list
        [nu_matriz, nu_inclusao]
    deformacao_macro : list
        [e11, e22, e33, g12, g13, g23]
    """

    if not os.path.isfile(arquivo_msh):
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_msh}")

    rve_msh2dat(
        arquivo_msh,
        E_modulos,
        nu_poissons,
        deformacao_macro,
    )

    print("Conversão concluída.")

E_modulos = [78e3, 395e3]
nu_poissons = [0.33, 0.20]
deformacao_macro = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]

jive_to_fast(
    arquivo_msh="rve_vf10.msh",
    E_modulos=E_modulos,
    nu_poissons=nu_poissons,
    deformacao_macro=deformacao_macro,
)