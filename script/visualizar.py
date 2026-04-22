import pyvista as pv
import meshio
import gmsh
import os

def visualizar_resultados(caminho_ficheiro):
    print(f"A carregar o ficheiro: {caminho_ficheiro} ...")
    
    try:
        # 1. Tentar carregar diretamente com PyVista (se for vtk, vtu, etc.)
        malha = pv.read(caminho_ficheiro)
        
    except Exception:
        print("A tentar carregar via Meshio/Gmsh...")
        try:
            # 2. Forçar o meshio a ignorar o problema do ponto no nome e ler como malha Gmsh
            malha_meshio = meshio.read(caminho_ficheiro, file_format="gmsh")
            malha = pv.wrap(malha_meshio)
            
        except Exception as e:
            # 3. Solução robusta: Se for um "View" legacy do Gmsh, o meshio falha. 
            # Usamos o próprio Gmsh para converter para VTK em background.
            print("Formato .pos complexo detetado. A converter para .vtk usando a API do Gmsh...")
            
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0) # Silencia o output do Gmsh
            
            # Carrega o resultado .pos
            gmsh.merge(caminho_ficheiro)
            
            # Cria um ficheiro .vtk temporário
            caminho_vtk = caminho_ficheiro.replace(".pos", ".vtk")
            gmsh.write(caminho_vtk)
            gmsh.finalize()
            
            # Lê o ficheiro recém-criado com o PyVista
            malha = pv.read(caminho_vtk)
            
            # Opcional: apagar o ficheiro .vtk temporário após a leitura
            # os.remove(caminho_vtk)

    # Inspecionar os campos de dados disponíveis
    campos_disponiveis = malha.array_names
    print(f"\nCampos de resultados disponíveis na malha: {campos_disponiveis}")

    if not campos_disponiveis:
        print("Aviso: Nenhum campo de resultados foi encontrado no ficheiro.")
        campo_ativo = None
    else:
        campo_ativo = campos_disponiveis[0] 
        print(f"A visualizar o campo: {campo_ativo}")

    # Visualização
    plotter = pv.Plotter()
    plotter.add_mesh(
        malha, 
        scalars=campo_ativo, 
        cmap='jet', 
        show_edges=True,
        edge_color='white',
        scalar_bar_args={'title': str(campo_ativo), 'vertical': True}
    )

    plotter.add_axes()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_text("Pós-processamento de Resultados 3D", font_size=12)
    plotter.show()

if __name__ == '__main__':
    # Utilize o nome real do seu ficheiro
    ficheiro_pos = "RVE_vf20_m0.2_ord2_N4467_E2856_FAST.pos" 
    visualizar_resultados(ficheiro_pos)