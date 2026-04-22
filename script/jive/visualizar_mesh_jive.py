from PVPlotter import * # type: ignore
from pyvistaqt import BackgroundPlotter

def plot_result(mesh_file, result_file, data_type, comp):
    
    # A variável 'mesh' guarda a sua malha 3D sólida e completa
    mesh = PVMesh.readFromGmsh(mesh_file) # type: ignore
    pl = BackgroundPlotter()
    jpl = plotters.JivePlot.JivePlot(pl) # type: ignore

    # 1. O JivePlot faz o trabalho sujo de ler o .out e jogar os dados na malha
    jpl.setMesh(mesh)
    jpl.plot(result_file, tableType='NodeTable', dataType=data_type, comp=comp, scale=1.5)

    # 2. ESCONDEMOS a renderização original do JivePlot (que é oca)
    jpl.mactor.SetVisibility(False)

    # 3. Descobrimos qual foi o nome exato da variável de cor que o JivePlot gerou (ex: 'disp_dx')
    scalar_name = jpl.mactor.mapper.dataset.active_scalars_name

    # 4. Deformamos a malha sólida manualmente (já que escondemos a do JivePlot)
    # Assumimos que o vetor 3D principal se chama 'disp' ou tem o nome do data_type
    vetor_disp = data_type if data_type in mesh.point_data else 'disp'
    
    if vetor_disp in mesh.point_data:
        solid_mesh = mesh.warp_by_vector(vetor_disp, factor=1.5)
    else:
        solid_mesh = mesh

    # 5. Configuramos a cor ativa para ser a componente (comp) desejada
    solid_mesh.set_active_scalars(scalar_name)
    min_val = solid_mesh.active_scalars.min()
    max_val = solid_mesh.active_scalars.max()

    # ==========================================
    # CORTE VOLUMÉTRICO NATIVO (O SEGREDO!)
    # ==========================================
    # Diferente do VTK puro, o add_mesh_clip_plane do PyVista entende volumes
    # Ele corta os tetraedros por dentro e preenche o espaço visualmente!
    pl.add_mesh_clip_plane(solid_mesh, 
                           scalars=scalar_name,
                           clim=[min_val, max_val],
                           cmap='jet',           # Paleta clássica
                           show_edges=True,      # MOSTRA A GRADE INTERNA (Ótimo para RVEs!)
                           edge_color='black',
                           normal='x',
                           crinkle=True)           # O plano nasce cortando o eixo X

    pl.add_axes()
    return pl

if __name__ == "__main__":
    mesh_file = "rve.msh"
    result_file = "rve.1.out"
    data_type = "disp"
    comp = "dx"
    
    pl = plot_result(mesh_file, result_file, data_type, comp)
    pl.app.exec_()