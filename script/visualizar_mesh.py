import pyvista as pv
import meshio
import os

class VisualizadorInterativoRVE:
    def __init__(self, arquivo_msh: str):
        self.arquivo_msh = arquivo_msh
        self.plotter = pv.Plotter(title="Visualizador RVE")
        pv.set_plot_theme("paraview")
        
        # 1. Carrega a malha base (.vtu)
        self.mesh_bruta = self._carregar_vtu()
        
        # 2. Descobre os IDs físicos disponíveis
        fisicos = self.mesh_bruta.get_array("gmsh:physical")
        self.val_min = min(fisicos) if fisicos is not None else 1
        self.val_max = max(fisicos) if fisicos is not None else 2
        
        # Estado inicial dos filtros
        self.limiar_inferior = self.val_min
        self.limiar_superior = self.val_max
        self.bounds_corte = self.mesh_bruta.bounds # Limites iniciais (Cubo inteiro)
        self.menu_ativo = False
        
        # Variáveis para guardar os Atores e Menus
        self.ator_superficie = None
        self.ator_arestas = None
        self.widgets_menu = []
        
        # 3. Inicializa a interface e desenha a primeira vez
        self._configurar_atalhos_e_menu()
        self.atualizar_malha_visual()

    def _carregar_vtu(self):
        """Converte do Gmsh para VTK apenas na primeira leitura"""
        arquivo_vtu = self.arquivo_msh.replace('.msh', '.vtu')
        if not os.path.exists(arquivo_vtu):
            print("Convertendo malha .msh para formato visualizável .vtu...")
            meshio.read(self.arquivo_msh).write(arquivo_vtu)
            
        print("Carregando os dados brutos no motor gráfico...")
        mesh = pv.UnstructuredGrid(arquivo_vtu)
        if "gmsh:physical" in mesh.cell_data:
            mesh.set_active_scalars("gmsh:physical")
        return mesh

    def atualizar_malha_visual(self):
        """Aplica os filtros (Materiais + Corte) e refaz a subdivisão"""
        print("Aplicando recorte e processando geometria. Aguarde...")
        
        # Passo A: Filtra pelos Materiais (Sliders)
        mesh_filtrada = self.mesh_bruta.threshold(
            [self.limiar_inferior, self.limiar_superior], 
            scalars="gmsh:physical"
        )
        
        # Passo B: Recorta a malha usando os limites da Caixa 3D (Box Widget)
        # O 'invert=False' garante que apenas o que está DENTRO da caixa permaneça
        mesh_recortada = mesh_filtrada.clip_box(self.bounds_corte, invert=False, crinkle=True)
        
        # Passo C: Extrai as faces curvas e as arestas
        superficie = mesh_recortada.separate_cells().extract_surface(nonlinear_subdivision=4)
        arestas = superficie.extract_feature_edges()

        # Passo D: Limpa o ecrã antigo
        if self.ator_superficie:
            self.plotter.remove_actor(self.ator_superficie)
        if self.ator_arestas:
            self.plotter.remove_actor(self.ator_arestas)

        # Passo E: Desenha a nova malha cortada
        self.ator_superficie = self.plotter.add_mesh(
            superficie, scalars="gmsh:physical", style='surface',
            cmap=["#a8d0e6", "#f76c6c"], show_scalar_bar=False
        )
        self.ator_arestas = self.plotter.add_mesh(
            arestas, style='wireframe', color='black', line_width=1
        )
        
        self.ator_arestas.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.plotter.render()
        print("-> Visualização atualizada com sucesso!")

    def _configurar_atalhos_e_menu(self):
        """Monta os Sliders, a Caixa 3D e 'ouve' o teclado"""
        
        # Funções de callback para os menus
        def set_min(valor): self.limiar_inferior = valor
        def set_max(valor): self.limiar_superior = valor
        def set_box(bounds): self.bounds_corte = bounds # Atualiza os limites de corte

        # Sliders (Id Físico)
        slider_min = self.plotter.add_slider_widget(
            set_min, [self.val_min, self.val_max], value=self.val_min,
            title="ID Fisico Min", pointa=(0.05, 0.9), pointb=(0.25, 0.9), style='modern'
        )
        slider_max = self.plotter.add_slider_widget(
            set_max, [self.val_min, self.val_max], value=self.val_max,
            title="ID Fisico Max", pointa=(0.05, 0.75), pointb=(0.25, 0.75), style='modern'
        )
        
        # Caixa de Recorte (Box Widget)
        box_widget = self.plotter.add_box_widget(
            set_box, 
            bounds=self.bounds_corte, 
            rotation_enabled=False # Trava a rotação para cortes perfeitos nos eixos XYZ
        )
        
        # Guarda todos para ligar/desligar com o 'M'
        self.widgets_menu = [slider_min, slider_max, box_widget]
        for w in self.widgets_menu:
            w.Off()

        # Atalhos
        self.plotter.add_key_event("m", self.toggle_menu)
        self.plotter.add_key_event("space", self.atualizar_malha_visual)
        self.plotter.add_key_event("s", self.toggle_superficie)
        self.plotter.add_key_event("l", self.toggle_arestas)

        # Texto de Ajuda
        texto_ajuda = (
            "--- COMANDOS DO RVE ---\n"
            "[M] Ligar/Desligar Painel (Corte e Sliders)\n"
            "[Espaco] Aplicar Corte e Filtros\n"
            "[S] Ocultar Volume Sólido\n"
            "[L] Ocultar Malha de Arame"
        )
        self.plotter.add_text(texto_ajuda, position="lower_left", font_size=10, color="black")

    def toggle_menu(self):
        self.menu_ativo = not self.menu_ativo
        for w in self.widgets_menu:
            w.On() if self.menu_ativo else w.Off()

    def toggle_superficie(self):
        if self.ator_superficie:
            self.ator_superficie.visibility = not self.ator_superficie.visibility
            self.plotter.render()

    def toggle_arestas(self):
        if self.ator_arestas:
            self.ator_arestas.visibility = not self.ator_arestas.visibility
            self.plotter.render()

    def mostrar(self):
        self.plotter.show_axes()
        self.plotter.show()


if __name__ == "__main__":
    # Substitua pelo seu arquivo .msh
    arquivo = 'RVE_vf20_m0.2_ord2_N4467_E2856_FAST.dat' 
    
    visualizador = VisualizadorInterativoRVE(arquivo)
    visualizador.mostrar()