from pathlib import Path
import numpy as np
import os
from lmcv_tools.commands.reorder import start as lmcv_reorder_start

# AUXILIARY FUNCTIONS TO READ MESH - VERSION 4.x
def read_file_until_text(fd, text_to_find):
    while True:
        line = fd.readline()
        if not line:
            raise EOFError(f"Text {text_to_find} not found in file.")
        if text_to_find in line:
            return

def read_materials(fd):
    read_file_until_text(fd, '$PhysicalNames')
    line_with_n_materials = fd.readline().strip()
    try:
        n_materials = int(line_with_n_materials)
    except ValueError:
        print(f"ERROR: Expected integer after $PhysicalNames, got: '{line_with_n_materials}'")
        return 0, [], []
    materials, mat_names = [], []
    for _ in range(n_materials):
        parts = fd.readline().strip().split()
        if len(parts) < 3:
            print(f"WARNING: unexpected PhysicalNames line: '{' '.join(parts)}'")
            continue
        dim, phys_id, name = int(parts[0]), int(parts[1]), parts[2].strip('"')
        materials.append((dim, phys_id, name))
        mat_names.append(name)
    return len(materials), materials, mat_names

def read_nodes(fd):
    read_file_until_text(fd, '$Nodes')
    header_line = fd.readline().strip()
    parts = header_line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid $Nodes header: '{header_line}'")
    numEntityBlocks = int(parts[0])
    numNodes = int(parts[1])
    nodes = np.zeros((numNodes, 4))
    node_index = 0
    for _ in range(numEntityBlocks):
        block_header = fd.readline().strip().split()
        numNodesInBlock = int(block_header[3])
        node_ids = []
        for _ in range(numNodesInBlock):
            nid_line = fd.readline().strip()
            node_ids.append(int(nid_line))
        for nid in node_ids:
            coords_line = fd.readline().strip()
            coords = list(map(float, coords_line.split()))
            nodes[node_index, 0] = nid
            nodes[node_index, 1:4] = coords[:3]
            node_index += 1
    read_file_until_text(fd, '$EndNodes')
    return nodes

def read_elems(fd):
    read_file_until_text(fd, '$Elements')
    header_line = fd.readline().strip()
    parts = header_line.split()
    numEntityBlocks = int(parts[0])
    numElements = int(parts[1])
    valid_elem_types = {4, 5, 11, 12}
    elements = []
    elem_count = 0
    for _ in range(numEntityBlocks):
        block_header = fd.readline().strip().split()
        entity_tag = int(block_header[1])
        elemType = int(block_header[2])
        numElementsInBlock = int(block_header[3])
        if elemType not in valid_elem_types:
            for _ in range(numElementsInBlock):
                fd.readline()
            continue
        for _ in range(numElementsInBlock):
            parts = fd.readline().strip().split()
            elem_id = int(parts[0])
            node_ids = list(map(int, parts[1:]))
            if elemType == 4:  # TET4
                node_ids = [node_ids[0], node_ids[1], node_ids[2], node_ids[3]]
            elif elemType == 11:  # TET10
                node_ids = [node_ids[0], node_ids[4], node_ids[1], node_ids[9],  
                            node_ids[3], node_ids[7], node_ids[6], node_ids[5],  
                            node_ids[8], node_ids[2]]
            elif elemType == 5:  # HEX8
                node_ids = [node_ids[4], node_ids[5], node_ids[6], node_ids[7],
                            node_ids[0], node_ids[1], node_ids[2], node_ids[3]]
            elif elemType == 12:  # HEX20
                node_ids = [node_ids[4], node_ids[16], node_ids[5], node_ids[18], node_ids[6],
                            node_ids[19], node_ids[7], node_ids[17], node_ids[10], node_ids[12], node_ids[14],
                            node_ids[15], node_ids[0], node_ids[8], node_ids[1], node_ids[11],
                            node_ids[2], node_ids[13], node_ids[3], node_ids[9]]
            elements.append([elem_id, elemType, entity_tag] + node_ids)
            elem_count += 1
    read_file_until_text(fd, '$EndElements')
    print(f"Total valid elements read: {elem_count}")
    return elements

# FUNCTIONS TO READ GMSH FILE - VERSION 2.x
def read_nodes_v2(fd):
    """
    Reads the nodes from the $Nodes block (Gmsh format 2.2).
    Args:
        fd (file object): The file descriptor.
    Returns:
        np.array: A NumPy array containing node ID and coordinates (id, x, y, z).
    """
    read_file_until_text(fd, '$Nodes')
    
    # Number of nodes
    try:
        numNodes = int(fd.readline().strip())
    except ValueError:
        raise ValueError("Invalid format: could not read number of nodes in $Nodes section.")
    
    nodes = np.zeros((numNodes, 4))  # id, x, y, z
    
    for i in range(numNodes):
        parts = fd.readline().strip().split()
        if len(parts) < 4:
            raise ValueError(f"Incomplete node line at {i+1}: '{parts}'")
        
        nid = int(parts[0])
        x, y, z = map(float, parts[1:4])
        nodes[i, 0] = nid
        nodes[i, 1] = x
        nodes[i, 2] = y
        nodes[i, 3] = z

    read_file_until_text(fd, '$EndNodes')
    return nodes

def read_elems_v2(fd):
    """
    Reads the elements from the $Elements block (Gmsh format 2.1).
    Returns:
        list: [elem_id, elemType, phys_id, node1, node2, ...]
    """
    read_file_until_text(fd, '$Elements')
    try:
        numElements = int(fd.readline().strip())
    except ValueError as e:
        raise ValueError(f"Invalid format: could not read number of elements in $Elements section. Details: {e}")

    # Valid Element Types:  TET4(4), HEX8(5), TET10(11), HEX20(12)
    valid_elem_types = {4, 5, 11, 12}
    elements = []
    elem_count = 0

    for i in range(numElements):
        line = fd.readline().strip()
        if not line:
            print(f"WARNING: Reading of elements was unexpectedly interrupted on the line {i+1} of {numElements}")
            break
        
        parts = line.split()
        try:
            # Format v2.2: elem_id | elemType | numTags | Tag1 (PhysID) | [Other Tags] | Node1 | ...
            elem_id = int(parts[0])
            elemType = int(parts[1])
            numTags = int(parts[2])

            if elemType not in valid_elem_types:
                continue 

            # Tag1 (parts[3]) is the Physical Entity (Material) ID.
            if numTags < 1:
                print(f"WARNING: Element {elem_id} (Type {elemType}) doesn't have a Physical ID (Tag). Skipping.")
                continue
                
            entity_tag = int(parts[3]) # ID Physical (Material)
            
            # Nodes start after: elem_id, elemType, numTags, and (numTags) of tags.
            node_start_index = 3 + numTags
            node_ids = list(map(int, parts[node_start_index:]))
            
        except (ValueError, IndexError) as e:
            print(f"WARNING: malformed v2.x element line: '{line}' - {e}")
            continue
        if elemType == 4:  # TET4
            node_ids = [node_ids[0], node_ids[1], node_ids[2], node_ids[3]]
        elif elemType == 11:  # TET10
            node_ids = [node_ids[0], node_ids[4], node_ids[1], node_ids[9],  
                            node_ids[3], node_ids[7], node_ids[6], node_ids[5],  
                            node_ids[8], node_ids[2]]  
        elif elemType == 5:  # HEX8
            node_ids = [node_ids[4], node_ids[5], node_ids[6], node_ids[7],
                        node_ids[0], node_ids[1], node_ids[2], node_ids[3]]
        elif elemType == 12:  # HEX20
            node_ids = [node_ids[4], node_ids[16], node_ids[5], node_ids[18], node_ids[6],
                        node_ids[19], node_ids[7], node_ids[17], node_ids[10], node_ids[12], node_ids[14],
                        node_ids[15], node_ids[0], node_ids[8], node_ids[1], node_ids[11],
                        node_ids[2], node_ids[13], node_ids[3], node_ids[9]]
        elements.append([elem_id, elemType, entity_tag] + node_ids)
        elem_count += 1
    read_file_until_text(fd, '$EndElements')
    print(f"Total valid elements read: {elem_count}")
    return elements

def micro_model_3D(epsM, nodes):
    xmin, ymin, zmin = np.min(nodes[:,1]), np.min(nodes[:,2]), np.min(nodes[:,3])
    xmax, ymax, zmax = np.max(nodes[:,1]), np.max(nodes[:,2]), np.max(nodes[:,3])
    lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin
    tol = 1e-6*min(lx,ly,lz) if min(lx,ly,lz)>0 else 1e-6

    # Identify Vertices
    vtx_dict = {}
    i_vertices = {
        'A': (xmin, ymin, zmin),
        'B': (xmin, ymax, zmin),
        'C': (xmin, ymin, zmax),
        'D': (xmin, ymax, zmax),
        'E': (xmax, ymin, zmin),
        'F': (xmax, ymax, zmin),
        'G': (xmax, ymin, zmax),
        'H': (xmax, ymax, zmax)
    }

    for name, coords in i_vertices.items():
        cx, cy, cz = coords
        mask = (np.isclose(nodes[:,1], cx, atol=tol) &
                np.isclose(nodes[:,2], cy, atol=tol) &
                np.isclose(nodes[:,3], cz, atol=tol))
        candidates = nodes[mask]
        if len(candidates) > 1:
            # Choose the one closest to the ideal point.
            dists = np.linalg.norm(candidates[:,1:4] - np.array(coords), axis=1)
            idx = np.argmin(dists)
            vtx_dict[name] = int(candidates[idx,0])
        elif len(candidates) == 1:
            vtx_dict[name] = int(candidates[0,0])

    def sel(cond): 
        return nodes[cond].reshape(-1,4) if cond.any() else np.zeros((0,4))

    print("\nChecking Identified Vertices:")
    if not vtx_dict:
        print("No vertices were identified. Check the tolerance 'tol'.")
    else:
        for vtx_name, node_id in vtx_dict.items():
            print(f"  {vtx_name} (ID: {node_id})")

    # Identify Edges
    edges = {
        'yRight_zTop': sel(np.isclose(nodes[:,2],ymax,atol=tol)&np.isclose(nodes[:,3],zmax,atol=tol)),      # DH
        'yLeft_zBottom': sel(np.isclose(nodes[:,2],ymin,atol=tol)&np.isclose(nodes[:,3],zmin,atol=tol)),    # AE
        'yRight_zBottom': sel(np.isclose(nodes[:,2],ymax,atol=tol)&np.isclose(nodes[:,3],zmin,atol=tol)),   # BF
        'yLeft_zTop': sel(np.isclose(nodes[:,2],ymin,atol=tol)&np.isclose(nodes[:,3],zmax,atol=tol)),       # CG
        'xFront_zTop': sel(np.isclose(nodes[:,1],xmax,atol=tol)&np.isclose(nodes[:,3],zmax,atol=tol)),      # GH
        'xBack_zBottom': sel(np.isclose(nodes[:,1],xmin,atol=tol)&np.isclose(nodes[:,3],zmin,atol=tol)),    # AB     
        'xFront_zBottom': sel(np.isclose(nodes[:,1],xmax,atol=tol)&np.isclose(nodes[:,3],zmin,atol=tol)),   # EF
        'xBack_zTop': sel(np.isclose(nodes[:,1],xmin,atol=tol)&np.isclose(nodes[:,3],zmax,atol=tol)),       # CD  
        'xFront_yRight': sel(np.isclose(nodes[:,1],xmax,atol=tol)&np.isclose(nodes[:,2],ymax,atol=tol)),    # FH
        'xBack_yLeft': sel(np.isclose(nodes[:,1],xmin,atol=tol)&np.isclose(nodes[:,2],ymin,atol=tol)),      # AC
        'xFront_yLeft': sel(np.isclose(nodes[:,1],xmax,atol=tol)&np.isclose(nodes[:,2],ymin,atol=tol)),     # EG
        'xBack_yRight': sel(np.isclose(nodes[:,1],xmin,atol=tol)&np.isclose(nodes[:,2],ymax,atol=tol)),     # BD
              
    }

    # Identify Faces
    faces = {
        'xBack': sel(np.isclose(nodes[:,1],xmin,atol=tol)),      # ABDC
        'xFront': sel(np.isclose(nodes[:,1],xmax,atol=tol)),     # EFHG
        'yLeft': sel(np.isclose(nodes[:,2],ymin,atol=tol)),      # AEGC
        'yRight': sel(np.isclose(nodes[:,2],ymax,atol=tol)),     # BFHD
        'zBottom': sel(np.isclose(nodes[:,3],zmin,atol=tol)),    # EFBA
        'zTop': sel(np.isclose(nodes[:,3],zmax,atol=tol)),       # GHDC
    }
    return vtx_dict, edges, faces

# MAIN FUNCTION
def rve_msh2dat(msh_file, E_modulus, nu_poisson, epsM):
    
    """
    Main function to read a Gmsh .msh file and generate a .dat file
    """ 
    # Convert E and nu passed from main
    E_modulus = np.array(E_modulus, dtype=float)
    nu_poisson = np.array(nu_poisson, dtype=float)

    # Validate epsM
    epsM = np.array(epsM, dtype=float)

    # Gets the name to the .msh file
    msh_name = str(msh_file)
    print(f"File name: {msh_name}")

    try:
        fmsh = open(msh_name, 'r')
        head_preview = ''.join([fmsh.readline() for _ in range(5)])
        fmsh.seek(0)
        is_v2_format = ("$MeshFormat\n2.2" in head_preview) or ("$MeshFormat\r\n2.2" in head_preview)
    except IOError:
        print(f"Error: file {msh_name} could not be opened!")
        return

    try:
        # Read the materials of the RVE.
        num_mat_total, materials_list_all, mat_name_all = read_materials(fmsh)
        n_E = len(E_modulus)

        materials_list = [m for m in materials_list_all if m[0] == 3]
        num_mat = len(materials_list)

        # The physical entity's dimension tag is the first element of the tuple, `m[0]`
        if num_mat != n_E:
                    print(f"VALIDATION ERROR: Number of material groups ({num_mat}) does not match number of E_modulus entries ({n_E}).")
                    return 
                    
        # Ensures that materials_list is sorted by Physical ID
        materials_list.sort(key=lambda m: m[1])

        # Automatically detect the .msh file format (2.2 or 4.x)
        if is_v2_format:
            print("Gmsh 2.x format detected.")
            nodes = read_nodes_v2(fmsh)
        else:
            print("Gmsh 4.x format detected.")
            nodes = read_nodes(fmsh)

        nn = len(nodes)
        if nn == 0:
            print("Error: number of nodes = 0.")
            fmsh.close()
            return

        global_tol = 1e-9
        # Translate the origin to (0, 0, 0).
        xmin = np.min(nodes[:, 1])
        ymin = np.min(nodes[:, 2])
        zmin = np.min(nodes[:, 3])
        nodes[:, 1] -= xmin
        nodes[:, 2] -= ymin
        nodes[:, 3] -= zmin
        print(f"Nodes translated. xmin: {xmin}, ymin: {ymin}, zmin: {zmin}")

        # New node ID origin: Uses tolerance to find the node at (0, 0, 0)
        mask_origin = (np.isclose(nodes[:, 1], 0.0, atol=global_tol) & 
                       np.isclose(nodes[:, 2], 0.0, atol=global_tol) & 
                       np.isclose(nodes[:, 3], 0.0, atol=global_tol))

        if np.any(mask_origin):
            # Retrieves the ID of the first node that satisfies the condition from the origin.
            origin_node_id = nodes[mask_origin][0, 0] 
        else:
            raise ValueError("Error: Could not find source node (0, 0, 0) after translation.")


        # New node index origin
        origin_node_index = np.where(nodes[:, 0] == origin_node_id)[0][0]

        # Reorder the list so that the origin node comes first.
        nodes[0], nodes[origin_node_index] = nodes[origin_node_index], nodes[0].copy()

        # Copy the old order of IDs.
        old_ids = nodes[:, 0].copy()

        # New sequential vector of IDs
        new_ids = np.arange(1, nn + 1)

        # Create the map: Old ID -> New ID
        id_map = {old: new for old, new in zip(old_ids, new_ids)}

        # Applies renumbering to the nodes.
        nodes[:, 0] = new_ids

        print("\nChecking the coordinates of the first node:")
        print(f"Coordinates of the first node after translation: {nodes[0, 1:]}")

        fmsh.seek(0) 
        if is_v2_format:
            elements = read_elems_v2(fmsh)
        else:
            elements = read_elems(fmsh)

        ne = len(elements)
        if ne == 0:
            print("Error: number of elements = 0.")
            fmsh.close()
            return

        # Get element type and related data.
        elem_type = elements[0][1]
        if elem_type == 4:
            elm_string = '%ELEMENT.TET4'
            nne = 4
            ngr, ngs, ngt = 1, 1, 1
        elif elem_type == 11:
            elm_string = '%ELEMENT.TET10'
            nne = 10
            ngr, ngs, ngt = 2, 2, 2
        elif elem_type == 5:
            elm_string = '%ELEMENT.BRICK8'
            nne = 8
            ngr, ngs, ngt = 2, 2, 2
        elif elem_type == 12:
            elm_string = '%ELEMENT.BRICK20'
            nne = 20
            ngr, ngs, ngt = 3, 3, 3
        else:
            print(f"Unsupported element type: {elem_type}")
            fmsh.close()
            return

        # Open dat file and write header
        # Path needs to be fixed to user-specified FAST directory
        output_dir = r"C:\Users\bruna\OneDrive\Abaqus_UFC\FASTv2.4.4\FASTv2.4.4"
        base = os.path.splitext(os.path.basename(msh_file))[0]
        dat_name = base + ".dat"
        try:
            fdat = open(dat_name, 'w')
        except IOError:
            print(f"Error: file {dat_name} could not be opened for writing!")
            fmsh.close()
            return
                 
        # RVE dimensions for offset calculation
        xmax, ymax, zmax = np.max(nodes[:,1]), np.max(nodes[:,2]), np.max(nodes[:,3])
        Lx = np.max(nodes[:,1]) - np.min(nodes[:,1])
        Ly = np.max(nodes[:,2]) - np.min(nodes[:,2])
        Lz = np.max(nodes[:,3]) - np.min(nodes[:,3])

        fdat.write('%HEADER\n')
        fdat.write('File generated by msh2dat_3D.\n\n')

        fdat.write('%HEADER.ANALYSIS.ALGORITHM\n')
        fdat.write("'microlin'\n\n")

        fdat.write('%HEADER.PRINT.AVERAGE.NODAL.STRESS\n')
        fdat.write('1\n\n')

        # Write nodes to the dat file.
        fdat.write('%NODE\n')
        fdat.write(f'{nn}\n\n')

        fdat.write('%NODE.COORD\n')
        fdat.write(f'{nn}\n')
        # nodes[:, 0] contains original IDs
        for i in range(nn):
            fdat.write(f'{int(nodes[i, 0]):-5d}  {nodes[i, 1]:17.12e}  {nodes[i, 2]:17.12e}  {nodes[i, 3]:17.12e}\n')

        # Get the vertices and edges of the RVE.
        vtx_dict, edges, faces = micro_model_3D(epsM, nodes)
        print("Micro-model data obtained.")

        # Write node supports.
        fdat.write('\n%NODE.SUPPORT\n')
        fdat.write('1\n')  # Only 1 node supported
        fdat.write(f'1  {vtx_dict["A"]:-3d}   1   1    0   0   0\n')

        # Write springs.
        fdat.write('\n%MICRO.MODEL.REGULARIZATION.FACTOR\n')
        fdat.write('1.0e-8\n')

        # Write materials and sections.
        fdat.write('\n%MATERIAL\n')
        fdat.write(f'{num_mat}\n')

        fdat.write('\n%MATERIAL.ISOTROPIC\n')
        fdat.write(f'{num_mat}\n')
        for i, mat in enumerate(materials_list):
            fdat.write(f'{i+1}     {E_modulus[i]:0.5e}     {nu_poisson[i]:0.5f}\n')

        fdat.write('\n%SECTION\n')
        fdat.write(f'{num_mat}\n')

        fdat.write('\n%SECTION.HOMOGENEOUS.ISOTROPIC.3D\n')
        fdat.write(f'{num_mat}\n')
        for i, mat in enumerate(materials_list):
            fdat.write(f'{i+1}     {i+1}\n')

        # Write integration order - one per physical group
        fdat.write('\n%INTEGRATION.ORDER\n')
        fdat.write(f'{num_mat}\n')
        for i in range(num_mat):
            fdat.write(f'{i+1:<5d}     {ngr:<5d}     {ngs:<5d}     {ngt:<5d}     {ngr:<5d}     {ngs:<5d}     {ngt:<5d}\n')

        # Write elements.
        fdat.write('\n%ELEMENT\n')
        fdat.write(f'{ne}\n')

        fdat.write(f'\n{elm_string}\n')
        fdat.write(f'{ne}\n')

        # Loop for writing elements
        for i, elem_data in enumerate(elements, start=1):
            elem_id = i
            phys_group = elem_data[2] 
            node_ids = elem_data[3:] 
            # Reorder node IDs based on the id_map
            new_node_ids = [id_map[nid] for nid in node_ids]
            try:
                # Find the index of the physical group ID in the materials lists: [(dim, phys_id, name), ...]
                section_id = [m[1] for m in materials_list].index(phys_group) + 1
            except ValueError:
                section_id = 1  # If not found, assume 1
            
            node_format_str = ' '.join([f' {nid:-4d}' for nid in new_node_ids])
            fdat.write(f'{elem_id:<5d}   {section_id:<5d}     {section_id:<5d} {node_format_str}\n')

        dim = 3 if elem_type  in (4, 5, 11, 12) else 2
        fdat.write(f'\n%MICRO.MODEL\n')
        vol_3D = Lx * Ly * Lz
        fdat.write(f'{dim}   {vol_3D:.6e}\n\n')

        mpc_lines = []
        node_pairs = []
        all_constrained_nodes = set()
        # Adds the vertex nodes to the set of already processed nodes.
        for node_id in vtx_dict.values():
            all_constrained_nodes.add(node_id)

        # MPCs of vertex nodes
        vertex_rows = {}
        for vname, nid in vtx_dict.items():
            if vname == "A": 
                continue
            idxs = np.where(nodes[:, 0] == nid)[0]
            if idxs.size == 0:
                print(f"WARNING: vertex {vname} (id {nid}) not found in nodes array.")
            else:
                vertex_rows[vname] = int(idxs[0])

        # Map of Edges
        edge_pairs = [
            ('yRight_zTop','yLeft_zBottom'),
            ('yRight_zBottom','yLeft_zTop'),
            ('xFront_zTop','xBack_zBottom'),
            ('xFront_zBottom','xBack_zTop'),
            ('xFront_yRight','xBack_yLeft'),
            ('xFront_yLeft','xBack_yRight')
        ]

        for edge_a_name, edge_b_name in edge_pairs:
            edge_a_nodes = edges[edge_a_name]
            edge_b_nodes = edges[edge_b_name]

            if edge_a_nodes.shape[0] == 0 or edge_b_nodes.shape[0] == 0:
                continue

            # Determine which coordinate varies along the edge
            dx = np.ptp(edge_a_nodes[:,1])  # range in x
            dy = np.ptp(edge_a_nodes[:,2])  # range in y
            dz = np.ptp(edge_a_nodes[:,3])  # range in z
            if dx > dy and dx > dz:
                key = 1  # sorty by x
            elif dy > dx and dy > dz:
                key = 2  # sort by y
            else:
                key = 3  # sort by z

            # Sort the nodes on the two edges
            edge_a_sorted = edge_a_nodes[np.argsort(edge_a_nodes[:,key])]
            edge_b_sorted = edge_b_nodes[np.argsort(edge_b_nodes[:,key])]

            # Node-to-node pairing
            for node_a, node_b in zip(edge_a_sorted, edge_b_sorted):
                id_a, x_a, y_a, z_a = node_a
                id_b, x_b, y_b, z_b = node_b

                # Skip vertices (only internal nodes)
                if int(id_a) in vtx_dict.values() or int(id_b) in vtx_dict.values():
                    continue

                # Save pairs for %MICRO.MODEL.NODE.PAIRS
                node_pairs.append((int(id_a), int(id_b)))

        # MPCs of faces
        face_pairs = {
            'xFront': 'xBack', 'yRight': 'yLeft', 'zTop': 'zBottom',
        }
        for face_a_name, face_b_name in face_pairs.items():
            face_a_nodes = faces[face_a_name]
            face_b_nodes = faces[face_b_name]

            for node_a in face_a_nodes:
                id_a = int(node_a[0])
                if id_a in vtx_dict.values() or any(id_a in [n[0] for n in e] for e in edges.values()): continue

                x_a, y_a, z_a = node_a[1:4]

                if 'x' in face_a_name:
                    corresponding_node = face_b_nodes[np.isclose(face_b_nodes[:, 2], y_a, atol=1e-6) & np.isclose(face_b_nodes[:, 3], z_a, atol=1e-6)]
                elif 'y' in face_a_name:
                    corresponding_node = face_b_nodes[np.isclose(face_b_nodes[:, 1], x_a, atol=1e-6) & np.isclose(face_b_nodes[:, 3], z_a, atol=1e-6)]
                else: # z
                    corresponding_node = face_b_nodes[np.isclose(face_b_nodes[:, 1], x_a, atol=1e-6) & np.isclose(face_b_nodes[:, 2], y_a, atol=1e-6)]

                if corresponding_node.shape[0] > 0:
                    id_b = int(corresponding_node[0, 0])
                    if id_b in vtx_dict.values() or any(id_b in [n[0] for n in e] for e in edges.values()): continue

                    # Save pairs for %MICRO.MODEL.NODE.PAIRS 
                    node_pairs.append((int(id_a), int(id_b)))  # Add to Global List
            
        # Print Vertices
        fdat.write('%MICRO.MODEL.VERTEX.NODES\n')
        # Vertex Order
        vertex_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        vertex_ids = [vtx_dict[v] for v in vertex_order if v in vtx_dict]
        fdat.write(' '.join(str(vid) for vid in vertex_ids) + '\n')

        # Print Node Pairs
        fdat.write('\n%MICRO.MODEL.NODE.PAIRS\n')
        fdat.write(f'{len(node_pairs)}\n')
        for id_a, id_b in node_pairs:
            fdat.write(f'{id_a:4d}   {id_b:<4d}\n')
            
        # Print Macro Strains
        fdat.write('\n%MICRO.MODEL.MACRO.STRAINS\n')
        fdat.write(f'{epsM[0]: .6e}  {epsM[1]: .6e}  {epsM[2]: .6e}  {epsM[3]: .6e}  {epsM[4]: .6e}  {epsM[5]: .6e}\n')

        fdat.write('\n%END\n')
        print("\nSuccessfully generated .dat file.")

    except EOFError as e:
        print(f"Error reading .msh file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        fmsh.close()
        if 'fdat' in locals() and not fdat.closed:
            fdat.close()

    # Call reorder function
    print("\nStarting file reordering .dat...")
    reordering_method = 'sloan' # or rcm
    flags_to_use = {'-i': 'info'} 
    
    lmcv_reorder_start(reordering_method, str(dat_name), flags_to_use)
    print(f"Reordering complete. The %NODE.SOLVER.ORDER block has been added to '{dat_name}'.")

    try:
        with open(dat_name, 'r') as f:
            lines = f.readlines()
        
        # Find the positions of the blocks
        node_coord_start = -1
        solver_order_start = -1
        end_line = -1

        for i, line in enumerate(lines):
            if line.strip() == '%NODE.COORD':
                node_coord_start = i
            elif line.strip() == '%NODE.SOLVER.ORDER':
                solver_order_start = i
            elif line.strip() == '%END':
                end_line = i

        if node_coord_start == -1 or solver_order_start == -1 or end_line == -1:
            print("ERROR: Could not find required blocks in file. The file will not be overwritten..")
            return

        # Separates the reordering block
        solver_order_block = []
        for i in range(solver_order_start, len(lines)):
            line = lines[i]
            if i > solver_order_start and line.strip().startswith('%'):
                break  
            solver_order_block.append(line)

        f_solver_order_block = []
        for idx, line in enumerate(solver_order_block):
            if idx == 0:  
                f_solver_order_block.append(line)
                continue
            if line.strip().isdigit():
                f_solver_order_block.append(f"{int(line.strip()):d}\n")
            else:
                ids = line.split()
                f_line = "".join(f"{int(nid):6d}" for nid in ids) + "\n"
                f_solver_order_block.append(f_line)

        # Remove the end block and the %END line
        del lines[solver_order_start:solver_order_start + len(solver_order_block)]
        lines = [l for l in lines if l.strip() != '%END']

        # Inserts the reordering block after the coordinates
        node_coord_end = node_coord_start + 1 + nn + 1 
        new_lines = lines[:node_coord_end] + ["\n"] + f_solver_order_block + ["\n"] + lines[node_coord_end:]
        new_lines.append('%END\n')

        # Rewrites the file
        with open(dat_name, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Positioning of blocko %NODE.SOLVER.ORDER successfully adjusted in file '{dat_name}'.")

    except Exception as e:
        print(f"Error rewriting file to move reorder block: {e}")















