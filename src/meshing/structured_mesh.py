import numpy as np
from typing import Tuple

class StructuredHexMesh:
    """
    A class for generating structured hexahedral meshes of a 3D box.
    
    The mesh consists of regular hexahedral (8-node brick) elements arranged
    in a structured grid pattern within a box of dimensions Lx × Ly × Lz.
    """
    
    def __init__(self, Lx: float, Ly: float, Lz: float, 
                 nx: int, ny: int, nz: int,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize the structured hexahedral mesh.
        
        Parameters:
        -----------
        Lx, Ly, Lz : float
            Dimensions of the box in x, y, z directions
        nx, ny, nz : int
            Number of elements in x, y, z directions
        origin : tuple of float, optional
            Origin point of the box (default: (0, 0, 0))
        """
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.origin = np.array(origin)
        
        # Calculate spacing
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz
        
        # Number of nodes and elements
        self.n_nodes_x = nx + 1
        self.n_nodes_y = ny + 1
        self.n_nodes_z = nz + 1
        self.n_nodes = self.n_nodes_x * self.n_nodes_y * self.n_nodes_z
        self.n_elements = nx * ny * nz

        self.hex_faces = [
            [0, 1, 2, 3],  # Bottom face (z = constant, lower)
            [4, 7, 6, 5],  # Top face (z = constant, upper)
            [0, 4, 5, 1],  # Front face (y = constant, lower)
            [2, 6, 7, 3],  # Back face (y = constant, upper)
            [0, 3, 7, 4],  # Left face (x = constant, lower)
            [1, 5, 6, 2]   # Right face (x = constant, upper)
        ]
        
        # Generate mesh
        self._generate_nodes()
        self._generate_elements()
    
    def _generate_nodes(self):
        """Generate node coordinates."""
        self.coordinates = np.zeros((self.n_nodes, 3))
        
        node_id = 0
        for k in range(self.n_nodes_z):
            for j in range(self.n_nodes_y):
                for i in range(self.n_nodes_x):
                    x = self.origin[0] + i * self.dx
                    y = self.origin[1] + j * self.dy
                    z = self.origin[2] + k * self.dz
                    self.coordinates[node_id] = [x, y, z]
                    node_id += 1
    
    def _node_index(self, i: int, j: int, k: int) -> int:
        """Convert (i,j,k) grid indices to global node index."""
        return k * self.n_nodes_x * self.n_nodes_y + j * self.n_nodes_x + i
    
    def _generate_elements(self):
        """
        Generate hexahedral elements with standard node ordering.
        
        Node ordering for each hexahedron (right-hand rule):
        Bottom face (z=const): 0-1-2-3 (counterclockwise when viewed from +z)
        Top face (z=const+dz): 4-5-6-7 (counterclockwise when viewed from +z)
        
        Standard hexahedral connectivity:
        0: (i,j,k)     4: (i,j,k+1)
        1: (i+1,j,k)   5: (i+1,j,k+1)
        2: (i+1,j+1,k) 6: (i+1,j+1,k+1)
        3: (i,j+1,k)   7: (i,j+1,k+1)
        """
        self.elements = np.zeros((self.n_elements, 8), dtype=int)
        
        elem_id = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # Bottom face nodes
                    n0 = self._node_index(i, j, k)
                    n1 = self._node_index(i+1, j, k)
                    n2 = self._node_index(i+1, j+1, k)
                    n3 = self._node_index(i, j+1, k)
                    
                    # Top face nodes
                    n4 = self._node_index(i, j, k+1)
                    n5 = self._node_index(i+1, j, k+1)
                    n6 = self._node_index(i+1, j+1, k+1)
                    n7 = self._node_index(i, j+1, k+1)
                    
                    self.elements[elem_id] = [n0, n1, n2, n3, n4, n5, n6, n7]
                    elem_id += 1
    
    def get_element_center(self, elem_id: int) -> np.ndarray:
        """Get the center coordinates of an element."""
        elem_nodes = self.elements[elem_id]
        return np.mean(self.coordinates[elem_nodes], axis=0)
    
    def get_element_volume(self) -> float:
        """Get the volume of each element (all elements have same volume)."""
        return self.dx * self.dy * self.dz
    
    def get_boundary_faces(self) -> dict:
        """
        Get boundary faces for each face of the box.
        
        Returns:
        --------
        dict: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
              Each value is a list of (element_id, local_face_id) tuples
        """
        boundary_faces = {
            'x_min': [], 'x_max': [],
            'y_min': [], 'y_max': [],
            'z_min': [], 'z_max': []
        }
        
        elem_id = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # x_min face (i=0)
                    if i == 0:
                        boundary_faces['x_min'].append((elem_id, 3))  # face 3: nodes 0,3,7,4
                    
                    # x_max face (i=nx-1)
                    if i == self.nx - 1:
                        boundary_faces['x_max'].append((elem_id, 1))  # face 1: nodes 1,2,6,5
                    
                    # y_min face (j=0)
                    if j == 0:
                        boundary_faces['y_min'].append((elem_id, 0))  # face 0: nodes 0,1,5,4
                    
                    # y_max face (j=ny-1)
                    if j == self.ny - 1:
                        boundary_faces['y_max'].append((elem_id, 2))  # face 2: nodes 2,3,7,6
                    
                    # z_min face (k=0)
                    if k == 0:
                        boundary_faces['z_min'].append((elem_id, 4))  # face 4: nodes 0,1,2,3
                    
                    # z_max face (k=nz-1)
                    if k == self.nz - 1:
                        boundary_faces['z_max'].append((elem_id, 5))  # face 5: nodes 4,5,6,7
                    
                    elem_id += 1
        
        return boundary_faces
    
    
    def get_faces_from_nodes(self, surface_node_ids):
        """
        Extract surface elements (quadrilaterals) from node list
        based on known surface nodes.
        
        Parameters:
        -----------
        surface_node_ids : list or set
            Node IDs that belong to the surface
            
        Returns:
        --------
        surface_elements : list
            List of faces on the surface, each containing 4 node IDs
        """
        
        surface_node_set = set(surface_node_ids)
        surface_elements = []
        
        hex_faces = self.hex_faces
        
        # Iterate through all hexahedral elements in connectivity matrix
        for element_nodes in self.elements:
            # element_nodes should contain 8 node IDs for Q8 hex element
            
            # Check each face of the current hexahedral element
            for face_local_nodes in hex_faces:
                # Get global node IDs for this face
                face_global_nodes = [element_nodes[i] for i in face_local_nodes]
                
                # Check if all 4 face nodes are in the surface node set
                if all(node_id in surface_node_set for node_id in face_global_nodes):
                    # This face lies entirely on the surface
                    surface_elements.append(face_global_nodes)
        
        return surface_elements
    
    
    def export_vtk(self, filename: str):
        """
        Export mesh to VTK format for visualization.
        
        Parameters:
        -----------
        filename : str
            Output filename (should end with .vtk)
        """
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Structured Hexahedral Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            f.write(f"POINTS {self.n_nodes} float\n")
            for node in self.coordinates:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
            
            # Write cells
            f.write(f"CELLS {self.n_elements} {self.n_elements * 9}\n")
            for elem in self.elements:
                f.write(f"8 {' '.join(map(str, elem))}\n")
            
            # Write cell types (12 = VTK_HEXAHEDRON)
            f.write(f"CELL_TYPES {self.n_elements}\n")
            for _ in range(self.n_elements):
                f.write("12\n")
    
    def get_mesh_info(self) -> dict:
        """Get summary information about the mesh."""
        return {
            'dimensions': (self.Lx, self.Ly, self.Lz),
            'elements': (self.nx, self.ny, self.nz),
            'element_size': (self.dx, self.dy, self.dz),
            'total_nodes': self.n_nodes,
            'total_elements': self.n_elements,
            'element_volume': self.get_element_volume(),
            'total_volume': self.Lx * self.Ly * self.Lz
        }
    
    def __str__(self) -> str:
        info = self.get_mesh_info()
        return (f"StructuredHexMesh:\n"
                f"  Box dimensions: {info['dimensions']}\n"
                f"  Grid size: {info['elements']}\n"
                f"  Element size: {info['element_size']}\n"
                f"  Nodes: {info['total_nodes']}\n"
                f"  Elements: {info['total_elements']}")

