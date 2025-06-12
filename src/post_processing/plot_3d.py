import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class PostProcessorHexMesh:
    """
    Post-processing class for FEM results visualization
    Handles 3D rendering of hexahedral elements with flat surfaces
    """
    
    def __init__(self, coordinates, elements):
        """
        Initialize post-processor
        
        Parameters:
        coordinates: array of node coordinates (n_nodes x 3)
        elements: array of element connectivity (n_elements x 8)
        """
        self.coordinates = np.array(coordinates)
        self.elements = np.array(elements)
        self.n_nodes = len(coordinates)
        self.n_elements = len(elements)
        
        # Define the 6 faces of a hexahedron (Q8 element)
        # Each face is defined by 4 nodes in counter-clockwise order
        self.hex_faces = [
            [0, 1, 2, 3],  # Bottom face (z = constant, lower)
            [4, 7, 6, 5],  # Top face (z = constant, upper)
            [0, 4, 5, 1],  # Front face (y = constant, lower)
            [2, 6, 7, 3],  # Back face (y = constant, upper)
            [0, 3, 7, 4],  # Left face (x = constant, lower)
            [1, 5, 6, 2]   # Right face (x = constant, upper)
        ]

        self.displacements = None 
        self.scale_factor = None

    def set_displacements(self, displacements, scale_factor=1.0):
        """
        Set displacements for the nodes
        
        Parameters:
        displacements: array of displacements (n_nodes x 3)
        """
        if displacements.shape[0] != self.n_nodes or displacements.shape[1] != 3:
            raise ValueError(f"Displacement array shape {displacements.shape} \
                               doesn't match number of nodes {self.n_nodes}")
        self.displacements = np.array(displacements)
        self.scale_factor = scale_factor  

    def get_element_faces(self, element_id, displaced=False):
        """
        Get all faces of a specific element
        
        Parameters:
        element_id: element index
        displaced: if True, apply displacements to the coordinates
        
        Returns:
        faces: list of face coordinates (6 faces x 4 vertices x 3 coordinates)
        """
        elem_nodes = self.elements[element_id]
        elem_coords = self.coordinates[elem_nodes]

        # If displacements are provided, apply them
        if self.displacements is not None and displaced:
            elem_coords += self.scale_factor * self.displacements[elem_nodes]
        
        faces = []
        for face_nodes in self.hex_faces:
            face_coords = elem_coords[face_nodes]
            faces.append(face_coords)
            
        return faces
    

    def get_all_faces(self, skip_internal=True, displaced=False):
        """
        Get all faces from all elements
        
        Parameters:
        skip_internal: if True, skip internal faces (shared between elements)
        displaced: if True, apply displacements to the coordinates
        
        Returns:
        all_faces: list of all face coordinates
        face_elements: corresponding element ID for each face
        """
        all_faces = []
        face_elements = []
        
        if skip_internal:
            # More complex: identify and skip internal faces
            face_dict = {}  # Dictionary to track face sharing
            
            for elem_id in range(self.n_elements):
                faces = self.get_element_faces(elem_id, displaced=displaced)
                
                for i, face in enumerate(faces):
                    # Create a unique identifier for this face based on sorted node indices
                    elem_nodes = self.elements[elem_id]
                    face_node_indices = elem_nodes[self.hex_faces[i]]
                    face_key = tuple(sorted(face_node_indices))
                    
                    if face_key in face_dict:
                        # This face is shared - mark both as internal
                        face_dict[face_key]['shared'] = True
                    else:
                        # New face
                        face_dict[face_key] = {
                            'coords': face,
                            'element': elem_id,
                            'face_id': i,
                            'shared': False
                        }
            
            # Add only non-shared (external) faces
            for face_info in face_dict.values():
                if not face_info['shared']:
                    all_faces.append(face_info['coords'])
                    face_elements.append(face_info['element'])
                    
        else:
            # Simple: add all faces
            for elem_id in range(self.n_elements):
                faces = self.get_element_faces(elem_id, displaced=displaced)
                for face in faces:
                    all_faces.append(face)
                    face_elements.append(elem_id)
        
        return all_faces, face_elements
    

    def plot_solid(self, field_data=None, field_name="Field", cmap='viridis', 
                   alpha=0.8, show_edges=True, edge_color='black', edge_alpha=0.3,
                   skip_internal=True, figsize=(12, 9), original_outline=True):
        """
        Plot the solid with flat surfaces
        
        Parameters:
        field_data: optional field data for coloring (per node or per element)
        field_name: name of the field for colorbar
        cmap: colormap name
        alpha: transparency of faces
        show_edges: whether to show face edges
        edge_color: color of edges
        edge_alpha: transparency of edges
        skip_internal: whether to skip internal faces
        figsize: figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all faces
        all_faces, face_elements = self.get_all_faces(skip_internal=skip_internal, displaced=True)
        
        if not all_faces:
            print("No faces to plot!")
            return fig, ax
        
        # Prepare colors
        if field_data is not None:
            field_data = np.array(field_data)
            
            if len(field_data) == self.n_nodes:
                # Node-based field data - average to get face colors
                face_colors = []
                for i, face in enumerate(all_faces):
                    # Find the nodes of this face and average their field values
                    elem_id = face_elements[i]
                    elem_nodes = self.elements[elem_id]
                    
                    # # Get face node indices (need to map back to original nodes)
                    # face_center = np.mean(face, axis=0)
                    # This is approximate - for exact mapping, we'd need to track node indices
                    face_field_avg = np.mean(field_data[elem_nodes])
                    face_colors.append(face_field_avg)
                    
            elif len(field_data) == self.n_elements:
                # Element-based field data
                face_colors = [field_data[face_elements[i]] for i in range(len(all_faces))]
            else:
                print(f"Field data length {len(field_data)} doesn't match nodes {self.n_nodes} or elements {self.n_elements}")
                face_colors = None
        else:
            face_colors = None
        
        # Create face collection
        if face_colors is not None:
            # Normalize colors
            norm = Normalize(vmin=np.min(face_colors), vmax=np.max(face_colors))
            colormap = cm.get_cmap(cmap)
            
            # Create colored faces
            face_collection = []
            colors_list = []
            
            for i, face in enumerate(all_faces):
                face_collection.append(face)
                color_val = norm(face_colors[i])
                colors_list.append(colormap(color_val))
            
            # Add faces to plot
            poly3d = [[tuple(vertex) for vertex in face] for face in face_collection]
            face_patches = Poly3DCollection(poly3d, alpha=alpha)
            face_patches.set_facecolors(colors_list)
            
            if show_edges:
                face_patches.set_edgecolors(edge_color)
                face_patches.set_linewidths(0.5)
                face_patches.set_alpha(edge_alpha)
            
            ax.add_collection3d(face_patches)
            
            # Add colorbar
            mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
            mappable.set_array(face_colors)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(field_name)
            
        else:
            # Uniform coloring
            poly3d = [[tuple(vertex) for vertex in face] for face in all_faces]
            face_patches = Poly3DCollection(poly3d, alpha=alpha, facecolors='lightblue')
            
            if show_edges:
                face_patches.set_edgecolors(edge_color)
                face_patches.set_linewidths(0.5)
            
            ax.add_collection3d(face_patches)
        
        if self.displacements is not None and original_outline:
            # Plot original mesh outline
            all_faces, _ = self.get_all_faces(skip_internal=True, displaced=False)
            
            # Plot original faces as wireframe
            for face in all_faces:
                # Plot face edges
                for i in range(4):
                    start = face[i]
                    end = face[(i+1)%4]
                    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                            'k--', alpha=0.3, linewidth=0.5)
        

        # Set axis properties
        self._set_axis_properties(ax)
        
        # Add title
        if self.displacements is not None:
            title = "Deformed 3D Mesh. Scale factor: x{:.1f}".format(self.scale_factor)
        else:
            title = "3D Mesh"
        if field_data is not None:
            title += f" - {field_name}"
        ax.set_title(title)
        
        return fig, ax
    
    
    def plot_mode_shape_solid(self, mode_vector, scale_factor=1, mode_number=1, frequency=None, **kwargs):
        """
        Plot mode shape as deformed solid
        
        Parameters:
        mode_vector: eigenvector for the mode
        free_dofs: indices of free DOFs
        scale_factor: scaling factor for visualization
        mode_number: mode number for title
        frequency: natural frequency (Hz)
        **kwargs: additional arguments for plot_deformed_solid
        """        
        # Reshape to get displacements for each node

        modal_displacements = mode_vector.reshape((-1, 3))
        self.set_displacements(modal_displacements, scale_factor=scale_factor) 

        # Use displacement magnitude as field data
        disp_magnitude = np.linalg.norm(modal_displacements, axis=1)
        
        # Plot deformed solid
        fig, ax = self.plot_solid(
            field_data=disp_magnitude,
            field_name="Displacement Magnitude",
            original_outline=True,
            **kwargs
        )
        
        # Update title
        title = f"Mode {mode_number}"
        if frequency is not None:
            title += f" - {frequency:.2f} Hz"
        ax.set_title(title)
        
        return fig, ax
    
    def _set_axis_properties(self, ax):
        """Set axis properties for 3D plot"""
        all_coords = np.copy(self.coordinates) 
        if self.displacements is not None:
            # If displacements are set, adjust the nodes accordingly
            all_coords += self.scale_factor * self.displacements
        # Calculate bounds
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        z_min, z_max = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        
        padding = 0.05 * max_range
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        # ax.set_box_aspect([x_range, y_range, z_range])
    
    def plot_slice(self, plane_normal='z', plane_position=0.5, field_data=None, 
                   field_name="Field", **kwargs):
        """
        Plot a slice through the mesh
        
        Parameters:
        plane_normal: 'x', 'y', or 'z' for plane orientation
        plane_position: position along the normal axis (0 to 1)
        field_data: optional field data for coloring
        field_name: name of the field
        **kwargs: additional plotting arguments
        """
        # This would require intersection calculations - placeholder for now
        print("Slice plotting functionality - to be implemented")
        pass
    

# Example usage and integration with the Q8HexMesh class
def demo_postprocessing():
    """Demonstrate post-processing capabilities"""
    
    # Create a simple test mesh
    nx, ny, nz = 3, 2, 2
    lx, ly, lz = 2.0, 1.0, 1.0
    
    # Generate nodes
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    nodes = []
    
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    for k in range(nnz):
        for j in range(nny):
            for i in range(nnx):
                nodes.append([i*dx, j*dy, k*dz])
    
    # Generate elements
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = k * nnx * nny + j * nnx + i
                n2 = k * nnx * nny + j * nnx + (i + 1)
                n3 = k * nnx * nny + (j + 1) * nnx + (i + 1)
                n4 = k * nnx * nny + (j + 1) * nnx + i
                n5 = (k + 1) * nnx * nny + j * nnx + i
                n6 = (k + 1) * nnx * nny + j * nnx + (i + 1)
                n7 = (k + 1) * nnx * nny + (j + 1) * nnx + (i + 1)
                n8 = (k + 1) * nnx * nny + (j + 1) * nnx + i
                
                elements.append([n1, n2, n3, n4, n5, n6, n7, n8])
    
    # Create post-processor
    postproc = PostProcessorHexMesh(nodes, elements)
    
    # Test 1: Basic solid plotting
    print("Plotting basic solid...")
    fig1, ax1 = postproc.plot_solid(figsize=(10, 8))
    plt.show()
    
    # Test 2: Solid with field data (example: distance from origin)
    print("Plotting solid with field data...")
    nodes_array = np.array(nodes)
    field_data = np.linalg.norm(nodes_array, axis=1)  # Distance from origin
    
    fig2, ax2 = postproc.plot_solid(
        field_data=field_data,
        field_name="Distance from Origin",
        cmap='plasma',
        alpha=0.9
    )
    plt.show()
    
    # Test 3: Deformed solid (example displacement)
    print("Plotting deformed solid...")
    displacements = np.zeros_like(nodes_array)
    # Create a simple bending displacement pattern
    for i, node in enumerate(nodes_array):
        x, y, z = node
        displacements[i] = 2*np.array([0, 0, 0.1 * x * z])  # Bending in z-direction
    
    postproc.set_displacements(displacements)
    fig3, ax3 = postproc.plot_solid(
        field_data=np.linalg.norm(displacements, axis=1),
        field_name="Displacement Magnitude",
        cmap='coolwarm'
    )
    plt.show()

if __name__ == "__main__":
    demo_postprocessing()