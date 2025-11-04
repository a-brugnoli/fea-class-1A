import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from src.meshing.structured_mesh import StructuredHexMesh
from src.post_processing.configuration import configure_matplotlib
configure_matplotlib()


class PostProcessorHexMesh:
    """
    Post-processing class for FEM results visualization
    Handles 3D rendering of hexahedral elements with flat surfaces
    """
    
    def __init__(self, mesh: StructuredHexMesh):
        """
        Initialize post-processor
        
        Parameters:
        coordinates: array of node coordinates (n_nodes x 3)
        elements: array of element connectivity (n_elements x 8)
        """
        self.coordinates = np.array(mesh.coordinates)
        self.elements = np.array(mesh.elements)
        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        
        self.hex_faces = mesh.hex_faces
        
        # Define the 6 faces of a hexahedron (Q8 element)
        # Each face is defined by 4 nodes in counter-clockwise order

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
    

    def plot_slice(self, plane='x', plane_position=0.5, field_data=None, 
                field_name="Field", cmap='viridis', figsize=(10, 8),
                show_edges=True, edge_color='black', alpha=0.8,
                displaced=False):
        """
        Plot a 2D slice through the mesh
        
        Parameters:
        plane: 'x', 'y', or 'z' - normal direction of the slice plane
        plane_position: position along the plane axis (0 to 1 for relative position)
        field_data: optional field data for coloring (per node or per element)
        field_name: name of the field for colorbar
        cmap: colormap name
        figsize: figure size
        show_edges: whether to show face edges
        edge_color: color of edges
        alpha: transparency of faces
        displaced: if True, use displaced coordinates
        
        Returns:
        fig, ax: matplotlib figure and axis objects
        """
        # Determine plane axis
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        if plane.lower() not in plane_map:
            raise ValueError("Plane must be 'x', 'y', or 'z'")
        
        plane_axis = plane_map[plane.lower()]
        
        # Get coordinates (with or without displacement)
        coords = np.copy(self.coordinates)
        if displaced and self.displacements is not None:
            coords += self.scale_factor * self.displacements
        
        # Convert relative position to absolute if needed
        if 0 <= plane_position <= 1:
            axis_min = np.min(coords[:, plane_axis])
            axis_max = np.max(coords[:, plane_axis])
            plane_pos_abs = axis_min + plane_position * (axis_max - axis_min)
        else:
            raise ValueError("Plane position must be between 0 and 1 (relative position with respect to axis legth)")
        
        # Define other two axes for 2D projection
        other_axes = [i for i in range(3) if i != plane_axis]
        axis_names = ['X', 'Y', 'Z']
        
        # Find elements that intersect the plane
        intersecting_faces = []
        face_field_values = []
        
        tolerance = 1e-10
        
        for elem_id in range(self.n_elements):
            elem_nodes = self.elements[elem_id]
            elem_coords = coords[elem_nodes]
            
            # Check if element spans the plane
            elem_min = np.min(elem_coords[:, plane_axis])
            elem_max = np.max(elem_coords[:, plane_axis])
            
            if elem_min - tolerance <= plane_pos_abs <= elem_max + tolerance:
                # Element intersects plane - check each face
                for face_nodes in self.hex_faces:
                    face_coords = elem_coords[face_nodes]
                    
                    # Check if face intersects the plane
                    face_min = np.min(face_coords[:, plane_axis])
                    face_max = np.max(face_coords[:, plane_axis])
                    
                    if face_min - tolerance <= plane_pos_abs <= face_max + tolerance:
                        # Compute intersection polygon
                        intersection_points = _intersect_face_with_plane(
                            face_coords, plane_axis, plane_pos_abs, tolerance
                        )
                        
                        if len(intersection_points) >= 3:
                            # Project to 2D
                            points_2d = intersection_points[:, other_axes]
                            intersecting_faces.append(points_2d)
                            
                            # Get field value for this face
                            if field_data is not None:
                                field_data = np.array(field_data)
                                if len(field_data) == self.n_nodes:
                                    # Average field values of face nodes
                                    face_field_avg = np.mean(field_data[elem_nodes[face_nodes]])
                                elif len(field_data) == self.n_elements:
                                    face_field_avg = field_data[elem_id]
                                else:
                                    face_field_avg = 0
                                face_field_values.append(face_field_avg)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if not intersecting_faces:
            ax.text(0.5, 0.5, 'No intersection found at this position',
                    ha='center', va='center', transform=ax.transAxes)
            return fig, ax
        
        # Plot the intersecting faces
        if field_data is not None and face_field_values:
            # Color by field data
            norm = Normalize(vmin=np.min(face_field_values), 
                            vmax=np.max(face_field_values))
            colormap = cm.get_cmap(cmap)
            
            for face_2d, field_val in zip(intersecting_faces, face_field_values):
                color = colormap(norm(field_val))
                polygon = plt.Polygon(face_2d, facecolor=color, alpha=alpha,
                                    edgecolor=edge_color if show_edges else 'none',
                                    linewidth=0.5 if show_edges else 0)
                ax.add_patch(polygon)
            
            # Add colorbar
            mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
            mappable.set_array(face_field_values)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.8)
            cbar.set_label(field_name)
        else:
            # Uniform coloring
            for face_2d in intersecting_faces:
                polygon = plt.Polygon(face_2d, facecolor='lightblue', alpha=alpha,
                                    edgecolor=edge_color if show_edges else 'none',
                                    linewidth=0.5 if show_edges else 0)
                ax.add_patch(polygon)
        
        # Set axis properties
        ax.set_xlabel(axis_names[other_axes[0]])
        ax.set_ylabel(axis_names[other_axes[1]])
        # ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set title
        title = f"Slice at {plane.upper()} = {plane_pos_abs:.3f}"
        if field_data is not None:
            title += f" - {field_name}"
        ax.set_title(title)
        
        # Auto-scale axes
        ax.autoscale()
        
        return fig, ax


    def extract_line_profile(self, axis, coord1, coord2, 
                            field_data=None, num_points=100):
        """
        Extract field values along a line parallel to x, y, or z axis
        
        Parameters:
        axis: 'x', 'y', or 'z' - direction of the line
        coord1: first fixed coordinate (e.g., if axis='z', this is x)
        coord2: second fixed coordinate (e.g., if axis='z', this is y)
        field_data: field data (per node or per element)
        num_points: number of interpolation points along axis
        
        Returns:
        axis_coords: coordinates along the specified axis
        field_values: interpolated field values
        
        Examples:
        - extract_line_profile('z', x=0.5, y=0.5)  # along z at fixed x,y
        - extract_line_profile('y', coord1=0.5, coord2=0.5)  # along y at fixed x,z
        - extract_line_profile('x', coord1=0.5, coord2=0.5)  # along x at fixed y,z
        """
        # Map axis to index
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis.lower() not in axis_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        axis_idx = axis_map[axis.lower()]
        other_axes = [i for i in range(3) if i != axis_idx]
        
        # Get coordinates (with or without displacement)
        coords = np.copy(self.coordinates)
        
        # Find range along specified axis
        axis_min = np.min(coords[:, axis_idx])
        axis_max = np.max(coords[:, axis_idx])
        axis_coords = np.linspace(axis_min, axis_max, num_points)
        
        # Validate fixed coordinates
        if coord1 is None or coord2 is None:
            raise ValueError(f"Both coord1 and coord2 must be specified for axis='{axis}'")
        
        # Prepare field data
        if field_data is None:
            return axis_coords, None
        
        field_data = np.array(field_data)
        
        # Determine if field data is nodal or elemental
        is_nodal = len(field_data) == self.n_nodes
        is_elemental = len(field_data) == self.n_elements
        
        if not (is_nodal or is_elemental):
            raise ValueError(f"Field data length {len(field_data)} doesn't match "
                            f"nodes {self.n_nodes} or elements {self.n_elements}")
        
        field_values = []
        tolerance = 1e-10
        
        for axis_val in axis_coords:
            # Build point coordinates based on which axis we're traversing
            point = np.zeros(3)
            point[axis_idx] = axis_val
            point[other_axes[0]] = coord1
            point[other_axes[1]] = coord2
            
            # Find element containing this point
            field_val = None
            
            for elem_id in range(self.n_elements):
                elem_nodes = self.elements[elem_id]
                elem_coords = coords[elem_nodes]
                
                # Check if point is inside element bounding box (with tolerance)
                if _point_in_hex_bbox(point, elem_coords, tolerance):
                    # Point is potentially in this element
                    if is_nodal:
                        # Interpolate from nodes using trilinear interpolation
                        field_val = _interpolate_in_hex(
                            point, elem_coords, field_data[elem_nodes]
                        )
                    else:  # is_elemental
                        # Use element value directly
                        field_val = field_data[elem_id]
                    
                    if field_val is not None:
                        break
            
            field_values.append(field_val if field_val is not None else np.nan)
        
        return axis_coords, np.array(field_values)


def _interpolate_in_hex(point, hex_coords, hex_values):
    """
    Trilinear interpolation within a hexahedron
    
    Parameters:
    point: 3D point coordinates
    hex_coords: coordinates of hex vertices (8 x 3)
    hex_values: field values at hex vertices (8,)
    
    Returns:
    interpolated value or None if point outside element
    """
    # Transform to local coordinates (xi, eta, zeta) in [-1, 1]
    # For a structured hex, we can use the inverse mapping
    
    # Simple approach: use the hex corners to define local coordinates
    # Assuming hex_coords follows standard ordering:
    # 0-3: bottom face, 4-7: top face
    
    x, y, z = point
    
    # Get bounds
    x_min, x_max = np.min(hex_coords[:, 0]), np.max(hex_coords[:, 0])
    y_min, y_max = np.min(hex_coords[:, 1]), np.max(hex_coords[:, 1])
    z_min, z_max = np.min(hex_coords[:, 2]), np.max(hex_coords[:, 2])
    
    # Check if point is outside (with small tolerance)
    tolerance = 1e-10
    if (x < x_min - tolerance or x > x_max + tolerance or
        y < y_min - tolerance or y > y_max + tolerance or
        z < z_min - tolerance or z > z_max + tolerance):
        return None
    
    # Map to local coordinates [-1, 1]
    if x_max - x_min > tolerance:
        xi = 2 * (x - x_min) / (x_max - x_min) - 1
    else:
        xi = 0
    
    if y_max - y_min > tolerance:
        eta = 2 * (y - y_min) / (y_max - y_min) - 1
    else:
        eta = 0
    
    if z_max - z_min > tolerance:
        zeta = 2 * (z - z_min) / (z_max - z_min) - 1
    else:
        zeta = 0
    
    # Trilinear interpolation using shape functions
    # Standard hex8 shape functions
    N = np.array([
        (1 - xi) * (1 - eta) * (1 - zeta) / 8,  # Node 0
        (1 + xi) * (1 - eta) * (1 - zeta) / 8,  # Node 1
        (1 + xi) * (1 + eta) * (1 - zeta) / 8,  # Node 2
        (1 - xi) * (1 + eta) * (1 - zeta) / 8,  # Node 3
        (1 - xi) * (1 - eta) * (1 + zeta) / 8,  # Node 4
        (1 + xi) * (1 - eta) * (1 + zeta) / 8,  # Node 5
        (1 + xi) * (1 + eta) * (1 + zeta) / 8,  # Node 6
        (1 - xi) * (1 + eta) * (1 + zeta) / 8,  # Node 7
    ])
    
    # Interpolate
    value = np.sum(N * hex_values)
    
    return value


def _point_in_hex_bbox(point, hex_coords, tolerance):
    """
    Check if point is within hexahedron bounding box
    
    Parameters:
    point: 3D point coordinates
    hex_coords: coordinates of hex vertices (8 x 3)
    tolerance: numerical tolerance
    
    Returns:
    bool: True if point is in bounding box
    """
    for axis in range(3):
        coord_min = np.min(hex_coords[:, axis]) - tolerance
        coord_max = np.max(hex_coords[:, axis]) + tolerance
        
        if point[axis] < coord_min or point[axis] > coord_max:
            return False
    return True


def _intersect_face_with_plane(face_coords, plane_axis, plane_pos, tolerance):
    """
    Compute intersection of a face with a plane
    
    Parameters:
    face_coords: coordinates of face vertices (4 x 3)
    plane_axis: axis perpendicular to plane (0, 1, or 2)
    plane_pos: position of plane along axis
    tolerance: numerical tolerance
    
    Returns:
    intersection_points: array of intersection points
    """
    intersection_points = []
    n_vertices = len(face_coords)
    
    for i in range(n_vertices):
        v1 = face_coords[i]
        v2 = face_coords[(i + 1) % n_vertices]
        
        v1_val = v1[plane_axis]
        v2_val = v2[plane_axis]
        
        # Check if vertex is on the plane
        if abs(v1_val - plane_pos) < tolerance:
            if len(intersection_points) == 0 or \
            not np.allclose(intersection_points[-1], v1, atol=tolerance):
                intersection_points.append(v1)
        
        # Check if edge crosses the plane
        if (v1_val - plane_pos) * (v2_val - plane_pos) < 0:
            # Linear interpolation to find intersection point
            t = (plane_pos - v1_val) / (v2_val - v1_val)
            intersection_point = v1 + t * (v2 - v1)
            
            # Avoid duplicates
            if len(intersection_points) == 0 or \
            not np.allclose(intersection_points[-1], intersection_point, atol=tolerance):
                intersection_points.append(intersection_point)
    
    return np.array(intersection_points) if intersection_points else np.array([]).reshape(0, 3)