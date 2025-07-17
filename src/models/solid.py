import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple
from src.meshing.structured_mesh import StructuredHexMesh
from src.utilities.vonmises import compute_von_mises_stress


class Q8FiniteElementAssembler:
    """
    Finite element assembler for Q8 (8-node hexahedral) elements.
    
    This class takes a structured hexahedral mesh and assembles global
    stiffness and mass matrices using trilinear shape functions.
    """
    
    def __init__(self, mesh: StructuredHexMesh, material_props: dict):
        """
        Initialize the Q8 finite element assembler.
        
        Parameters:
        -----------
        mesh : StructuredHexMesh
            The hexahedral mesh object
        material_props : dict
            Material properties dictionary containing:
            - 'E': Young's modulus
            - 'nu': Poisson's ratio  
            - 'rho': Density
        """
        self.mesh = mesh
        self.E = material_props['E']
        self.nu = material_props['nu']
        self.rho = material_props['rho']
        
        # Number of DOFs (3 per node for 3D displacement)
        self.dof_per_node = 3
        self.n_dofs = self.dof_per_node * mesh.n_nodes
        
        # Gauss quadrature points and weights for 2x2x2 integration
        self._setup_gauss_quadrature()
        
        # Compute material matrix
        self.D = self._compute_material_matrix()
        
        # Initialize matrices
        self.K_global = None
        self.M_global = None
    
    def _setup_gauss_quadrature(self):
        """Set up Gauss quadrature points and weights for 2x2x2 integration."""
        # Gauss points in natural coordinates [-1, 1]
        g = 1.0 / np.sqrt(3.0)  # ±1/√3
        self.gauss_points = np.array([
            [-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g],
            [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g]
        ])
        
        # Weights (all equal for 2x2x2 Gauss quadrature)
        self.gauss_weights = np.ones(8)
    
    def _compute_material_matrix(self) -> np.ndarray:
        """
        Compute the material matrix D for 3D elasticity.
        
        Returns:
        --------
        D : np.ndarray
            6x6 material matrix relating stress to strain
        """
        # Material constants
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        
        D = np.zeros((6, 6))
        
        # Diagonal terms
        D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - self.nu)
        
        # Off-diagonal terms (normal stresses)
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = factor * self.nu
        
        # Shear terms
        shear_factor = self.E / (2 * (1 + self.nu))
        D[3, 3] = D[4, 4] = D[5, 5] = shear_factor
        
        return D
    
    def _shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Compute Q8 shape functions at natural coordinates (xi, eta, zeta).
        
        Parameters:
        -----------
        xi, eta, zeta : float
            Natural coordinates in [-1, 1]
            
        Returns:
        --------
        N : np.ndarray
            Shape function values at the 8 nodes
        """
        N = np.zeros(8)
        
        # Trilinear shape functions
        N[0] = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)
        N[1] = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)
        N[2] = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)
        N[3] = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)
        N[4] = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)
        N[5] = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)
        N[6] = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)
        N[7] = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
        
        return N
    
    def _shape_function_derivatives(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Compute derivatives of Q8 shape functions with respect to natural coordinates.
        
        Returns:
        --------
        dN : np.ndarray
            3x8 array where dN[i,j] = ∂N_j/∂ξ_i
        """
        dN = np.zeros((3, 8))
        
        # Derivatives with respect to xi
        dN[0, 0] = -0.125 * (1 - eta) * (1 - zeta)
        dN[0, 1] = 0.125 * (1 - eta) * (1 - zeta)
        dN[0, 2] = 0.125 * (1 + eta) * (1 - zeta)
        dN[0, 3] = -0.125 * (1 + eta) * (1 - zeta)
        dN[0, 4] = -0.125 * (1 - eta) * (1 + zeta)
        dN[0, 5] = 0.125 * (1 - eta) * (1 + zeta)
        dN[0, 6] = 0.125 * (1 + eta) * (1 + zeta)
        dN[0, 7] = -0.125 * (1 + eta) * (1 + zeta)
        
        # Derivatives with respect to eta
        dN[1, 0] = -0.125 * (1 - xi) * (1 - zeta)
        dN[1, 1] = -0.125 * (1 + xi) * (1 - zeta)
        dN[1, 2] = 0.125 * (1 + xi) * (1 - zeta)
        dN[1, 3] = 0.125 * (1 - xi) * (1 - zeta)
        dN[1, 4] = -0.125 * (1 - xi) * (1 + zeta)
        dN[1, 5] = -0.125 * (1 + xi) * (1 + zeta)
        dN[1, 6] = 0.125 * (1 + xi) * (1 + zeta)
        dN[1, 7] = 0.125 * (1 - xi) * (1 + zeta)
        
        # Derivatives with respect to zeta
        dN[2, 0] = -0.125 * (1 - xi) * (1 - eta)
        dN[2, 1] = -0.125 * (1 + xi) * (1 - eta)
        dN[2, 2] = -0.125 * (1 + xi) * (1 + eta)
        dN[2, 3] = -0.125 * (1 - xi) * (1 + eta)
        dN[2, 4] = 0.125 * (1 - xi) * (1 - eta)
        dN[2, 5] = 0.125 * (1 + xi) * (1 - eta)
        dN[2, 6] = 0.125 * (1 + xi) * (1 + eta)
        dN[2, 7] = 0.125 * (1 - xi) * (1 + eta)
        
        return dN
    
    def _compute_jacobian(self, elem_coords: np.ndarray, dN: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute Jacobian matrix and its determinant.
        
        Parameters:
        -----------
        elem_coords : np.ndarray
            8x3 array of element node coordinates
        dN : np.ndarray
            3x8 array of shape function derivatives
            
        Returns:
        --------
        J : np.ndarray
            3x3 Jacobian matrix
        det_J : float
            Determinant of Jacobian
        """
        J = dN @ elem_coords  # 3x8 @ 8x3 = 3x3
        det_J = np.linalg.det(J)
        
        if det_J <= 0:
            raise ValueError(f"Negative or zero Jacobian determinant: {det_J}")
        
        return J, det_J
    
    def _compute_B_matrix(self, dN_global: np.ndarray) -> np.ndarray:
        """
        Compute the strain-displacement matrix B.
        
        Parameters:
        -----------
        dN_global : np.ndarray
            3x8 array of shape function derivatives in global coordinates
            
        Returns:
        --------
        B : np.ndarray
            6x24 strain-displacement matrix
        """
        B = np.zeros((6, 24))  # 6 strain components, 24 DOFs (8 nodes × 3 DOFs)
        
        for i in range(8):
            # Node i contributes to columns 3*i, 3*i+1, 3*i+2
            col = 3 * i
            
            # Normal strains
            B[0, col] = dN_global[0, i]      # ∂u/∂x
            B[1, col + 1] = dN_global[1, i]  # ∂v/∂y
            B[2, col + 2] = dN_global[2, i]  # ∂w/∂z
            
            # Shear strains
            B[3, col] = dN_global[1, i]      # ∂u/∂y
            B[3, col + 1] = dN_global[0, i]  # ∂v/∂x
            
            B[4, col + 1] = dN_global[2, i]  # ∂v/∂z
            B[4, col + 2] = dN_global[1, i]  # ∂w/∂y
            
            B[5, col] = dN_global[2, i]      # ∂u/∂z
            B[5, col + 2] = dN_global[0, i]  # ∂w/∂x
        
        return B
    
    def _compute_element_stiffness(self, elem_id: int) -> np.ndarray:
        """
        Compute element stiffness matrix for element elem_id.
        
        Parameters:
        -----------
        elem_id : int
            Element index
            
        Returns:
        --------
        K_elem : np.ndarray
            24x24 element stiffness matrix
        """
        # Get element nodes
        elem_coords = self.mesh.coordinates[self.mesh.elements[elem_id]]
        
        K_elem = np.zeros((24, 24))
        
        # Integrate using Gauss quadrature
        for gp in range(8):  # 8 Gauss points
            xi, eta, zeta = self.gauss_points[gp]
            weight = self.gauss_weights[gp]
            
            # Shape function derivatives in natural coordinates
            dN = self._shape_function_derivatives(xi, eta, zeta)
            
            # Jacobian
            J, det_J = self._compute_jacobian(elem_coords, dN)
            
            # Shape function derivatives in global coordinates
            dN_global = np.linalg.solve(J, dN)
            
            # Strain-displacement matrix
            B = self._compute_B_matrix(dN_global)
            
            # Add contribution to element stiffness matrix
            K_elem += B.T @ self.D @ B * det_J * weight
        
        return K_elem
    
    def _compute_element_mass(self, elem_id: int) -> np.ndarray:
        """
        Compute element mass matrix for element elem_id.
        
        Parameters:
        -----------
        elem_id : int
            Element index
            
        Returns:
        --------
        M_elem : np.ndarray
            24x24 element mass matrix
        """
        # Get element nodes
        elem_nodes = self.mesh.coordinates[self.mesh.elements[elem_id]]
        
        M_elem = np.zeros((24, 24))
        
        # Integrate using Gauss quadrature
        for gp in range(8):  # 8 Gauss points
            xi, eta, zeta = self.gauss_points[gp]
            weight = self.gauss_weights[gp]
            
            # Shape functions
            N = self._shape_functions(xi, eta, zeta)
            
            # Shape function derivatives for Jacobian
            dN = self._shape_function_derivatives(xi, eta, zeta)
            
            # Jacobian
            J, det_J = self._compute_jacobian(elem_nodes, dN)
            
            # Create N matrix for 3D (3 DOFs per node)
            N_matrix = np.zeros((3, 24))
            for i in range(8):
                N_matrix[0, 3*i] = N[i]      # u displacement
                N_matrix[1, 3*i + 1] = N[i]  # v displacement
                N_matrix[2, 3*i + 2] = N[i]  # w displacement
            
            # Add contribution to element mass matrix
            M_elem += self.rho * N_matrix.T @ N_matrix * det_J * weight
        
        return M_elem
    

    def get_element_dof_indices(self, element_idx):
        """
        Return the global displacement vector indices for the nodes of the element.
        """
        # Get global DOF indices for this element
        elem_nodes = self.mesh.elements[element_idx]
        dof_indices = []
        for node in elem_nodes:
            dof_indices.extend([3*node, 3*node+1, 3*node+2])
        return dof_indices
    

    def assemble_stiffness_matrix(self) -> csr_matrix:
        """
        Assemble the global stiffness matrix.
        
        Returns:
        --------
        K_global : csr_matrix
            Global stiffness matrix in sparse format
        """
        print("Assembling stiffness matrix...")
        
        # Use lil_matrix for efficient assembly
        K_global = lil_matrix((self.n_dofs, self.n_dofs))
        
        for elem_idx in range(self.mesh.n_elements):
            if elem_idx % 100 == 0:
                print(f"  Processing element {elem_idx}/{self.mesh.n_elements}")
            
            # Compute element stiffness matrix
            K_elem = self._compute_element_stiffness(elem_idx)
            dof_indices = self.get_element_dof_indices(elem_idx)
            
            # Assemble into global matrix
            for i, glob_i in enumerate(dof_indices):
                for j, glob_j in enumerate(dof_indices):
                    K_global[glob_i, glob_j] += K_elem[i, j]
        
        self.K_global = K_global.tocsr()
        print("Stiffness matrix assembly complete.")
        return self.K_global
    
    def assemble_mass_matrix(self) -> csr_matrix:
        """
        Assemble the global mass matrix.
        
        Returns:
        --------
        M_global : csr_matrix
            Global mass matrix in sparse format
        """
        print("Assembling mass matrix...")
        
        # Use lil_matrix for efficient assembly
        M_global = lil_matrix((self.n_dofs, self.n_dofs))
        
        for elem_idx in range(self.mesh.n_elements):
            if elem_idx % 100 == 0:
                print(f"  Processing element {elem_idx}/{self.mesh.n_elements}")
            
            # Compute element mass matrix
            M_elem = self._compute_element_mass(elem_idx)
            dof_indices = self.get_element_dof_indices(elem_idx)

            # Assemble into global matrix
            for i, glob_i in enumerate(dof_indices):
                for j, glob_j in enumerate(dof_indices):
                    M_global[glob_i, glob_j] += M_elem[i, j]
        
        self.M_global = M_global.tocsr()
        print("Mass matrix assembly complete.")
        return self.M_global
    
    def assemble_matrices(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Assemble both stiffness and mass matrices.
        
        Returns:
        --------
        K_global, M_global : tuple of csr_matrix
            Global stiffness and mass matrices
        """
        K = self.assemble_stiffness_matrix()
        M = self.assemble_mass_matrix()
        return K, M
    

    def compute_element_stresses(self, element_idx, element_displacements):
        """
        Compute stresses at Gauss points for a single 3D hexahedral bilinear element (H8)
        
        Args:
            element_coords: Element node coordinates (8, 3) - [x, y, z]
            element_displacements: Element displacements (24,) - [u1,v1,w1,u2,v2,w2,...]
            D_matrix: Constitutive matrix (6, 6) for 3D elasticity
            
        Returns:
            stresses: Stresses at Gauss points (n_gauss, 6) - [σx, σy, σz, τxy, τyz, τxz]
            gauss_coords: Global coordinates of Gauss points (n_gauss, 3)
        """
        
        element_stresses = np.zeros((len(self.gauss_weights), 6))
        element_coords = self.mesh.coordinates[self.mesh.elements[element_idx]]
        
        for i, (xi, eta, zeta) in enumerate(self.gauss_points):
            # H8 shape functions and derivatives

            dN = self._shape_function_derivatives(xi, eta, zeta)
            # Jacobian matrix (3x3)
            J, _ = self._compute_jacobian(element_coords, dN)
                        
            # Derivatives in global coordinates
            # Shape function derivatives in global coordinates
            dN_global = np.linalg.solve(J, dN)
            # Strain-displacement matrix B (6x24)
            B = self._compute_B_matrix(dN_global)
            
            # Compute strains
            element_strains = B @ element_displacements
            
            # Compute stresses
            element_stresses[i] = self.D @ element_strains
            
        return element_stresses

    
    def compute_average_von_mises_stress(self, element_stresses):
        """
        Compute average von Mises stress for an element
        Two approaches: compute von Mises at each Gauss point and then average
        
        Args:
            stresses: Stress components at Gauss points (n_gauss, 6) - [σx, σy, σz, τxy, τyz, τxz]
            
        Returns:
            avg_von_mises_method: Average of von Mises stresses at Gauss points
        """
        
        von_mises_gauss = compute_von_mises_stress(element_stresses)
        avg_von_mises = np.mean(von_mises_gauss)
        
        return avg_von_mises

     
    def compute_global_von_mises_stresses(self, displacement_vector):
        """
        Compute average von Mises stress for each element in the mesh.

        Parameters:
            displacement_vector : (n_nodes * 3,) ndarray
                Global displacement vector.

        Returns:
            von_mises_stresses : (n_elements,) ndarray
                Average von Mises stress per element.
        """
        von_mises_stresses = np.zeros(self.mesh.n_elements)

        for elem_idx in range(self.mesh.n_elements):
            elem_nodes = self.mesh.elements[elem_idx]
            u_elem = displacement_vector[elem_nodes].reshape(-1)  # local displacement vector

            # External functions (assumed provided)
            stresses_at_gauss_points = self.compute_element_stresses(elem_idx, u_elem)
            avg_vm_stress = self.compute_average_von_mises_stress(stresses_at_gauss_points)

            von_mises_stresses[elem_idx] = avg_vm_stress

        return von_mises_stresses
    

    def assemble_surface_traction_force(self, traction_vector, traction_plane='x_max'):
        """
        Assemble force vector for surface traction applied to lateral surface of Q8 solid elements.
        
        Parameters:
        -----------
        surface_elements : list
            List of surface element connectivity (4 nodes per face for Q8 hex elements)
        traction_vector : array_like
            Applied traction vector [tx, ty, tz] (force per unit area)
        traction_plane : str
            Which plane to fix ('z_min', 'z_max', 'x_min', 'x_max', 'y_min', 'y_max')
            
        Returns:
        --------
        F : ndarray
            Global force vector
        """
        
        # Initialize global force vector
 
        F = np.zeros(self.n_dofs)
        
        # Convert traction to numpy array
        traction = np.array(traction_vector)
        
        # Gauss quadrature points and weights for 2D surface integration (2x2)
        g = 1.0 / np.sqrt(3.0)  # ±1/√3
        gauss_points = np.array([[-g, -g],
                                 [+g, -g],
                                 [+g, +g],
                                 [-g, +g]])
        
        gauss_weights = np.array([1.0, 1.0, 1.0, 1.0])

        tolerance = 1e-10
        if traction_plane == 'z_min':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 2] - self.mesh.origin[2]) < tolerance)[0]
        elif traction_plane == 'z_max':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 2] - (self.mesh.origin[2] + self.mesh.Lz)) < tolerance)[0]
        elif traction_plane == 'x_min':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 0] - self.mesh.origin[0]) < tolerance)[0]
        elif traction_plane == 'x_max':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 0] - (self.mesh.origin[0] + self.mesh.Lx)) < tolerance)[0]
        elif traction_plane == 'y_min':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 1] - self.mesh.origin[1]) < tolerance)[0]
        elif traction_plane == 'y_max':
            self.traction_nodes = np.where(np.abs(self.mesh.coordinates[:, 1] - (self.mesh.origin[1] + self.mesh.Ly)) < tolerance)[0]
        else:
            raise ValueError(f"Unknown traction_plane: {traction_plane}")
        
        surface_elements = self.mesh.get_faces_from_nodes(self.traction_nodes)
        # Process each surface element
        for surface_elem in surface_elements:
            # Get nodal coordinates for the 4-node surface element
            surface_nodes = np.array(surface_elem)  # 4 nodes defining the surface
            coords = np.array([self.mesh.coordinates[node_id] for node_id in surface_nodes])
            
            # Initialize element force vector (4 nodes × 3 DOF = 12 DOF)
            f_elem = np.zeros(12)
            
            # Numerical integration over surface element
            for gp, (xi, eta) in enumerate(gauss_points):
                weight = gauss_weights[gp]
                
                # Shape functions for 4-node quadrilateral (bilinear)
                N = self._compute_surface_shape_functions(xi, eta)
                
                # Shape function derivatives in natural coordinates
                dN_dxi, dN_deta = self._compute_surface_shape_derivatives(xi, eta)
                
                # Compute Jacobian matrix for surface element
                dx_dxi = np.dot(dN_dxi, coords[:, 0])
                dy_dxi = np.dot(dN_dxi, coords[:, 1])
                dz_dxi = np.dot(dN_dxi, coords[:, 2])
                
                dx_deta = np.dot(dN_deta, coords[:, 0])
                dy_deta = np.dot(dN_deta, coords[:, 1])
                dz_deta = np.dot(dN_deta, coords[:, 2])
                
                # Surface tangent vectors
                t1 = np.array([dx_dxi, dy_dxi, dz_dxi])
                t2 = np.array([dx_deta, dy_deta, dz_deta])
                
                # Surface normal vector (cross product)
                normal = np.cross(t1, t2)
                
                # Jacobian determinant (surface area element)
                J_det = np.linalg.norm(normal)
                
                # Compute element force contribution at this Gauss point
                for i in range(4):  # 4 nodes per surface element
                    # Force contribution for node i
                    force_contrib = N[i] * traction * J_det * weight
                    
                    # Add to element force vector
                    f_elem[i*3:(i+1)*3] += force_contrib
            
            # Assemble element force vector into global force vector
            for i, node_id in enumerate(surface_nodes):
                global_dof_start = node_id * self.dof_per_node
                f_elem_start = i * self.dof_per_node
                F[global_dof_start:global_dof_start + self.dof_per_node] += \
                    f_elem[f_elem_start:f_elem_start + self.dof_per_node]
        
        return F
    

    def _compute_surface_shape_functions(self, xi, eta):
        """
        Compute shape functions for 4-node quadrilateral surface element.
        
        Parameters:
        -----------
        xi, eta : float
            Natural coordinates
            
        Returns:
        --------
        N : ndarray
            Shape function values at (xi, eta)
        """
        N = np.zeros(4)
        N[0] = 0.25 * (1 - xi) * (1 - eta)  # Node 1
        N[1] = 0.25 * (1 + xi) * (1 - eta)  # Node 2
        N[2] = 0.25 * (1 + xi) * (1 + eta)  # Node 3
        N[3] = 0.25 * (1 - xi) * (1 + eta)  # Node 4
        
        return N


    def _compute_surface_shape_derivatives(self, xi, eta):
        """
        Compute shape function derivatives for 4-node quadrilateral surface element.
        
        Parameters:
        -----------
        xi, eta : float
            Natural coordinates
            
        Returns:
        --------
        dN_dxi, dN_deta : ndarray
            Shape function derivatives with respect to xi and eta
        """
        dN_dxi = np.zeros(4)
        dN_deta = np.zeros(4)
        
        # Derivatives with respect to xi
        dN_dxi[0] = -0.25 * (1 - eta)
        dN_dxi[1] =  0.25 * (1 - eta)
        dN_dxi[2] =  0.25 * (1 + eta)
        dN_dxi[3] = -0.25 * (1 + eta)
        
        # Derivatives with respect to eta
        dN_deta[0] = -0.25 * (1 - xi)
        dN_deta[1] = -0.25 * (1 + xi)
        dN_deta[2] =  0.25 * (1 + xi)
        dN_deta[3] =  0.25 * (1 - xi)
        
        return dN_dxi, dN_deta


    def get_matrix_info(self) -> dict:
        """Get information about the assembled matrices."""
        info = {
            'n_dofs': self.n_dofs,
            'n_elements': self.mesh.n_elements,
            'material_properties': {
                'E': self.E,
                'nu': self.nu,
                'rho': self.rho
            }
        }
        
        if self.K_global is not None:
            info['stiffness_nnz'] = self.K_global.nnz
            info['stiffness_sparsity'] = self.K_global.nnz / (self.n_dofs ** 2)
        
        if self.M_global is not None:
            info['mass_nnz'] = self.M_global.nnz
            info['mass_sparsity'] = self.M_global.nnz / (self.n_dofs ** 2)
        
        return info


if __name__ == "__main__":   
    # Create a simple mesh
    mesh = StructuredHexMesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
    
    # Material properties for steel
    material_props = {
        'E': 210e9,      # Young's modulus (Pa)
        'nu': 0.3,       # Poisson's ratio
        'rho': 7850.0    # Density (kg/m³)
    }
    
    # Create assembler
    assembler = Q8FiniteElementAssembler(mesh, material_props)
    
    # Assemble matrices
    K, M = assembler.assemble_matrices()
    
    # Print information
    print("\nMatrix Assembly Results:")
    info = assembler.get_matrix_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nStiffness matrix shape: {K.shape}")
    print(f"Mass matrix shape: {M.shape}")
    print(f"Stiffness matrix condition number estimate: {np.linalg.cond(K.toarray()[:100, :100]):.2e}")