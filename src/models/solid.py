import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple
from src.meshing.structured_mesh import StructuredHexMesh



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
        self.n_dofs = 3 * mesh.n_nodes
        
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
    
    def _compute_jacobian(self, elem_nodes: np.ndarray, dN: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute Jacobian matrix and its determinant.
        
        Parameters:
        -----------
        elem_nodes : np.ndarray
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
        J = dN @ elem_nodes  # 3x8 @ 8x3 = 3x3
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
        elem_nodes = self.mesh.coordinates[self.mesh.elements[elem_id]]
        
        K_elem = np.zeros((24, 24))
        
        # Integrate using Gauss quadrature
        for gp in range(8):  # 8 Gauss points
            xi, eta, zeta = self.gauss_points[gp]
            weight = self.gauss_weights[gp]
            
            # Shape function derivatives in natural coordinates
            dN = self._shape_function_derivatives(xi, eta, zeta)
            
            # Jacobian
            J, det_J = self._compute_jacobian(elem_nodes, dN)
            
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
        
        for elem_id in range(self.mesh.n_elements):
            if elem_id % 100 == 0:
                print(f"  Processing element {elem_id}/{self.mesh.n_elements}")
            
            # Compute element stiffness matrix
            K_elem = self._compute_element_stiffness(elem_id)
            
            # Get global DOF indices for this element
            elem_nodes = self.mesh.elements[elem_id]
            dof_indices = []
            for node in elem_nodes:
                dof_indices.extend([3*node, 3*node+1, 3*node+2])
            
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
        
        for elem_id in range(self.mesh.n_elements):
            if elem_id % 100 == 0:
                print(f"  Processing element {elem_id}/{self.mesh.n_elements}")
            
            # Compute element mass matrix
            M_elem = self._compute_element_mass(elem_id)
            
            # Get global DOF indices for this element
            elem_nodes = self.mesh.elements[elem_id]
            dof_indices = []
            for node in elem_nodes:
                dof_indices.extend([3*node, 3*node+1, 3*node+2])
            
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