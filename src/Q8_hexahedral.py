import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix, lil_matrix
import time
from post_processing.plot_3d import FEMPostProcessor

class Q8HexMesh:
    """
    Class for structured hexahedral mesh generation and FEM assembly using Q8 elements
    """
    
    def __init__(self, nx, ny, nz, lx=1.0, ly=1.0, lz=1.0):
        """
        Initialize mesh parameters
        
        Parameters:
        nx, ny, nz: number of elements in x, y, z directions
        lx, ly, lz: domain dimensions
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lx, self.ly, self.lz = lx, ly, lz
        
        # Number of nodes in each direction
        self.nnx = nx + 1
        self.nny = ny + 1
        self.nnz = nz + 1
        
        # Total number of nodes and elements
        self.n_nodes = self.nnx * self.nny * self.nnz
        self.n_elements = nx * ny * nz
        
        # Generate mesh
        self.nodes = self._generate_nodes()
        self.elements = self._generate_elements()
        
        # Material properties (default values)
        self.E = 210e9  # Young's modulus (Pa)
        self.nu = 0.3   # Poisson's ratio
        self.rho = 7850  # Density (kg/m³)
        
        # Gauss quadrature points and weights for 2x2x2 integration
        self._setup_gauss_points()
        
    def _generate_nodes(self):
        """Generate node coordinates"""
        nodes = np.zeros((self.n_nodes, 3))
        
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        dz = self.lz / self.nz
        
        node_id = 0
        for k in range(self.nnz):
            for j in range(self.nny):
                for i in range(self.nnx):
                    nodes[node_id] = [i * dx, j * dy, k * dz]
                    node_id += 1
                    
        return nodes
    
    def _generate_elements(self):
        """Generate element connectivity (Q8 hexahedral elements)"""
        elements = np.zeros((self.n_elements, 8), dtype=int)
        
        elem_id = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # Node indices for hexahedral element
                    n1 = k * self.nnx * self.nny + j * self.nnx + i
                    n2 = k * self.nnx * self.nny + j * self.nnx + (i + 1)
                    n3 = k * self.nnx * self.nny + (j + 1) * self.nnx + (i + 1)
                    n4 = k * self.nnx * self.nny + (j + 1) * self.nnx + i
                    n5 = (k + 1) * self.nnx * self.nny + j * self.nnx + i
                    n6 = (k + 1) * self.nnx * self.nny + j * self.nnx + (i + 1)
                    n7 = (k + 1) * self.nnx * self.nny + (j + 1) * self.nnx + (i + 1)
                    n8 = (k + 1) * self.nnx * self.nny + (j + 1) * self.nnx + i
                    
                    elements[elem_id] = [n1, n2, n3, n4, n5, n6, n7, n8]
                    elem_id += 1
                    
        return elements
    
    def _setup_gauss_points(self):
        """Setup Gauss quadrature points and weights for 2x2x2 integration"""
        gp = 1.0 / np.sqrt(3.0)  # Gauss point coordinate
        
        # Gauss points in natural coordinates (ξ, η, ζ)
        self.gauss_points = np.array([
            [-gp, -gp, -gp], [gp, -gp, -gp], [gp, gp, -gp], [-gp, gp, -gp],
            [-gp, -gp, gp], [gp, -gp, gp], [gp, gp, gp], [-gp, gp, gp]
        ])
        
        # Weights (all equal for 2x2x2 integration)
        self.gauss_weights = np.ones(8)
    
    def shape_functions(self, xi, eta, zeta):
        """
        Compute Q8 shape functions and their derivatives
        
        Parameters:
        xi, eta, zeta: natural coordinates (-1 to 1)
        
        Returns:
        N: shape functions
        dN_dxi: derivatives w.r.t. natural coordinates
        """
        # Shape functions
        N = np.zeros(8)
        N[0] = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)
        N[1] = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)
        N[2] = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)
        N[3] = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)
        N[4] = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)
        N[5] = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)
        N[6] = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)
        N[7] = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
        
        # Derivatives w.r.t. natural coordinates
        dN_dxi = np.zeros((3, 8))
        
        # dN/dxi
        dN_dxi[0, 0] = -0.125 * (1 - eta) * (1 - zeta)
        dN_dxi[0, 1] = 0.125 * (1 - eta) * (1 - zeta)
        dN_dxi[0, 2] = 0.125 * (1 + eta) * (1 - zeta)
        dN_dxi[0, 3] = -0.125 * (1 + eta) * (1 - zeta)
        dN_dxi[0, 4] = -0.125 * (1 - eta) * (1 + zeta)
        dN_dxi[0, 5] = 0.125 * (1 - eta) * (1 + zeta)
        dN_dxi[0, 6] = 0.125 * (1 + eta) * (1 + zeta)
        dN_dxi[0, 7] = -0.125 * (1 + eta) * (1 + zeta)
        
        # dN/deta
        dN_dxi[1, 0] = -0.125 * (1 - xi) * (1 - zeta)
        dN_dxi[1, 1] = -0.125 * (1 + xi) * (1 - zeta)
        dN_dxi[1, 2] = 0.125 * (1 + xi) * (1 - zeta)
        dN_dxi[1, 3] = 0.125 * (1 - xi) * (1 - zeta)
        dN_dxi[1, 4] = -0.125 * (1 - xi) * (1 + zeta)
        dN_dxi[1, 5] = -0.125 * (1 + xi) * (1 + zeta)
        dN_dxi[1, 6] = 0.125 * (1 + xi) * (1 + zeta)
        dN_dxi[1, 7] = 0.125 * (1 - xi) * (1 + zeta)
        
        # dN/dzeta
        dN_dxi[2, 0] = -0.125 * (1 - xi) * (1 - eta)
        dN_dxi[2, 1] = -0.125 * (1 + xi) * (1 - eta)
        dN_dxi[2, 2] = -0.125 * (1 + xi) * (1 + eta)
        dN_dxi[2, 3] = -0.125 * (1 - xi) * (1 + eta)
        dN_dxi[2, 4] = 0.125 * (1 - xi) * (1 - eta)
        dN_dxi[2, 5] = 0.125 * (1 + xi) * (1 - eta)
        dN_dxi[2, 6] = 0.125 * (1 + xi) * (1 + eta)
        dN_dxi[2, 7] = 0.125 * (1 - xi) * (1 + eta)
        
        return N, dN_dxi
    
    def jacobian_matrix(self, elem_nodes, dN_dxi):
        """
        Compute Jacobian matrix for coordinate transformation
        
        Parameters:
        elem_nodes: coordinates of element nodes (8x3)
        dN_dxi: shape function derivatives w.r.t. natural coordinates (3x8)
        
        Returns:
        J: Jacobian matrix (3x3)
        det_J: Jacobian determinant
        """
        J = np.dot(dN_dxi, elem_nodes)
        det_J = np.linalg.det(J)
        return J, det_J
    
    def strain_displacement_matrix(self, dN_dx):
        """
        Compute strain-displacement matrix B
        
        Parameters:
        dN_dx: shape function derivatives w.r.t. global coordinates (3x8)
        
        Returns:
        B: strain-displacement matrix (6x24)
        """
        B = np.zeros((6, 24))
        
        for i in range(8):
            col = 3 * i
            B[0, col] = dN_dx[0, i]      # εxx
            B[1, col + 1] = dN_dx[1, i]  # εyy
            B[2, col + 2] = dN_dx[2, i]  # εzz
            B[3, col] = dN_dx[1, i]      # γxy
            B[3, col + 1] = dN_dx[0, i]
            B[4, col + 1] = dN_dx[2, i]  # γyz
            B[4, col + 2] = dN_dx[1, i]
            B[5, col] = dN_dx[2, i]      # γxz
            B[5, col + 2] = dN_dx[0, i]
            
        return B
    
    def constitutive_matrix(self):
        """
        Compute constitutive matrix D for isotropic material
        
        Returns:
        D: constitutive matrix (6x6)
        """
        D = np.zeros((6, 6))
        
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        
        D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - self.nu)
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = factor * self.nu
        D[3, 3] = D[4, 4] = D[5, 5] = factor * (1 - 2 * self.nu) / 2
        
        return D
    
    def element_stiffness_matrix(self, elem_id):
        """
        Compute element stiffness matrix
        
        Parameters:
        elem_id: element index
        
        Returns:
        Ke: element stiffness matrix (24x24)
        """
        # Get element nodes
        elem_nodes = self.nodes[self.elements[elem_id]]
        
        # Initialize element stiffness matrix
        Ke = np.zeros((24, 24))
        
        # Constitutive matrix
        D = self.constitutive_matrix()
        
        # Numerical integration using Gauss quadrature
        for gp in range(8):
            xi, eta, zeta = self.gauss_points[gp]
            weight = self.gauss_weights[gp]
            
            # Shape functions and derivatives
            N, dN_dxi = self.shape_functions(xi, eta, zeta)
            
            # Jacobian matrix and determinant
            J, det_J = self.jacobian_matrix(elem_nodes, dN_dxi)
            
            # Shape function derivatives w.r.t. global coordinates
            dN_dx = np.linalg.solve(J, dN_dxi)
            
            # Strain-displacement matrix
            B = self.strain_displacement_matrix(dN_dx)
            
            # Add contribution to stiffness matrix
            Ke += np.dot(np.dot(B.T, D), B) * det_J * weight
            
        return Ke
    
    def element_mass_matrix(self, elem_id):
        """
        Compute element mass matrix
        
        Parameters:
        elem_id: element index
        
        Returns:
        Me: element mass matrix (24x24)
        """
        # Get element nodes
        elem_nodes = self.nodes[self.elements[elem_id]]
        
        # Initialize element mass matrix
        Me = np.zeros((24, 24))
        
        # Numerical integration using Gauss quadrature
        for gp in range(8):
            xi, eta, zeta = self.gauss_points[gp]
            weight = self.gauss_weights[gp]
            
            # Shape functions and derivatives
            N, dN_dxi = self.shape_functions(xi, eta, zeta)
            
            # Jacobian matrix and determinant
            J, det_J = self.jacobian_matrix(elem_nodes, dN_dxi)
            
            # Assemble mass matrix contribution
            for i in range(8):
                for j in range(8):
                    for dof in range(3):
                        row = 3 * i + dof
                        col = 3 * j + dof
                        Me[row, col] += self.rho * N[i] * N[j] * det_J * weight
                        
        return Me
    
    def assemble_global_matrices(self):
        """
        Assemble global stiffness and mass matrices
        
        Returns:
        K_global: global stiffness matrix
        M_global: global mass matrix
        """
        print("Assembling global matrices...")
        start_time = time.time()
        
        # Initialize global matrices
        dof_total = 3 * self.n_nodes
        K_global = lil_matrix((dof_total, dof_total))
        M_global = lil_matrix((dof_total, dof_total))
        
        # Assembly loop
        for elem_id in range(self.n_elements):
            if elem_id % 100 == 0:
                print(f"Processing element {elem_id}/{self.n_elements}")
                
            # Element matrices
            Ke = self.element_stiffness_matrix(elem_id)
            Me = self.element_mass_matrix(elem_id)
            
            # Element DOFs
            elem_nodes = self.elements[elem_id]
            elem_dofs = np.zeros(24, dtype=int)
            for i, node in enumerate(elem_nodes):
                elem_dofs[3*i:3*i+3] = [3*node, 3*node+1, 3*node+2]
            
            # Add to global matrices
            for i in range(24):
                for j in range(24):
                    K_global[elem_dofs[i], elem_dofs[j]] += Ke[i, j]
                    M_global[elem_dofs[i], elem_dofs[j]] += Me[i, j]
        
        end_time = time.time()
        print(f"Assembly completed in {end_time - start_time:.2f} seconds")
        
        return K_global.tocsr(), M_global.tocsr()
    
    def visualize_mesh(self):
        """Visualize the mesh"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], 
                  c='red', s=20, alpha=0.6)
        
        # Plot element edges (sample elements for visualization)
        sample_elements = min(self.n_elements, 50)  # Limit for visualization
        for i in range(0, sample_elements, max(1, sample_elements//20)):
            elem_nodes = self.nodes[self.elements[i]]
            
            # Define the 12 edges of a hexahedron
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            for edge in edges:
                points = elem_nodes[edge]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Q8 Hexahedral Mesh ({self.nx}×{self.ny}×{self.nz} elements)')
        
        plt.tight_layout()
        plt.show()
    
    def get_clamped_dofs(self):
        """
        Get DOFs for clamped boundary condition on z=0 plane
        All nodes with z=0 are completely fixed (all 3 DOFs)
        
        Returns:
        fixed_dofs: array of fixed DOF indices
        """
        fixed_dofs = []
        
        for node_id in range(self.n_nodes):
            if abs(self.nodes[node_id, 2]) < 1e-10:  # z ≈ 0
                # Fix all 3 DOFs for this node
                fixed_dofs.extend([3*node_id, 3*node_id+1, 3*node_id+2])
        
        return np.array(fixed_dofs, dtype=int)
    
    def plot_mode_shapes(self, eigenvals, eigenvecs, free_dofs, n_modes=6):
        """
        Plot the first n_modes eigenfunction (mode shapes)
        
        Parameters:
        eigenvals: eigenvalues
        eigenvecs: eigenvectors (mode shapes)
        free_dofs: indices of free DOFs
        n_modes: number of modes to plot
        """
        print("\nPlotting mode shapes...")
        
        # Calculate natural frequencies
        frequencies = np.sqrt(eigenvals) / (2 * np.pi)
        
        # Create subplot grid
        n_cols = 3
        n_rows = (n_modes + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(15, 5*n_rows))
        
        for mode in range(min(n_modes, len(eigenvals))):
            ax = fig.add_subplot(n_rows, n_cols, mode+1, projection='3d')
            
            # Reconstruct full displacement vector
            full_displacement = np.zeros(3 * self.n_nodes)
            full_displacement[free_dofs] = eigenvecs[:, mode]
            
            # Reshape to get displacements for each node
            displacements = full_displacement.reshape((-1, 3))
            
            # Scale factor for visualization
            max_disp = np.max(np.abs(displacements))
            if max_disp > 0:
                scale_factor = 0.1 * min(self.lx, self.ly, self.lz) / max_disp
            else:
                scale_factor = 1.0
            
            # Deformed node positions
            deformed_nodes = self.nodes + scale_factor * displacements
            
            # Color nodes based on displacement magnitude
            disp_magnitude = np.linalg.norm(displacements, axis=1)
            
            # Plot deformed configuration
            scatter = ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], deformed_nodes[:, 2],
                               c=disp_magnitude, cmap='viridis', s=30, alpha=0.8)
            
            # Plot original nodes (fixed nodes in red)
            fixed_nodes = []
            for node_id in range(self.n_nodes):
                if abs(self.nodes[node_id, 2]) < 1e-10:
                    fixed_nodes.append(node_id)
            
            if fixed_nodes:
                ax.scatter(self.nodes[fixed_nodes, 0], self.nodes[fixed_nodes, 1],
                          self.nodes[fixed_nodes, 2], c='red', s=40, marker='s', alpha=0.9)
            
            # Add some element edges for structure visualization
            sample_elements = min(self.n_elements, 20)
            for i in range(0, sample_elements, max(1, sample_elements//10)):
                elem_nodes_orig = self.nodes[self.elements[i]]
                elem_nodes_def = deformed_nodes[self.elements[i]]
                
                # Define some edges of hexahedron for visualization
                edges = [[0, 1], [1, 2], [2, 3], [3, 0],  # bottom
                        [4, 5], [5, 6], [6, 7], [7, 4],   # top
                        [0, 4], [1, 5], [2, 6], [3, 7]]   # vertical
                
                for edge in edges[::2]:  # Plot every other edge to reduce clutter
                    points = elem_nodes_def[edge]
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                             'b-', alpha=0.3, linewidth=0.5)
            
            # Formatting
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Mode {mode+1}: {frequencies[mode]:.2f} Hz')
            
            # Set equal aspect ratio
            max_range = max(self.lx, self.ly, self.lz) * 1.1
            ax.set_xlim([-0.1*max_range, max_range])
            ax.set_ylim([-0.1*max_range, max_range])
            ax.set_zlim([0, max_range])
            
            # Add colorbar
            if mode == 0:  # Only add colorbar to first subplot
                plt.colorbar(scatter, ax=ax, shrink=0.8, label='|Displacement|')
        
        plt.tight_layout()
        plt.suptitle(f'Mode Shapes - Clamped at z=0', fontsize=16, y=0.98)
        plt.show()
        
        # Additional analysis plot
        self._plot_frequency_analysis(frequencies)
    
    def _plot_frequency_analysis(self, frequencies):
        """Plot frequency analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Frequency spectrum
        ax1.stem(range(1, len(frequencies)+1), frequencies, basefmt=' ')
        ax1.set_xlabel('Mode Number')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Natural Frequencies')
        ax1.grid(True, alpha=0.3)
        
        # Frequency ratios
        if len(frequencies) > 1:
            ratios = frequencies[1:] / frequencies[0]
            ax2.plot(range(2, len(frequencies)+1), ratios, 'o-')
            ax2.set_xlabel('Mode Number')
            ax2.set_ylabel('Frequency Ratio (f_n/f_1)')
            ax2.set_title('Frequency Ratios')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        """Print mesh information"""
        print("="*50)
        print("Q8 HEXAHEDRAL MESH INFORMATION")
        print("="*50)
        print(f"Elements: {self.nx} × {self.ny} × {self.nz} = {self.n_elements}")
        print(f"Nodes: {self.nnx} × {self.nny} × {self.nnz} = {self.n_nodes}")
        print(f"Total DOFs: {3 * self.n_nodes}")
        print(f"Domain size: {self.lx} × {self.ly} × {self.lz}")
        print(f"Material properties:")
        print(f"  Young's modulus: {self.E/1e9:.1f} GPa")
        print(f"  Poisson's ratio: {self.nu}")
        print(f"  Density: {self.rho} kg/m³")
        print("="*50)

    def print_mesh_info(self):
        """Print mesh information"""
        print("="*50)
        print("Q8 HEXAHEDRAL MESH INFORMATION")
        print("="*50)
        print(f"Elements: {self.nx} × {self.ny} × {self.nz} = {self.n_elements}")
        print(f"Nodes: {self.nnx} × {self.nny} × {self.nnz} = {self.n_nodes}")
        print(f"Total DOFs: {3 * self.n_nodes}")
        print(f"Domain size: {self.lx} × {self.ly} × {self.lz}")
        print(f"Material properties:")
        print(f"  Young's modulus: {self.E/1e9:.1f} GPa")
        print(f"  Poisson's ratio: {self.nu}")
        print(f"  Density: {self.rho} kg/m³")
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Create mesh (try different sizes for different effects)
    print("Creating cantilever beam mesh...")
    # Create a more interesting geometry - longer in x direction
    mesh = Q8HexMesh(nx=6, ny=2, nz=2, lx=3.0, ly=1.0, lz=1.0)
    
    postproc = FEMPostProcessor(mesh.nodes, mesh.elements)

    # Test 1: Basic solid plotting
    print("Plotting basic solid...")
    fig1, ax1 = postproc.plot_solid(figsize=(10, 8))
    plt.show()

    # # Print mesh information
    # mesh.print_mesh_info()
    
    # # Assemble matrices
    # K, M = mesh.assemble_global_matrices()
    
    # print(f"\nGlobal stiffness matrix: {K.shape}, nnz = {K.nnz}")
    # print(f"Global mass matrix: {M.shape}, nnz = {M.nnz}")
    
    # # Apply clamped boundary conditions on z=0 plane
    # print("\nApplying clamped boundary conditions on z=0 plane...")
    # fixed_dofs = mesh.get_clamped_dofs()
    # free_dofs = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
    
    # print(f"Fixed DOFs: {len(fixed_dofs)}")
    # print(f"Free DOFs: {len(free_dofs)}")
    
    # # Test: compute some eigenvalues (first few natural frequencies)
    # print("\nComputing first 10 eigenvalues...")
    # try:
    #     from scipy.sparse.linalg import eigsh
        
    #     K_free = K[np.ix_(free_dofs, free_dofs)]
    #     M_free = M[np.ix_(free_dofs, free_dofs)]
        
    #     # Solve generalized eigenvalue problem
    #     eigenvals, eigenvecs = eigsh(K_free, M=M_free, k=10, which='SM')
        
    #     print("Natural frequencies (Hz):")
    #     frequencies = np.sqrt(eigenvals) / (2 * np.pi)
    #     for i, freq in enumerate(frequencies):
    #         print(f"  Mode {i+1}: {freq:.2f} Hz")
        
    #     # Plot first few mode shapes
    #     # mesh.plot_mode_shapes(eigenvals, eigenvecs, free_dofs, n_modes=6)

    #     first_eigenvecs = eigenvecs[:, :6]  # First 6 modes

            
    # except ImportError:
    #     print("scipy.sparse.linalg not available for eigenvalue analysis")
    
    # # Visualize original mesh
    # print("\nVisualizing original mesh...")
    # mesh.visualize_mesh()