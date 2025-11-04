import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from src.meshing.structured_mesh import StructuredHexMesh
from src.models.hexahedral_solid import Q8FiniteElementAssembler  

import warnings
warnings.filterwarnings('ignore')
 

class CantileverModalAnalysis:
    """
    Modal analysis of a cantilever beam using Q8 finite elements.
    The beam is fixed at one end (xy plane at z=0) and free at the other end.
    """
    
    def __init__(self, mesh: StructuredHexMesh, assembler: Q8FiniteElementAssembler):
        """
        Initialize cantilever modal analysis.
        
        Parameters:
        -----------
        mesh : StructuredHexMesh
            The hexahedral mesh of the cantilever beam
        assembler : Q8FiniteElementAssembler
            The finite element assembler with matrices
        """
        self.mesh = mesh
        self.assembler = assembler
        self.n_dofs = assembler.n_dofs
        
        # Store boundary conditions
        self.free_dofs = np.arange(self.n_dofs)
        self.fixed_nodes = []
        self.fixed_dofs = []
        
        # Modal analysis results
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        
    def apply_boundary_conditions(self, fix_plane: str = 'z_min'):
        """
        Apply cantilever boundary conditions by fixing one face.
        
        Parameters:
        -----------
        fix_plane : str
            Which plane to fix ('z_min', 'z_max', 'x_min', 'x_max', 'y_min', 'y_max')
        """
        print(f"Applying boundary conditions: fixing {fix_plane} plane...")
        
        # Get nodes on the specified boundary plane
        tolerance = 1e-10
        
        if fix_plane == 'z_min':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 2] - self.mesh.origin[2]) < tolerance)[0]
        elif fix_plane == 'z_max':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 2] - (self.mesh.origin[2] + self.mesh.Lz)) < tolerance)[0]
        elif fix_plane == 'x_min':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 0] - self.mesh.origin[0]) < tolerance)[0]
        elif fix_plane == 'x_max':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 0] - (self.mesh.origin[0] + self.mesh.Lx)) < tolerance)[0]
        elif fix_plane == 'y_min':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 1] - self.mesh.origin[1]) < tolerance)[0]
        elif fix_plane == 'y_max':
            self.fixed_nodes = np.where(np.abs(self.mesh.coordinates[:, 1] - (self.mesh.origin[1] + self.mesh.Ly)) < tolerance)[0]
        else:
            raise ValueError(f"Unknown fix_plane: {fix_plane}")
        
        # Convert node numbers to DOF numbers (3 DOFs per node)
        self.fixed_dofs = []
        for node in self.fixed_nodes:
            self.fixed_dofs.extend([3*node, 3*node+1, 3*node+2])  # u, v, w
        
        self.fixed_dofs = np.array(sorted(self.fixed_dofs))
        
        # Get free DOFs
        all_dofs = np.arange(self.n_dofs)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)
        
        print(f"  Fixed {len(self.fixed_nodes)} nodes ({len(self.fixed_dofs)} DOFs)")
        print(f"  Free DOFs: {len(self.free_dofs)}")
    
    def solve_eigenvalue_problem(self, n_modes: int = 6, sigma: float = 0.0):
        """
        Solve the generalized eigenvalue problem (K - λM)φ = 0 for free vibration.
        
        Parameters:
        -----------
        n_modes : int
            Number of modes to compute
        sigma : float
            Shift parameter for eigenvalue solver (use small positive value to find lowest modes)
            
        Returns:
        --------
        frequencies : np.ndarray
            Natural frequencies in Hz
        mode_shapes : np.ndarray
            Mode shapes (eigenvectors)
        """
        print(f"Solving eigenvalue problem for {n_modes} modes...")
        
        if self.assembler.K_global is None or self.assembler.M_global is None:
            raise ValueError("Matrices not assembled. Call assembler.assemble_matrices() first.")
        
        if len(self.fixed_dofs) == 0:
            raise Warning("No fixed DOFs")
        
        # Extract free DOF submatrices
        print("  Extracting free DOF submatrices...")
        K_free = self.assembler.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        M_free = self.assembler.M_global[np.ix_(self.free_dofs, self.free_dofs)]
        
        print(f"  Free system size: {K_free.shape[0]} x {K_free.shape[1]}")
        print(f"  K_free nnz: {K_free.nnz}, M_free nnz: {M_free.nnz}")
        
        # Ensure matrices are in CSR format for efficient eigenvalue computation
        if not sparse.isspmatrix_csr(K_free):
            K_free = K_free.tocsr()
        if not sparse.isspmatrix_csr(M_free):
            M_free = M_free.tocsr()
        
        # Solve generalized eigenvalue problem: K*phi = lambda*M*phi
        # Use shift-invert mode to find smallest eigenvalues
        print("  Computing eigenvalues and eigenvectors...")
        try:
            if sigma > 0:
                # Shift-invert mode for better convergence to smallest eigenvalues
                eigenvals, eigenvecs = eigsh(K_free, M=M_free, k=n_modes, sigma=sigma, which='LM')
            else:
                # Standard mode - find smallest eigenvalues
                eigenvals, eigenvecs = eigsh(K_free, M=M_free, k=n_modes, which='SM')
        except Exception as e:
            print(f"  Eigenvalue computation failed: {e}")
            print("  Trying with shift-invert mode...")
            eigenvals, eigenvecs = eigsh(K_free, M=M_free, k=min(n_modes, K_free.shape[0]-1), 
                                       sigma=1e-6, which='LM')
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Convert eigenvalues to frequencies
        # ω² = λ, so ω = sqrt(λ), f = ω/(2π)
        frequencies = np.sqrt(np.abs(eigenvals)) / (2 * np.pi)
        
        # Extend eigenvectors to full DOF set
        full_eigenvecs = np.zeros((self.n_dofs, len(eigenvals)))
        full_eigenvecs[self.free_dofs, :] = eigenvecs

        self.eigenvalues = eigenvals
        self.eigenvectors = full_eigenvecs
        self.frequencies = frequencies
        
        print(f"  Successfully computed {len(frequencies)} modes")
        print(f"  Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
        
        return frequencies, full_eigenvecs
    
    def print_modal_results(self, n_modes: int = None):
        """Print modal analysis results in a formatted table."""
        if self.frequencies is None:
            print("No modal analysis results available.")
            return
        
        if n_modes is None:
            n_modes = len(self.frequencies)
        else:
            n_modes = min(n_modes, len(self.frequencies))
        
        print("\n" + "="*60)
        print("MODAL ANALYSIS RESULTS")
        print("="*60)
        print(f"{'Mode':<6} {'Frequency (Hz)':<15} {'Period (s)':<15} {'ω (rad/s)':<15}")
        print("-"*60)
        
        for i in range(n_modes):
            freq = self.frequencies[i]
            period = 1.0 / freq if freq > 1e-12 else np.inf
            omega = 2 * np.pi * freq
            print(f"{i+1:<6} {freq:<15.4f} {period:<15.4f} {omega:<15.2f}")
        
        print("="*60)
    

    
    def compare_with_analytical(self, beam_length: float, beam_width: float, beam_height: float, n_modes: int = 6):
        """
        Compare numerical results with analytical cantilever beam theory.
        
        Parameters:
        -----------
        beam_length, beam_height, beam_width : float
            Physical dimensions of the beam
        """
        if self.frequencies is None:
            print("No modal analysis results available.")
            return
        
        # Material properties
        E = self.assembler.E
        rho = self.assembler.rho
        A = beam_width * beam_height  # Cross-sectional area

        # Exact solution computation
        from scipy.optimize import root
        from math import cos, cosh, sqrt, pi
        falpha = lambda x: cos(x)*cosh(x)+1
        alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]
        
        analytical_freq = np.zeros(n_modes)
        for i in range(n_modes):

        # Beam eigenfrequency
            if i % 2 == 0: # exact solution should correspond to weak axis bending
                I_bend = beam_height*beam_width**3/12.
            else:          #exact solution should correspond to strong axis bending
                I_bend = beam_width*beam_height**3/12.
            analytical_freq[i] = alpha(i/2)**2*sqrt(float(E)*I_bend/(float(rho)*A*beam_length**4))/2/pi
        
        print("\n" + "="*80)
        print("COMPARISON WITH ANALYTICAL SOLUTION")
        print("="*80)
        print(f"Beam dimensions: L={beam_length:.3f}, H={beam_height:.3f}, W={beam_width:.3f}")
        print(f"Material: E={E:.2e} Pa, ρ={rho:.1f} kg/m³")
        print("-"*80)
        print(f"{'Mode':<6} {'FEM (Hz)':<12} {'Analytical (Hz)':<15} {'Error (%)':<12}")
        print("-"*80)
        
        n_compare = min(len(self.frequencies), len(analytical_freq))
        for i in range(n_compare):
            fem_freq = self.frequencies[i]
            ana_freq = analytical_freq[i]
            error = abs(fem_freq - ana_freq) / ana_freq * 100
            print(f"{i+1:<6} {fem_freq:<12.4f} {ana_freq:<15.4f} {error:<12.2f}")
        
        print("="*80)
        print("Note: Analytical solution is for pure bending (Euler-Bernoulli beam theory)")
        print("FEM includes shear deformation and 3D effects, explaining some differences.")
