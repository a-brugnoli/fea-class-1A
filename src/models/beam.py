import numpy as np


class Beam:
    def __init__(self, length, num_elements, material_props: dict):
        """
        Initialize beam vibration analysis parameters
        
        Parameters:
        - length: Total beam length
        - num_elements: Number of finite elements
        - E: Young's modulus
        - I: Moment of inertia
        - rho: Density
        - A: Cross-sectional area
        """
        self.length = length
        self.num_elements = num_elements
        self.E = material_props['E']
        self.I = material_props['I']
        self.rho = material_props['rho']
        self.A = material_props['A']
        
        # Element length
        self.el_size = length / num_elements


    def generate_stiffness_matrix(self):
        """Generate global stiffness matrix using finite element method"""
        k_local = np.array([
            [12, 6*self.el_size, -12, 6*self.el_size],
            [6*self.el_size, 4*self.el_size**2, -6*self.el_size, 2*self.el_size**2],
            [-12, -6*self.el_size, 12, -6*self.el_size],
            [6*self.el_size, 2*self.el_size**2, -6*self.el_size, 4*self.el_size**2]
        ]) * self.E * self.I / self.el_size**3
        
        # Assemble global stiffness matrix
        K = np.zeros((2*(self.num_elements+1), 2*(self.num_elements+1)))
        for i in range(self.num_elements):
            i_el = [2*i, 2*i+1, 2*i+2, 2*i+3]
            K[np.ix_(i_el, i_el)] += k_local
        
        return K
    

    def generate_mass_matrix(self):
        """Generate global mass matrix using finite element method"""
        m_local = np.array([
            [156, 22*self.el_size, 54, -13*self.el_size],
            [22*self.el_size, 4*self.el_size**2, 13*self.el_size, -3*self.el_size**2],
            [54, 13*self.el_size, 156, -22*self.el_size],
            [-13*self.el_size, -3*self.el_size**2, -22*self.el_size, 4*self.el_size**2]
        ]) * self.rho * self.A * self.el_size / 420
        
        # Assemble global mass matrix
        M = np.zeros((2*(self.num_elements+1), 2*(self.num_elements+1)))
        for i in range(self.num_elements):
            i_el = [2*i, 2*i+1, 2*i+2, 2*i+3]
            M[np.ix_(i_el, i_el)] += m_local
        
        return M
    
    
    def apply_boundary_conditions_matrix(self, A, bc_dofs):
        """Apply boundary conditions"""
        # Fix first and last degrees of freedom
        mask_rows = np.ones(A.shape[0], dtype=bool)
        mask_rows[bc_dofs] = False

        A_red = A[mask_rows, :][:, mask_rows]

        return A_red
    

    def apply_boundary_conditions_vector(self, f, bc_dofs):
        """Apply boundary conditions"""
        # Fix first and last degrees of freedom
        mask_rows = np.ones(f.shape[0], dtype=bool)
        mask_rows[bc_dofs] = False

        f_red = f[mask_rows]

        return f_red
  
    

    def constant_distributed_load(self, p):
        """Generate global stiffness matrix using finite element method"""
        f_local = p * np.array([
            [self.el_size/2],
            [self.el_size**2/12],
            [self.el_size/2],
            [-self.el_size**2/12]
        ]) 
        
        # Assemble global stiffness matrix
        f = np.zeros((2*(self.num_elements+1), 1))
        for i in range(self.num_elements):
            i_el = [2*i, 2*i+1, 2*i+2, 2*i+3]
            f[i_el] += f_local
        
        return f
    

    def displacement_at_points(self, x_plot, q_sol):
        """
        Compute vertical displacements at specified x-coordinates using Hermite interpolation

        Parameters:
        - x_plot: Array of x-coordinates where displacements are evaluated
        - q_sol: Solution vector of size 2*(num_elements+1), alternating [v0, θ0, v1, θ1, ...]
        
        Returns:
        - v_plot: Array of vertical displacements at each x in x_plot
        """

        assert len(q_sol) == 2*(self.num_elements+1), "Solution vector size mismatch"

        # Node positions along the beam
        node_positions = np.linspace(0, self.length, self.num_elements + 1)

        # Extract transverse displacements (even indices) and rotations (odd indices)
        displacements = q_sol[0::2].flatten()
        rotations     = q_sol[1::2].flatten()

        v_plot = np.zeros(len(x_plot))

        for j, x in enumerate(x_plot):
            # Find which element x belongs to (clamp edge case x == length to last element)
            i = min(np.searchsorted(node_positions, x, side='right') - 1,
                    self.num_elements - 1)

            # Local coordinate xi in [0, 1]
            xi = (x - node_positions[i]) / self.el_size

            # Nodal values for this element
            v0, t0 = displacements[i],   rotations[i]
            v1, t1 = displacements[i+1], rotations[i+1]
            L = self.el_size

            # Cubic Hermite shape functions
            N1 =  1 - 3*xi**2 + 2*xi**3
            N2 =  L * xi * (1 - xi)**2
            N3 =  3*xi**2 - 2*xi**3
            N4 =  L * xi**2 * (xi - 1)

            v_plot[j] = N1*v0 + N2*t0 + N3*v1 + N4*t1

        return v_plot


   