import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from time_integration import newmark
from post_processing import plot_vertical_displacement


class BeamVibration:
    def __init__(self, length, num_elements, E, I, rho, A):
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
        self.E = E
        self.I = I
        self.rho = rho
        self.A = A
        
        # Element length
        self.dx = length / num_elements
        
    def generate_stiffness_matrix(self):
        """Generate global stiffness matrix using finite element method"""
        k_local = np.array([
            [12, 6*self.dx, -12, 6*self.dx],
            [6*self.dx, 4*self.dx**2, -6*self.dx, 2*self.dx**2],
            [-12, -6*self.dx, 12, -6*self.dx],
            [6*self.dx, 2*self.dx**2, -6*self.dx, 4*self.dx**2]
        ]) * (self.E * self.I / (self.dx**3))
        
        # Assemble global stiffness matrix
        K = lil_matrix((2*(self.num_elements+1), 2*(self.num_elements+1)))
        for i in range(self.num_elements):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            K[np.ix_(idx, idx)] += k_local
        
        
        return K.tocsc()
    

    def generate_mass_matrix(self):
        """Generate global mass matrix using finite element method"""
        m_local = np.array([
            [156, 22*self.dx, 54, -13*self.dx],
            [22*self.dx, 4*self.dx**2, 13*self.dx, -3*self.dx**2],
            [54, 13*self.dx, 156, -22*self.dx],
            [-13*self.dx, -3*self.dx**2, -22*self.dx, 4*self.dx**2]
        ]) * ((self.rho * self.A * self.dx) / 420)
        
        # Assemble global mass matrix
        M = lil_matrix((2*(self.num_elements+1), 2*(self.num_elements+1)))
        for i in range(self.num_elements):
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            M[np.ix_(idx, idx)] += m_local
        
        return M.tocsc()
    
    
    def apply_boundary_conditions(self, K, M, q0, v0, dofs):
        """Apply simply supported beam boundary conditions"""
        # Fix first and last degrees of freedom
        mask_rows = np.ones(M.shape[0], dtype=bool)
        mask_rows[dofs] = False

        K_red = K[mask_rows, :][:, mask_rows]
        M_red = M[mask_rows, :][:, mask_rows]

        q0_red = q0[mask_rows]
        v0_red = v0[mask_rows]
        
        return K_red, M_red, q0_red, v0_red
    
    
    
# Test
if __name__ == "__main__":
    # Beam parameters
    length = 1.0  # beam length
    E = 2.0e11  # Young's modulus (Pa)
    I = 1.0e-6  # Moment of inertia (m^4)
    rho = 7800  # Density (kg/m^3)
    A = 1.0e-4  # Cross-sectional area (m^2)
    
    num_elements = 10
    coordinates = np.linspace(0, length, num_elements + 1)
    num_dofs = 2*len(coordinates)

    # Create beam analysis object
    beam = BeamVibration(length, num_elements, E, I, rho, A)
    
    # Generate matrices
    K = beam.generate_stiffness_matrix()
    M = beam.generate_mass_matrix()
    
    exp_initial_displacement = lambda x: np.sin(np.pi*x/length)
    exp_initial_rotation = lambda x: np.pi/length*np.cos(np.pi*x/length)

    q0 = np.zeros(num_dofs)
    v0 = np.zeros(num_dofs)

    q0[0::2] = exp_initial_displacement(coordinates)
    q0[1::2] = exp_initial_rotation(coordinates)

    # Apply boundary conditions
    dofs_bcs = [0, 2*num_elements]
    K_reduced, M_reduced, q0_red, v0_red = beam.apply_boundary_conditions(K, M, q0, v0, dofs_bcs)

    # Solve dynamic response
    T_end = 10  # Total simulation time
    dt = 0.01  # Time step
    n_times = int(np.ceil(T_end/dt))
    q_array_red, v_array_red = newmark(q0_red, v0_red, M_reduced, K_reduced, dt, n_times)


    n_rows_red = q_array_red.shape[0]
    q_array = np.insert(q_array_red, [0, n_rows_red-1], 0, axis=0)

    # Post-processing
    plot_vertical_displacement(dt, coordinates, q_array)

