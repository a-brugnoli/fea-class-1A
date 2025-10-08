import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from src.solvers.time_integration import newmark
from src.post_processing.configuration import configure_matplotlib
from src.post_processing.plot_1d import plot_1d_vertical_displacement, animate_1d_mode
configure_matplotlib()
from src.utilities.restore_data import restore_data

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
    
    
    def apply_boundary_conditions(self, K, M, bc_dofs):
        """Apply boundary conditions"""
        # Fix first and last degrees of freedom
        mask_rows = np.ones(M.shape[0], dtype=bool)
        mask_rows[bc_dofs] = False

        K_red = K[mask_rows, :][:, mask_rows]
        M_red = M[mask_rows, :][:, mask_rows]

        return K_red, M_red
    

   
if __name__ == "__main__":
    # Beam parameters
    length = 1.0  # beam length
    E = 2.0e11  # Young's modulus (Pa)
    I = 1.0e-6  # Moment of inertia (m^4)
    rho = 7800  # Density (kg/m^3)
    A = 1.0e-4  # Cross-sectional area (m^2)

    properties = {
        'E': E,
        'I': I,
        'rho': rho,
        'A': A
    }

    num_elements = 200
    coordinates = np.linspace(0, length, num_elements + 1)
    num_dofs = 2*len(coordinates)

    # Create beam analysis object
    beam = Beam(length, num_elements, properties)
    
    # Generate matrices
    K = beam.generate_stiffness_matrix()
    M = beam.generate_mass_matrix()

    # dofs_bcs = [0, 2*num_elements]
    dofs_bcs = [0, 1]
    
    # Apply boundary conditions
    K_reduced, M_reduced = beam.apply_boundary_conditions(K, M, dofs_bcs)
    omega_squared, modes_red = sla.eigh(K_reduced, b = M_reduced)
    omega_vec = np.sqrt(np.real(omega_squared))

    eigenvectors = restore_data(modes_red, dofs_bcs)

    n_modes = 4
    for ii in range(n_modes):
        plt.plot(coordinates, eigenvectors[::2, ii], label=f"$\omega_{ii+1}={omega_vec[ii]:.1f}$ [rad/s]")
        plt.legend()

    # num_mode = 1
    # mode_shape = eigenvectors[::2, num_mode]
    # omega_mode = omega_vec[num_mode]
    # animation = animate_1d_mode(coordinates, mode_shape, omega_mode)    


    # # Initial conditions corresponding to first mode
    # q0 = np.zeros(num_dofs)
    # v0 = np.zeros(num_dofs)
    
    # q0[::2] = eigenvectors[0::2, num_mode]
    # q0[1::2] = eigenvectors[1::2, num_mode]

    # q0_red = np.delete(q0, dofs_bcs)
    # v0_red = np.delete(v0, dofs_bcs)

    # # This part is to be done by the students:
    # # - declare dofs subjected to bcs
    # # - extract modes
    # # - plot them
    # # For clamped bcs and for free bcs


    # # Solve dynamic response
    # T_end = 1  # Total simulation time
    # dt = 2*np.pi/omega_vec[num_mode]/10  # Time step
    # print(f"Time step: {dt:.4f} [s]")
    # n_times = int(np.ceil(T_end/dt))
    # q_array_red, v_array_red = newmark(q0_red, v0_red, M_reduced, K_reduced, dt, n_times)

    # q_array = restore_data(q_array_red, dofs_bcs)
    # # Post-processing
    # animation = plot_1d_vertical_displacement(dt, coordinates, q_array)

    plt.show()
