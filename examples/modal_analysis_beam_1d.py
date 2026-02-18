import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from src.models.beam import Beam
from src.solvers.time_integration import newmark
from src.post_processing.configuration import configure_matplotlib
from src.post_processing.plot_1d import plot_1d_vertical_displacement, animate_1d_mode
configure_matplotlib()
from src.utilities.restore_data import restore_data

import os
results_folder = "./examples/results/modal_analysis_beam_1d/"
os.makedirs(results_folder, exist_ok=True)

# Beam parameters
length = 1.0  # beam length
E = 2.0e11  # Young's modulus (Pa)
rho = 7800  # Density (kg/m^3)

I = 1.0e-6  # Moment of inertia (m^4)
A = 1.0e-4  # Cross-sectional area (m^2)

print(f"Flexural rigidity EI: {E*I:.2e} [N*m^2]")
print(f"Mass per unit length: {rho*A:.2f} [kg/m]")

properties = {
    'E': E,
    'I': I,
    'rho': rho,
    'A': A
}

num_elements = 50
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
K_reduced = beam.apply_boundary_conditions_matrix(K, dofs_bcs)
M_reduced = beam.apply_boundary_conditions_matrix(M, dofs_bcs)
omega_squared, modes_red = sla.eigh(K_reduced, b = M_reduced)
omega_vec_hz = np.sqrt(np.real(omega_squared))/(2*np.pi)

eigenvectors = restore_data(modes_red, dofs_bcs)

plt.figure(figsize=(8, 6))
n_modes = 4
for ii in range(n_modes):
    print(f"Mode {ii+1}: {omega_vec_hz[ii]:.4f} Hz")
    plt.plot(coordinates, eigenvectors[::2, ii], label=f"$\omega_{ii+1}={omega_vec_hz[ii]:.1f}$ [Hz]")
    plt.xlabel("x [m]")
    plt.grid()
plt.legend(loc='best')
plt.title(f"First {n_modes} Mode Shapes")
plt.savefig(results_folder + 'mode_shapes_cantilever.pdf')
plt.show()

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
