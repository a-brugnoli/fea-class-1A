import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from src.utilities.restore_data import restore_data

import os
results_folder = "./be_fem_dynamics_1d/results/"
os.makedirs(results_folder, exist_ok=True)



def element_stiffness_matrix(L_element, E, I):
    """Generate element stiffness matrix using finite element method"""

    ### To be done by students
    K_element = np.array([
        [12, 6*L_element, -12, 6*L_element],
        [6*L_element, 4*L_element**2, -6*L_element, 2*L_element**2],
        [-12, -6*L_element, 12, -6*L_element],
        [6*L_element, 2*L_element**2, -6*L_element, 4*L_element**2]
    ]) * E * I / L_element**3

    return K_element
    

def element_mass_matrix(L_element, rho, A):
    """Generate element mass matrix using finite element method"""

    ### To be done by students

    M_element = np.array([
        [156, 22*L_element, 54, -13*L_element],
        [22*L_element, 4*L_element**2, 13*L_element, -3*L_element**2],
        [54, 13*L_element, 156, -22*L_element],
        [-13*L_element, -3*L_element**2, -22*L_element, 4*L_element**2]
    ]) * rho * A * L_element / 420
    
    return M_element



def assemble_global_matrix(num_elements: int, element_matrix):
    """Assemble global matrix from element matrices"""

    ### This done by students in class

    n_dofs = 2 * (num_elements + 1)
    global_matrix = np.zeros((n_dofs, n_dofs))
    
    for i in range(num_elements):
        indices = [2*i, 2*i+1, 2*i+2, 2*i+3]

        for i in range(4):
            for j in range(4):
                global_matrix[indices[i], indices[j]] += element_matrix[i, j]
            
    return global_matrix


def compute_natural_frequencies(K, M, num_modes=4):
    """Compute natural frequencies from stiffness and mass matrices"""

    ### To be done by students in class

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(K, M)
    # Natural frequencies in Hz
    eigvals[np.where(eigvals < 0)] = 0  # Avoid negative eigenvalues due to numerical errors

    frequencies = np.sqrt(eigvals) / (2 * np.pi)
    
    return frequencies[:num_modes], eigvecs[:, :num_modes]


def apply_boundary_conditions(K, M, bc_dofs):
    """Apply boundary conditions"""
    all_dofs = np.arange(M.shape[0])
    free_dofs = np.setdiff1d(all_dofs, bc_dofs)

    K_red = K[free_dofs, :][:, free_dofs]
    M_red = M[free_dofs, :][:, free_dofs]

    return K_red, M_red


n_elements = 10

L = 18.3
width = 3
height = 0.06 
A = width * height  # 0.18
I = width * height**3 / 12  # 5.4e-5

coordinates = np.linspace(0, L, n_elements + 1)

L_element = L / n_elements
E = 168e9
rho = 2330


M_global = assemble_global_matrix(n_elements, \
                                  element_mass_matrix(L_element, rho, A))
K_global = assemble_global_matrix(n_elements, \
                                  element_stiffness_matrix(L_element, E, I))

frequencies, eigenvectors_free = compute_natural_frequencies(K_global, M_global)

print(f"Natural Frequencies (Free-Free Beam) for {n_elements} elements:")
for ii, freq in enumerate(frequencies):
    print(f"Mode {ii+1}: {freq:.4f} Hz")

dofs_bcs = [0, 1]
    
# # Apply boundary conditions
K_red, M_red = apply_boundary_conditions(K_global, M_global, dofs_bcs)

frequencies_red, modes_red = compute_natural_frequencies(K_red,  M_red)
n_modes = 4

eigenvectors_cantilever = restore_data(modes_red, dofs_bcs)
print(f"Natural Frequencies (Cantilever Beam) for {n_elements} elements:")

plt.figure()
for ii, freq in enumerate(frequencies_red):
    print(f"Mode {ii+1}: {freq:.4f} Hz")

    plt.plot(coordinates, eigenvectors_cantilever[::2, ii], label=f"$\omega_{ii+1}={freq:.4f}$ [rad/s]")
    plt.title(f"Mode {ii+1} Shape")
    plt.xlabel("Position along beam (m)")
    plt.grid()
plt.legend()
plt.savefig(results_folder + 'mode_shapes_cantilever.pdf')
plt.show()


# Vecteur effort 
force_ext = 25
### To be done by students
force_vector=np.zeros((M_red.shape[0],1))
force_vector[-2,0]=force_ext

# Amortissement 
alpha = 0.000317
beta = 0
C_red=alpha*K_red + beta*M_red

n_samples_freq = 500
omega_vec = 2*np.pi*np.linspace(0, 3*frequencies_red[-1], n_samples_freq)
amplitude_displacement_freq =np.zeros((len(omega_vec), 1))
# Reponse en frequence
for ii in range(n_samples_freq):
    w = omega_vec[ii]

    # To be done by students
    impedence_matrix=-M_red*w**2+1j*w*C_red+K_red
    # disp_om=np.dot(np.linalg.inv(impedence_matrix), force_vector)
    disp_om=np.linalg.solve(impedence_matrix, force_vector)

    amplitude_displacement_freq[ii]=20*np.log10(abs(disp_om[-2,0]))


# Projection 

## To be done by students
n_mode_projection=2
Phi = modes_red[:,:n_mode_projection]

K_mod=np.dot(np.dot(np.transpose(Phi),K_red),Phi)
M_mod=np.dot(np.dot(np.transpose(Phi),M_red),Phi)
C_mod=np.dot(np.dot(np.transpose(Phi),C_red),Phi)


K_mod= Phi.T @ K_red @ Phi
M_mod= Phi.T @ M_red @ Phi
C_mod= Phi.T @ C_red @ Phi

print(f"C_mod: {C_mod}")
f_mod=np.transpose(Phi)@ force_vector

amplitude_disp_proj=np.zeros((n_samples_freq, 1))
# Reponse en frequence
for ii in range(n_samples_freq):

    w = omega_vec[ii]
    impedence_matrix_proj=-M_mod*w**2+1j*w*C_mod+K_mod

    eta_mod = np.linalg.inv(impedence_matrix_proj) @ f_mod
    displacement_modal=np.dot(Phi, eta_mod)

    amplitude_disp_proj[ii]=20*np.log10(abs(displacement_modal[-2,0]))
    
##############################
plt.plot(omega_vec/2/np.pi,amplitude_displacement_freq, 'b', label='Full model')
plt.plot(omega_vec/2/np.pi,amplitude_disp_proj, 'r--', label='Reduced model')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Displacement magnitude [dB]')
plt.legend()
plt.show()


# max_amplitude = np.max(10**(amplitude_disp_proj/20))

# print(f"Maximum amplitude at force application point: {max_amplitude:.4f} m")

# desired_max_amplitude = max_amplitude/10  # in meters

# static_ampl = abs(np.linalg.solve(K_mod, f_mod)[0, 0])

# print(f"Static amplitude at force application point: {static_ampl:.4f} m")

# quality_factor = max_amplitude / static_ampl

# print(f"Quality factor: {quality_factor:.4f}")

# desired_quality_factor = desired_max_amplitude / static_ampl

# print(f"Desired quality factor: {desired_quality_factor:.4f}")

# desired_xi = 1 / (2 * desired_quality_factor)
# print(f"Desired damping ratio: {desired_xi:.4f}")

# C_mod_desired = 2 * desired_xi * np.sqrt(K_mod * M_mod)

# C_mod_desired = 2 * desired_xi * (2*np.pi*frequencies_red[0]) * M_mod

# print(f"Desired damping coefficient: {C_mod_desired[0,0]:.6f} Ns/m")

# desired_amplitude_disp_proj=np.zeros((n_samples_freq, 1))
# for ii in range(n_samples_freq):

#     w = omega_vec[ii]
#     desired_dimpedence_matrix_proj=-M_mod*w**2+1j*w*C_mod_desired+K_mod

#     desired_eta_mod = np.linalg.inv(desired_dimpedence_matrix_proj) @ f_mod
#     desired_displacement_modal=np.dot(Phi, desired_eta_mod)

#     desired_amplitude_disp_proj[ii]=20*np.log10(abs(desired_displacement_modal[-2,0]))
    

# ##############################
# # plt.plot(omega_vec/2/np.pi,amplitude_displacement_freq, 'b', label='Full model')
# plt.plot(omega_vec/2/np.pi,amplitude_disp_proj, 'r--', label='Reduced model')
# plt.plot(omega_vec/2/np.pi,desired_amplitude_disp_proj, 'g--', label='Reduced model with desired damping')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Displacement magnitude [dB]')
# plt.legend()
# plt.show()