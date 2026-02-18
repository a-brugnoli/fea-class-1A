from src.models.beam import Beam
import numpy as np
import matplotlib.pyplot as plt
from src.utilities.restore_data import restore_data


import os
results_folder = "./examples/results/cantilever_beam_gravity/"
os.makedirs(results_folder, exist_ok=True)


L = 1.0  # beam length
E = 2.0e11  # Young's modulus (Pa)
I = 1.0e-6  # Moment of inertia (m^4)
rho = 7800  # Density (kg/m^3)
A = 1.0e-4  # Cross-sectional area (m^2)
EI = E * I    
load_midpoint = -20  # Load at the tip of the beam (N)
distributed_load = -100

def cantilever_exact_displacement(p, F):
    """
    Vertical displacement of a cantilever beam under
    uniform distributed load p and force F at the midpont.

    Parameters
    ----------
    p  : float
         Uniform distributed load (force per length)
    F  : float
         Force at the midpoint 

    Returns
    -------
    w : float or array_like
        Vertical displacement at x
    """

    w_p = lambda x: (p / (24 * EI)) * x**2 * (x**2 - 4*L*x + 6*L**2)
    w_F_mid = lambda x: (F / (6*E*I)) * (3*L/2*x**2 - x**3 + np.where(x >= L/2, (x - L/2)**3, 0.0))
    return lambda x: w_p(x) + w_F_mid(x)


x_plot = np.linspace(0, L, 30)
w_exact = cantilever_exact_displacement(distributed_load, load_midpoint)(x_plot)

properties = {
    'E': E,
    'I': I,
    'rho': rho,
    'A': A
}

dofs_bcs = [0, 1]

n_elements_array = [2, 4, 8, 16] 


fig, ax = plt.subplots()

# Exact solution: solid black, slightly thicker
ax.plot(x_plot, w_exact,
        color='grey', linewidth=2, linestyle=':',
        label='Exact', zorder=5)

error_norm = np.zeros(len(n_elements_array))

for ii, num_elements in enumerate(n_elements_array):
    beam = Beam(L, num_elements, properties)
    K = beam.generate_stiffness_matrix()
    f = beam.constant_distributed_load(distributed_load)

    node_midpoint = num_elements // 2
    dof_midpoint = 2 * node_midpoint  # vertical displacement DOF at midpoint
    f[dof_midpoint] += load_midpoint

    K_reduced = beam.apply_boundary_conditions_matrix(K, dofs_bcs)
    f_red     = beam.apply_boundary_conditions_vector(f, dofs_bcs)
    q_reduced = np.linalg.solve(K_reduced, f_red)
    q_full    = restore_data(q_reduced, dofs_bcs)

    v_plot = beam.displacement_at_points(x_plot, q_full)

    ax.plot(x_plot, v_plot,
            linewidth=1.8,
            label=f'{num_elements} els')

    # Compute error at tip (x=L)
    error_norm[ii] = np.linalg.norm(v_plot - w_exact)

ax.set_xlabel('$x \; [\mathrm{m}]$')
ax.set_ylabel('$v \; [\mathrm{m}]$')
ax.set_title('Cantilever Beam Deflection')
ax.legend(loc='best')
ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "cantilever_beam_deflection.pdf"))

# Plot error convergence
fig, ax = plt.subplots()
ax.plot(n_elements_array, error_norm, marker='o', linestyle='-')
ax.set_yscale('log')

ax.set_xlabel('Number of Elements')
ax.set_ylabel('$||v - v_{\mathrm{exact}}||$')
ax.set_title('Error Convergence')
ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "cantilever_beam_error_convergence.pdf"))

plt.show()