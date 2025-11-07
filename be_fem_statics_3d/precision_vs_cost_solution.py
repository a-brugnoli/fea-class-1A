from src.meshing.structured_mesh import StructuredHexMesh
from solvers.static_analysis_3d import CantileverBeamStaticSolver
import numpy as np
import matplotlib.pyplot as plt
import time # To measure computational time 
import os

results_folder = "./be_fem_statics_3d/results/"
os.makedirs(results_folder, exist_ok=True)

# Data

# Cantilever beam dimensions
length = 1000   # Length (mm)
width = 10  # Width (mm)
height = 10  # Height (mm) 

# Material properties (Aluminum)
rho = 0  # Density (tonne/mm^3) Only for dynamic problems
E = 69e3  # Young's modulus (MPa) 
nu = 0.3  # Poisson's ratio

# Material properties dictionary
material_props = {
    'E': E,      # Young's modulus (MPa)
    'nu': nu,    # Poisson's ratio
    'rho': rho,  # Density (tonne/mm^3)
}

# Magnitude of tip force
magnitude_tip_force = 1  # Unit force at tip (N)


# In this part we focus on the convergence of the 3D static deflection solution
# with respect to mesh refinement and computational cost.
# As baseline solution, we use the analytical solution from beam theory for a cantilever beam.
I_beam = (width * height**3) / 12  # Moment of inertia for rectangular cross-section
tip_deflection_beam = magnitude_tip_force * length**3 / (3 * E * I_beam)

n_elements_y = 4  # Number of elements along width
n_elements_z = 4  # Number of elements along height
n_elements_along_length = [10, 50, 100, 200, 300, 400, 500, 1000]  # Different mesh refinements

tip_deflection_3d_vec = np.zeros(len(n_elements_along_length))
error_tip_deflection_vec = np.zeros(len(n_elements_along_length))
time_elapsed_vec = np.zeros(len(n_elements_along_length))


for i, n_elements_x in enumerate(n_elements_along_length):
    # Create mesh
    start = time.perf_counter()
    mesh = StructuredHexMesh(length, width, height, n_elements_x, n_elements_y, n_elements_z)
    print(f"  Mesh: {n_elements_x}×{n_elements_y}×{n_elements_z} elements, {mesh.n_nodes} nodes")
    # Solve
    solver = CantileverBeamStaticSolver(mesh, material_props)
    results = solver.solve(magnitude_tip_force)

    tip_deflection_3d_vec[i] = results['tip_deflection']
    error_tip_deflection_vec[i] = abs(tip_deflection_3d_vec[i] - tip_deflection_beam) / abs(tip_deflection_beam)
    end = time.perf_counter()
    time_elapsed_vec[i] = end - start

    if error_tip_deflection_vec[i] < 0.01:
        print(f"Desired accuracy of 1% achieved with {n_elements_x} elements along length.")
        

plt.figure()
plt.plot(n_elements_along_length, tip_deflection_3d_vec, 'o-', label='3D FEA')
plt.axhline(y=tip_deflection_beam, color='r', linestyle='--', label='Beam Theory')
plt.xlabel('Number of elements along length')
plt.ylabel('Tip Deflection [mm]')
plt.title('Tip Deflection vs Mesh Refinement')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_folder, "tip_deflection_vs_mesh_refinement.pdf"), format='pdf')

# Create figure and primary axis
fig, ax1 = plt.subplots()
# Plot error on left y-axis
color = 'tab:blue'
ax1.set_xlabel('Mesh size (N)')
ax1.set_ylabel('Error', color=color)
ax1.loglog(n_elements_along_length, error_tip_deflection_vec, color=color, marker='o', label='Error')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Computational Time [s]', color=color)
ax2.loglog(n_elements_along_length, time_elapsed_vec, color=color, marker='s', linestyle='--', label='Time')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Error vs Computational Time')
plt.savefig(os.path.join(results_folder, "error_vs_computational_time.pdf"), format='pdf')

plt.show()