from src.meshing.structured_mesh import StructuredHexMesh
from src.solvers.static_analysis_3d import CantileverBeamStaticSolver
import numpy as np
import matplotlib.pyplot as plt
import time # To measure computational time 
import os
from data import length, width, height, \
                material_props, magnitude_tip_force, \
                tip_deflection_beam, results_folder

n_elements_along_length = [25, 50, 100, 200, 400, 800]  # Different mesh refinements
n_elements_y, n_elements_z = 4, 4  # Number of elements along width and height

tip_deflection_3d_vec = np.zeros(len(n_elements_along_length))
error_tip_deflection_vec = np.zeros(len(n_elements_along_length))
time_elapsed_vec = np.zeros(len(n_elements_along_length))

tolerance = 0.05  # 5% tolerance for tip deflection

for i, n_elements_x in enumerate(n_elements_along_length):
    # Create mesh
    start = time.perf_counter()
    mesh = StructuredHexMesh(length, width, height, n_elements_x, n_elements_y, n_elements_z)
    # Solve
    solver = CantileverBeamStaticSolver(mesh, material_props)
    results = solver.solve(magnitude_tip_force)

    tip_deflection_3d_vec[i] = results['tip_deflection']
    error_tip_deflection_vec[i] = abs(tip_deflection_3d_vec[i] - tip_deflection_beam) / abs(tip_deflection_beam)
    end = time.perf_counter()
    time_elapsed_vec[i] = end - start

    if error_tip_deflection_vec[i] < tolerance:
        print(f"Desired accuracy of {tolerance*100}% achieved with {n_elements_x} elements along length.")
        
time_elapsed_vec = time_elapsed_vec / np.max(time_elapsed_vec)  # Normalize time for plotting

plt.figure()
plt.plot(n_elements_along_length, tip_deflection_3d_vec, 'o-', label='3D FEA')
plt.axhline(y=tip_deflection_beam, color='r', linestyle='--', label='Beam Theory')
plt.axhline(y=(1-tolerance)*tip_deflection_beam, color='b', linestyle='--', label='Beam Theory - 5%')
plt.xlabel('Number of elements along x')
plt.ylabel('Tip Deflection [mm]')
plt.title('Tip Deflection vs Mesh Refinement')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_folder, "tip_deflection_vs_mesh_refinement.pdf"), format='pdf')

plt.figure()
plt.xlabel('Number of elements along x')
plt.plot(n_elements_along_length, error_tip_deflection_vec, label='Error')
plt.plot(n_elements_along_length, time_elapsed_vec, label='Time')
plt.plot(n_elements_along_length, error_tip_deflection_vec+time_elapsed_vec, label='Error + Time')
plt.title('Error/Computational Time vs Mesh Refinement')
# plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_folder, "error_plus_time_vs_mesh_refinement.pdf"), format='pdf')

# Create figure and primary axis
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Number of elements along x')
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