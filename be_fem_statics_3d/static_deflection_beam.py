from src.meshing.structured_mesh import StructuredHexMesh
from src.solvers.static_deflection_3d import CantileverBeamDeflection
from src.post_processing.plot_3d import PostProcessorHexMesh
import matplotlib.pyplot as plt
import os

results_folder = "./be_fem_statics_3d/results/"
os.makedirs(results_folder, exist_ok=True)

# Cantilever beam dimensions
length = 1000   # Length (mm)
width = 10  # Width (mm)
height = 10   # Height (mm) 

rho = 0  # Density (kg/mm^3)
E = 69e3  # Young's modulus (MPa)
nu = 0.3

I_beam = (width * height**3) / 12  # Moment of inertia for rectangular cross-section

# Mesh parameters
nx = 300 # Number of elements along length
ny = 2
nz = 6 

print("Creating cantilever beam mesh...")
mesh = StructuredHexMesh(length, width, height, nx, ny, nz)  # Note: y=width, z=height
print(f"  Mesh: {nx}×{ny}×{nz} elements, {mesh.n_nodes} nodes")

# Material properties
material_props = {
    'E': E,      # Young's modulus (MPa)
    'nu': nu,       # Poisson's ratio
    'rho': rho,  # Density (kg/mm^3)
}

magnitude_tip_force = 1  # Unit force at tip (N)
# Solve
solver = CantileverBeamDeflection(mesh, material_props)
results = solver.solve(magnitude_tip_force)

tip_deflection_3d = results['tip_deflection']
displacements = results['displacement_vector']
strain_tensor = results['strain_tensor']
stress_tensor = results['stress_tensor']
von_mises_stresses = results['von_mises_stresses']

tip_deflection_beam = magnitude_tip_force * length**3 / (3 * E * I_beam)

print(f"Tip deflection 3D: {tip_deflection_3d:.1f} [mm]")
print(f"Expected tip deflection (beam theory): {tip_deflection_beam:1f} [mm]")

plotter = PostProcessorHexMesh(mesh)
plotter.set_displacements(displacements, scale_factor=5)
plotter.plot_solid(field_data=von_mises_stresses, field_name="Von Mises [MPa]", cmap="viridis")

sigma_xx = stress_tensor[:,0]
tau_xz = stress_tensor[:,5]

# Deflection along the x-axis
x_coords, u_z_along_beam = plotter.extract_line_profile(
        axis='x',
        coord1=width/2, 
        coord2=height/2, 
        field_data=displacements[:,2]  # u_z displacement
    )

analytical_u_z_along_beam = - (magnitude_tip_force * x_coords**2) / (6 * E * I_beam) * (3*length - x_coords)

plt.figure(figsize=(8,6))
plt.plot(x_coords, u_z_along_beam, '+')
plt.plot(x_coords, analytical_u_z_along_beam, '-')
plt.xlabel("Position along beam x [mm]")
plt.ylabel("Deflection u_z [mm]")
plt.title("Deflection along beam length at centerline (y=width/2, z=height/2)")
plt.legend(["FEM 3D", "Beam Theory"], loc='best')
plt.grid()  
plt.savefig(os.path.join(results_folder, "deflection_along_beam.pdf"), format='pdf')


# # Results are extracted at specific x locations along the beam
# x_point_vec = [1, 500, 999]

# for x_point in x_point_vec:
#     y_coords, u_y_along_y = plotter.extract_line_profile(
#         axis='y',
#         coord1=x_point, 
#         coord2=height, 
#         field_data=displacements[:,1]  # u_y displacement
#     )
#     centered_y = y_coords - width/2

#     analytical_u_y_along_y = -nu*magnitude_tip_force*(length-x_point)*(height/2)/(E * I_beam)*centered_y

#     plt.figure(figsize=(8,6))
#     plt.plot(centered_y, u_y_along_y, '+')
#     plt.plot(centered_y, analytical_u_y_along_y, '-')
#     plt.xlabel("Width y (centered) [mm]")
#     plt.ylabel("Deflection u_y [mm]")
#     plt.title(f"Deflection u_y along width at x={x_point}, z={height}")
#     plt.legend(["FEM 3D", "Beam Theory"], loc='best')
#     plt.grid()
#     plt.savefig(os.path.join(results_folder, f"deflection_u_y_along_y_at_x_{x_point}.pdf"), format='pdf')   

#     # Extract stress profile along z-axis 
#     z_coords, sigma_xx_along_z = plotter.extract_line_profile(
#         axis='z',
#         coord1=x_point, 
#         coord2=width/2, 
#         field_data=sigma_xx
#     )
#     centered_z = z_coords - height/2

#     analytical_sigma_xx_along_z = magnitude_tip_force*(length- x_point)*centered_z/I_beam

#     plt.figure(figsize=(8,6))
#     plt.plot(sigma_xx_along_z, centered_z, '+')
#     plt.plot(analytical_sigma_xx_along_z, centered_z, '-')
#     plt.xlabel("Stress sigma_xx [MPa]")
#     plt.ylabel("Height z [mm]")
#     plt.title(f"Stress sigma_xx along z at x={x_point}, y={height/2}")
#     plt.legend(["FEM 3D", "Beam Theory"], loc='best')
#     plt.grid()  
#     plt.savefig(os.path.join(results_folder, f"stress_sigma_xx_along_z_at_x_{x_point}.pdf"), format='pdf')

#     # Extract stress profile along z-axis at x=length, y=width/2
#     z_coords, tau_xz_along_z = plotter.extract_line_profile(
#         axis='z',
#         coord1=x_point, 
#         coord2=width/2, 
#         field_data=tau_xz
#     )

#     analytical_tau_xz_along_z = - 3/2 * magnitude_tip_force / (width * height) * (1 - (2*centered_z/height)**2)

#     plt.figure(figsize=(8,6))
#     plt.plot(tau_xz_along_z, centered_z, '+')
#     plt.plot(analytical_tau_xz_along_z, centered_z, '-')
#     plt.xlabel("Stress tau_xz [MPa]")
#     plt.ylabel("Height z [mm]")
#     plt.title("Stress tau_xz along z at x=L, y=H/2")
#     plt.legend(["FEM 3D", "Beam Theory"], loc='best')
#     plt.grid()  
#     plt.savefig(os.path.join(results_folder, f"stress_tau_xz_along_z_at_x_{x_point}.pdf"), format='pdf')


plt.show()



