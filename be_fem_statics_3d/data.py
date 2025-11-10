
#  Cantilever beam dimensions
length = 1000   # Length (mm)
width = 10  # Width (mm)
height = 10   # Height (mm) 

# Material properties dictionary
rho = 0  # Density (tonne/mm^3)
E = 69e3  # Young's modulus (MPa)
nu = 0.3
material_props = {
    'E': E,      # Young's modulus (MPa)
    'nu': nu,    # Poisson's ratio
    'rho': rho,  # Density (tonne/mm^3)
}
# Unit force at tip (N)
magnitude_tip_force = 1  
# Moment of area for rectangular cross-section (mm^4)
I_beam = (width * height**3) / 12  
# Tip deflection from beam theory (mm)
tip_deflection_beam = magnitude_tip_force * length**3 / (3 * E * I_beam)

import os
results_folder = "./be_fem_statics_3d/results/"
os.makedirs(results_folder, exist_ok=True)
