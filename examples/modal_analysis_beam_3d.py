from src.meshing.structured_mesh import StructuredHexMesh
from src.models.hexahedral_solid import Q8FiniteElementAssembler  
from src.post_processing.plot_3d import PostProcessorHexMesh
from src.solvers.modal_analysis_3d import CantileverModalAnalysis
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Cantilever beam dimensions
    length = 20   # Length (m)
    width = 0.5  # Width (m)
    height = 1   # Height (m) 
    
    # Mesh parameters
    nx = 50
    ny = int(width/length*nx)+1
    nz = int(height/length*nx)+1
    
    print("Creating cantilever beam mesh...")
    mesh = StructuredHexMesh(length, width, height, nx, ny, nz)  # Note: y=width, z=height
    print(f"  Mesh: {nx}×{ny}×{nz} elements, {mesh.n_nodes} nodes")
    
    # Material properties (Steel)
    material_props = {
        'E': 1e5,      # Young's modulus (Pa)
        'nu': 0,       # Poisson's ratio
        'rho': 1e-3    # Density (kg/m³)
    }
    
    print("Assembling finite element matrices...")
    assembler = Q8FiniteElementAssembler(mesh, material_props)
    K, M = assembler.assemble_matrices()
    
    # Create modal analysis object
    modal_analysis = CantileverModalAnalysis(mesh, assembler)
    
    # Apply cantilever boundary conditions (fix x=0 face)
    modal_analysis.apply_boundary_conditions('x_min')
    
    # Solve for eigenfrequencies and mode shapes
    print("\nSolving modal analysis...")
    frequencies, mode_shapes = modal_analysis.solve_eigenvalue_problem(n_modes=6, sigma=1e-6)
    
    plotter = PostProcessorHexMesh(mesh)

    for ii in range(4):
        fig, ax = plotter.plot_mode_shape_solid(mode_shapes[:, ii], mode_number=ii+1, scale_factor=.1)
    
    # Print results
    modal_analysis.print_modal_results(10)
    # Compare with analytical solution
    modal_analysis.compare_with_analytical(length, beam_width=width, beam_height=height, n_modes=10)

    print("\nModal analysis complete!")

    plt.show()
