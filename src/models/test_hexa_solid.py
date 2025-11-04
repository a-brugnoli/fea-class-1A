from src.models.hexahedral_solid import Q8FiniteElementAssembler
from src.meshing.structured_mesh import StructuredHexMesh
import numpy as np


if __name__ == "__main__":   
    # Create a simple mesh
    mesh = StructuredHexMesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
    
    # Material properties for steel
    material_props = {
        'E': 210e9,      # Young's modulus (Pa)
        'nu': 0.3,       # Poisson's ratio
        'rho': 7850.0    # Density (kg/m³)
    }
    
    # Create assembler
    assembler = Q8FiniteElementAssembler(mesh, material_props)
    
    # Assemble matrices
    K, M = assembler.assemble_matrices()
    
    # Print information
    print("\nMatrix Assembly Results:")
    info = assembler.get_matrix_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nStiffness matrix shape: {K.shape}")
    print(f"Mass matrix shape: {M.shape}")
    print(f"Stiffness matrix condition number estimate: {np.linalg.cond(K.toarray()[:100, :100]):.2e}")
