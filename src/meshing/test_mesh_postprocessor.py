import os
from src.meshing.structured_mesh import StructuredHexMesh
from src.post_processing.plot_3d import PostProcessorHexMesh
import matplotlib.pyplot as plt
from src.post_processing.configuration import configure_matplotlib
configure_matplotlib()
import numpy as np

# Example usage
if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # Create a mesh for a 2×1×1 box with 4×2×2 elements
    mesh = StructuredHexMesh(Lx=2.0, Ly=1.0, Lz=1.0, nx=4, ny=2, nz=2)
    
    print(mesh)
    print("\nFirst few nodes:")
    print(mesh.coordinates[:8])
    print("\nFirst element connectivity:")
    print(mesh.elements[0])
    print("\nElement 0 center:", mesh.get_element_center(0))
    
    # Export to VTK for visualization
    mesh.export_vtk(current_folder +"/box_mesh.vtk")
    print("\nMesh exported to 'box_mesh.vtk'")

    # Create post-processor
    postproc = PostProcessorHexMesh(mesh)
    
    # Test 1: Basic solid plotting
    print("Plotting basic solid...")
    fig1, ax1 = postproc.plot_solid(figsize=(10, 8))
    plt.show()

    # Test 2: Solid with field data (example: distance from origin)
    print("Plotting solid with field data...")
    nodes_array = np.array(mesh.coordinates)
    field_data = np.linalg.norm(nodes_array, axis=1)  # Distance from origin
    
    fig2, ax2 = postproc.plot_solid(
        field_data=field_data,
        field_name="Distance from Origin",
        cmap='plasma',
        alpha=0.9
    )
    plt.show()
    
    # Test 3: Deformed solid (example displacement)
    print("Plotting deformed solid...")
    displacements = np.zeros_like(nodes_array)
    # Create a simple bending displacement pattern
    for i, node in enumerate(nodes_array):
        x, y, z = node
        displacements[i] = 2*np.array([0, 0, 0.1 * x * z])  # Bending in z-direction
    
    postproc.set_displacements(displacements)
    fig3, ax3 = postproc.plot_solid(
        field_data=np.linalg.norm(displacements, axis=1),
        field_name="Displacement Magnitude",
        cmap='coolwarm'
    )
    plt.show()