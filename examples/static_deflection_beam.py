import numpy as np
from scipy.sparse.linalg import spsolve
from src.meshing.structured_mesh import StructuredHexMesh
from src.models.solid import Q8FiniteElementAssembler
from src.post_processing.plot_3d import PostProcessorHexMesh

class CantileverBeamDeflection:
    """
    Class to compute static deflection of a cantilever beam under unit tip load.
    
    Assumes:
    - Beam is fixed at one end (x=0) and free at the other end
    - Unit load is applied at the tip in the negative y-direction
    - 2D beam elements with 2 DOF per node (y-displacement and rotation)
    """
    
    def __init__(self, mesh: StructuredHexMesh, material_props: dict):
        """
        Initialize the cantilever beam deflection solver.
        
        Parameters:
        -----------
        mesh : object
            Mesh object with attributes:
            - coordinates: array of nodal coordinates [(x1,y1), (x2,y2), ...]
            - elements: array of element connectivity [[node1, node2], ...]
        material_props : dict
            Dictionary containing material properties:
            - 'E': Young's modulus (Pa)
            - 'nu': Poisson's ratio
            - 'rho': Density (kg/m^3)
        -----------
        """
        self.material_props = material_props

        self.assembler = Q8FiniteElementAssembler(mesh, material_props)        
        self.mesh = mesh
        self.n_nodes = len(mesh.coordinates)
        self.n_dofs = 3 * self.n_nodes 
        x_coords = np.array(self.mesh.coordinates)[:, 0]
        
        # Fixed end: nodes at minimum x-coordinate
        min_x = np.min(x_coords)
        self.fixed_nodes = np.where(np.isclose(x_coords, min_x, rtol=1e-10))[0]
        self.fixed_dofs = []
        for node in self.fixed_nodes:
            self.fixed_dofs.extend([3*node, 3*node+1, 3*node+2])  # u, v, w

        all_dofs = np.arange(self.n_dofs)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)
        
        # Free end: nodes at maximum x-coordinate  
        max_x = np.max(x_coords)
        self.traction_nodes = np.where(np.isclose(x_coords, max_x, rtol=1e-10))[0]


    def update_mesh(self, mesh: StructuredHexMesh):
        self.assembler = Q8FiniteElementAssembler(mesh, self.material_props)        
        self.mesh = mesh
        self.n_nodes = len(mesh.coordinates)
        self.n_dofs = 2 * self.n_nodes  # 2 DOF per node (displacement, rotation)

        x_coords = np.array(self.mesh.coordinates)[:, 0]
        
        # Fixed end: nodes at minimum x-coordinate
        min_x = np.min(x_coords)
        self.fixed_nodes = np.where(np.abs(x_coords - min_x) < 1e-10)[0]
        self.fixed_dofs = []
        for node in self.fixed_nodes:
            self.fixed_dofs.extend([3*node, 3*node+1, 3*node+2])  # u, v, w

        all_dofs = np.arange(self.n_dofs)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)
        

    def _apply_boundary_conditions(self, K, F):
        """
        Apply boundary conditions: fix displacement and rotation at fixed end.
        
        Parameters:
        -----------
        K : sparse matrix
            Global stiffness matrix
        F : array
            Global force vector
            
        Returns:
        --------
        K_reduced : sparse matrix
            Reduced stiffness matrix after applying BCs
        F_reduced : array
            Reduced force vector after applying BCs
        free_dofs : array
            Array of free degrees of freedom indices
        """
         
        # Extract reduced system
        K_reduced = K[np.ix_(self.free_dofs, self.free_dofs)]
        F_reduced = F[self.free_dofs]
        
        return K_reduced, F_reduced
    
    def _create_force_vector(self, force):
        """
        Create force vector with unit tip load.
        
        Returns:
        --------
        F : array
            Global force vector with unit load at tip
        """
        # Free end: nodes at maximum x-coordinate  
        x_coords = np.array(self.mesh.coordinates)[:, 0]

        max_x = np.max(x_coords)

        self.traction_nodes = np.where(np.isclose(x_coords, max_x,rtol=1e-10))[0]

        value_nodal_force = - force / len(self.traction_nodes)  # Distribute load evenly
        F = np.zeros(self.n_dofs)
        
        # Apply unit downward force at free end nodes
        for node in self.traction_nodes:
            dof_along_z = 3 * node + 2  # z-displacement DOF
            F[dof_along_z] = value_nodal_force  # Distribute load among tip nodes
            
        return F
    
    def solve(self, magnitude_tip_force=1.0):
        """
        Solve for static deflection under unit tip load.
        
        Returns:
        --------
        deflections : dict
            Dictionary containing:
            - 'displacements': nodal displacements
            - 'rotations': nodal rotations  
            - 'tip_deflection': maximum deflection at tip
            - 'solution_vector': full DOF solution vector
        """
        # Assemble global stiffness matrix
        print("Assembling stiffness matrix...")
        K = self.assembler.assemble_stiffness_matrix()
        
        # Create force vector
        # F = self._create_force_vector(magnitude_tip_force)
        cross_section_area = self.mesh.Ly * self.mesh.Lz  # Cross-sectional area
        traction = np.array([0, 0, -magnitude_tip_force/cross_section_area])  # Force vector at tip
        F = self.assembler.assemble_surface_traction_force(traction, traction_plane="x_max")

        # Apply boundary conditions
        print("Applying boundary conditions...")
        K_reduced, F_reduced = self._apply_boundary_conditions(K, F)
        
        # Solve reduced system
        print("Solving linear system...")
        if hasattr(K_reduced, 'toarray'):
            # Handle sparse matrix
            u_reduced = spsolve(K_reduced, F_reduced)
        else:
            # Handle dense matrix
            u_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Reconstruct full solution vector
        u_full = np.zeros(self.n_dofs)
        u_full[self.free_dofs] = u_reduced

        tip_deflection = np.max(np.abs(u_full))  # y-displacement at tip nodes
        
        results = {
            'tip_deflection': tip_deflection,
            'solution_vector': u_full.reshape(-1, 3),  
        }
        
        return results
    

# Example usage and testing
if __name__ == "__main__":
    # Import required classes
    # Note: In practice, you would import these from separate files
    
    # Cantilever beam dimensions
    length = 1   # Length (m)
    width = 0.01  # Width (m)
    height = 0.01   # Height (m) 

    rho = 2700  # Density (kg/m^3)
    E = 69e9  # Young's modulus (Pa) for steel
    nu = 0.3
    
    # Mesh parameters
    nx = 100 # Number of elements along length
    ny = 2 # int(width/length*nx)+1
    nz = 2 # int(height/length*nx)+1
    
    print("Creating cantilever beam mesh...")
    mesh = StructuredHexMesh(length, width, height, nx, ny, nz)  # Note: y=width, z=height
    print(f"  Mesh: {nx}×{ny}×{nz} elements, {mesh.n_nodes} nodes")

    # Material properties (Steel)
    material_props = {
        'E': E,      # Young's modulus (Pa)
        'nu': nu,       # Poisson's ratio
        'rho': rho,  # Density (kg/m^3)
    }

    magnitude_tip_force = 1  # Unit force at tip (N)
    # Solve
    solver = CantileverBeamDeflection(mesh, material_props)
    results = solver.solve(magnitude_tip_force)
    
    I_beam = (width * height**3) / 12  # Moment of inertia for rectangular cross-section
    tip_deflection_beam = magnitude_tip_force * length**3 / (3 * E * I_beam)

    print(f"Tip deflection 3D: {results['tip_deflection']*1e3:.1f} [mm]")
    print(f"Expected tip deflection (beam theory): {tip_deflection_beam*1e3:1f} [mm]")

    # plotter = PostProcessorHexMesh(mesh)
    # plotter.set_displacements(results['solution_vector'], scale_factor=5)
    # plotter.plot_solid()

    # import matplotlib.pyplot as plt
    # plt.show()



