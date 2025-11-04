import numpy as np
def von_mises_stress(stresses):
        """
        Compute von Mises stress from 3D stress components
        
        Args:
            stresses: Stress components (n_points, 6) - [σx, σy, σz, τxy, τyz, τxz]
            
        Returns:
            von_mises: von Mises stress (n_points,)
        """
        sigma_x = stresses[:, 0]
        sigma_y = stresses[:, 1]
        sigma_z = stresses[:, 2]
        tau_xy = stresses[:, 3]
        tau_yz = stresses[:, 4]
        tau_xz = stresses[:, 5]
        
        # von Mises stress formula for 3D
        von_mises = np.sqrt(0.5 * ((sigma_x - sigma_y)**2 + 
                                (sigma_y - sigma_z)**2 + 
                                (sigma_z - sigma_x)**2 + 
                                6 * (tau_xy**2 + tau_yz**2 + tau_xz**2)))
        
        return von_mises