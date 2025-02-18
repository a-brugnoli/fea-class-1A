import numpy as np

def restore_data(reduced_data, removed_indices):
    """
    Restores a vector or matrix by inserting zeros at specified indices.
    
    Parameters:
        original_size (int): The original size (number of elements for vectors, rows for matrices).
        reduced_data (ndarray): The remaining data (1D vector or 2D matrix).
        removed_indices (set or list): Indices where elements/rows were removed.
    
    Returns:
        ndarray: The restored vector or matrix with zeros in removed positions.
    """
    # Determine if it's a vector (1D) or matrix (2D)
    
    if reduced_data.ndim == 1:
        original_size = len(reduced_data) + len(removed_indices)
        restored = np.zeros(original_size, dtype=reduced_data.dtype)
    else:
        original_size = reduced_data.shape[0] + len(removed_indices)
        restored = np.zeros((original_size, reduced_data.shape[1]), dtype=reduced_data.dtype)

    mask = np.ones(original_size, dtype=bool)  # Boolean mask for valid positions
    mask[list(removed_indices)] = False  # Mark removed positions as False
    restored[mask] = reduced_data  # Place back valid data
    return restored

# Example Usage:
if __name__=="main":
    # Restoring a Vector
    reduced_vector = np.array([1, 2, 3, 4])  
    removed_indices_vec = [1, 4]  

    restored_vector = restore_data(reduced_vector, removed_indices_vec)
    print("Restored Vector:\n", restored_vector)

    # Restoring a Matrix
    reduced_matrix = np.array([[1, 2], [3, 4], [5, 6]])  
    removed_indices_mat = {1, 3}  

    restored_matrix = restore_data(reduced_matrix, removed_indices_mat)
    print("Restored Matrix:\n", restored_matrix)
