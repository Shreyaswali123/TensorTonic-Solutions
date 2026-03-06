import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Applies a 4x4 homogeneous transformation to 3D points.
    
    Args:
        T: 4x4 transformation matrix (NumPy array)
        points: (3,) or (N, 3) array of points
        
    Returns:
        np.ndarray: Transformed spatial coordinates (3,) or (N, 3)
    """
    T = np.asarray(T)
    points = np.asarray(points)
    
    # Check if we have a single point or a batch
    is_single = points.ndim == 1
    if is_single:
        points = points[np.newaxis, :] # Convert (3,) to (1, 3)
        
    N = points.shape[0]
    
    # 1. Convert to homogeneous coordinates: append a column of ones
    # points_h shape becomes (N, 4)
    ones = np.ones((N, 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    
    # 2. Apply transform: T * ph (but transposed for batch multiplication)
    # Since points_h is (N, 4) and T is (4, 4), we do points_h @ T.T
    transformed_h = points_h @ T.T
    
    # 3. Extract spatial part (first 3 coordinates)
    result = transformed_h[:, :3]
    
    # Return in the original format (single point vs batch)
    return result[0] if is_single else result