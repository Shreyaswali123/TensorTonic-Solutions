import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generates anchor boxes for an object detection feature grid.
    
    Args:
        feature_size: Number of cells along one dimension of the grid (square).
        image_size: Dimension of the original square image.
        scales: List of anchor scales.
        aspect_ratios: List of aspect ratios (width/height).
        
    Returns:
        list[list[float]]: List of anchor boxes in [x1, y1, x2, y2] format.
    """
    anchors = []
    # Calculate how many image pixels each grid cell covers
    stride = image_size / feature_size
    
    # Iterate over grid cells in row-major order
    for i in range(feature_size): # row (y)
        for j in range(feature_size): # column (x)
            # Compute center of the current cell in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # For each cell, generate all combinations of scales and ratios
            for s in scales:
                for r in aspect_ratios:
                    # Calculate width and height based on scale and aspect ratio
                    # w/h = r  => w = h*r
                    # w*h = s^2 => (h*r)*h = s^2 => h^2 = s^2/r
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    
                    # Compute corner coordinates
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([float(x1), float(y1), float(x2), float(y2)])
                    
    return anchors