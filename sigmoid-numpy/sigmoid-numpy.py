import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.
    Works for scalar, list, or NumPy array inputs.
    """
    x = np.array(x)  # Convert input to NumPy array
    return 1 / (1 + np.exp(-x))