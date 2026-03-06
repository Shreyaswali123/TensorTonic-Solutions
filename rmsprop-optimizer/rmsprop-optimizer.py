import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Performs a single update step of the RMSProp optimizer.
    
    Args:
        w: Current parameters (np.ndarray)
        g: Current gradients (np.ndarray)
        s: Running squared gradient accumulator (np.ndarray)
        lr: Learning rate (eta)
        beta: Decay factor for moving average
        eps: Small constant for numerical stability
        
    Returns:
        tuple: (new_w, new_s)
    """
    # Convert inputs to numpy arrays to ensure vectorized math support
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)
    
    # Step 1: Update Running Average of squared gradients
    # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
    new_s = beta * s + (1 - beta) * (g**2)
    
    # Step 2: Parameter Update
    # w_t = w_{t-1} - (lr / (sqrt(s_t) + eps)) * g_t
    new_w = w - (lr / (np.sqrt(new_s) + eps)) * g
    
    return new_w, new_s