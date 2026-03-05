import numpy as np

def dropout(x, p, rng=None):
    """
    Implements inverted dropout on the input array.
    
    Args:
        x: Input NumPy array.
        p: Dropout probability (probability of setting an element to zero).
        rng: Optional NumPy random generator.
        
    Returns:
        tuple: (output, dropout_pattern)
    """
    x = np.array(x)
    
    # If p is 0, no dropout occurs.
    if p == 0.0:
        pattern = np.ones_like(x)
        return x, pattern

    # 1. Generate random mask
    # We use (1 - p) because that's the probability of KEEPING an element
    keep_prob = 1 - p
    
    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
        
    # 2. Create the mask: 1 for keep, 0 for drop
    # Then scale the mask immediately to handle "inverted dropout"
    # dropout_patterni = (1 / keep_prob) if kept, else 0
    mask = (random_values < keep_prob).astype(float)
    dropout_pattern = mask / keep_prob
    
    # 3. Apply the pattern to the input
    output = x * dropout_pattern
    
    return output, dropout_pattern