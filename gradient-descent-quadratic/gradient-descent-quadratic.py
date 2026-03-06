def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Finds the minimum of f(x) = ax^2 + bx + c using gradient descent.
    
    Args:
        a, b, c: Coefficients of the quadratic equation.
        x0: Initial starting point (float).
        lr: Learning rate.
        steps: Number of iterations.
        
    Returns:
        float: The value of x that minimizes the function.
    """
    x = float(x0)
    
    for _ in range(steps):
        # 1. Compute the derivative (gradient) at current x
        gradient = 2 * a * x + b
        
        # 2. Update x by moving in the opposite direction of the gradient
        x = x - lr * gradient
        
    return x