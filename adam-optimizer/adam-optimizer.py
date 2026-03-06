import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # Ensure inputs are numpy arrays to allow float multiplication
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    
    # mt = beta1 * mt-1 + (1 - beta1) * gt
    m_new = beta1 * m + (1 - beta1) * grad
    
    # vt = beta2 * vt-1 + (1 - beta2) * gt^2
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    
    # Bias correction
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    
    # Update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new