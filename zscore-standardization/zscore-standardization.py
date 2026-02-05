import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)
    
    # Empty input
    if X.size == 0:
        return X.copy()
    
    # Calculate mean and std along specified axis
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, ddof=0, keepdims=True)
    
    # Avoid division by zero
    std = np.maximum(std, eps)
    
    return (X - mean) / std