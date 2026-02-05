import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)

    if X.size == 0:
        return X.copy()
    
    mini= np.min(X, axis=axis, keepdims=True)
    maxi=np.max(X, axis=axis, keepdims=True)
    div = np.maximum(eps, maxi-mini)
    return (X-mini)/div