import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    #reshape
    if X_train.ndim ==1:
        X_train=X_train.reshape(-1,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    n_train=X_train.shape[0]
    n_test=X_test.shape[0]

    # 🔒 Handle empty test set early
    if n_test == 0:
        return np.empty((0, k), dtype=int)

    # Distance computation
    diff = X_test[:, None, :] - X_train[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)

    sorted_indices = np.argsort(dist_sq, axis=1)

    # Output array
    result = -np.ones((n_test, k), dtype=int)

    cols_to_copy = min(k, n_train)
    result[:, :cols_to_copy] = sorted_indices[:, :cols_to_copy]

    return result


