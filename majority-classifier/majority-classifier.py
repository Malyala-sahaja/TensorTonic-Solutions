import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)

    labels, counts = np.unique(y_train, return_counts=True)
    majority_label = labels[np.argmax(counts)]
    n_test = len(X_test)

    # Predict same label for all test points
    return np.full(n_test, majority_label, dtype=int)

    