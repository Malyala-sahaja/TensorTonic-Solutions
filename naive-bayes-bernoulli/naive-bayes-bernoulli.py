import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    # Get dimensions
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    # Find unique classes and sort them
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Step 1: Compute class priors probabilities P(y) in log-space
    # log P(y) = log(count(y) / n_train)
    log_priors = np.zeros(n_classes)
    for idx, c in enumerate(classes):
        class_count = np.sum(y_train == c)
        log_priors[idx] = np.log(class_count / n_train)

    # Step 2: Compute feature likelihoods P(x_i=1|y) with Laplace smoothing
    # θ_iy(likelihood probability) = P(x_i=1|y) = (count(x_i=1 in class y) + α) / (n_y + 2α)
    # where α = 1 (Laplace smoothing)
    
    # Initialize array to store P(x_i=1|y) for each feature and class
    # Shape: [n_classes, n_features]
    theta = np.zeros((n_classes, n_features))
    
    for idx, c in enumerate(classes):
        # Get samples belonging to class c
        X_c = X_train[y_train == c]  # [n_c, n_features]
        n_c = X_c.shape[0]
        # Count how many times each feature is 1 in class c
        count_ones = np.sum(X_c, axis=0)  # [n_features]
        # Apply Laplace smoothing: (count + 1) / (n_c + 2)
        theta[idx] = (count_ones + 1) / (n_c + 2)

        
    # Step 3: Compute log probabilities for test samples
    # log P(y|x) ∝ log P(y) + Σ log P(x_i|y)
    # log P(x_i|y) = x_i * log(θ_iy) + (1 - x_i) * log(1 - θ_iy)
    
    # Compute log(theta) and log(1 - theta)
    log_theta = np.log(theta)          # [n_classes, n_features]
    log_one_minus_theta = np.log(1 - theta)  # [n_classes, n_features]
    
    # Initialize result array
    log_posteriors = np.zeros((n_test, n_classes))
    
    # For each test sample
    for i in range(n_test):
        x = X_test[i]  # [n_features]
        
        # For each class
        for idx in range(n_classes):
            # Start with log prior
            log_prob = log_priors[idx]
            
            # Add log likelihood for each feature
            # log P(x_i|y) = x_i * log(θ) + (1 - x_i) * log(1 - θ)
            log_likelihood = x * log_theta[idx] + (1 - x) * log_one_minus_theta[idx]
            log_prob += np.sum(log_likelihood)
            
            log_posteriors[i, idx] = log_prob
    
    return log_posteriors

