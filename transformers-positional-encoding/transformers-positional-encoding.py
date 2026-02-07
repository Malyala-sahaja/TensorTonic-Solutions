import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model))
    
    # Create position indices: [0, 1, 2, ..., seq_length-1]
    # Shape: (seq_length, 1) for broadcasting
    position = np.arange(seq_length).reshape(-1, 1)
    
    # Compute the division term: 10000^(2i/d_model)
    # For numerical stability, we use: 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
    # Create dimension indices: [0, 1, 2, ..., d_model-1]
    # But we need [0, 2, 4, ...] for the formula, so we use np.arange(0, d_model, 2)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    # Shape: (d_model/2,)
    
    # Apply sine to even indices (0, 2, 4, ...)
    # position * div_term broadcasts to shape (seq_length, d_model/2)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe