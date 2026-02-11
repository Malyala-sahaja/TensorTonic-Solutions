import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    
    LayerNorm normalizes across the feature dimension (last dimension).
    Formula: y = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Args:
        x: Input tensor of shape [..., d_model]
        gamma: Scale parameter of shape [d_model]
        beta: Shift parameter of shape [d_model]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape as x
    """
    # Compute mean and variance along the last dimension (feature dimension)
    # keepdims=True maintains the dimension for broadcasting
    mean = np.mean(x, axis=-1, keepdims=True)  # [..., 1]
    variance = np.var(x, axis=-1, keepdims=True)  # [..., 1]
    
    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + eps)  # [..., d_model]
    
    # Scale and shift
    # gamma and beta broadcast across all batch/sequence dimensions
    output = gamma * x_normalized + beta  # [..., d_model]
    
    return output


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    
    Args:
        Q: Queries [batch, seq_len_q, d_model]
        K: Keys [batch, seq_len_k, d_model]
        V: Values [batch, seq_len_k, d_model]
        W_q, W_k, W_v: Weight matrices [d_model, d_model]
        W_o: Output weight matrix [d_model, d_model]
        num_heads: Number of attention heads
        
    Returns:
        Output [batch, seq_len_q, d_model]
    """
    batch_size, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    d_k = d_model // num_heads
    
    # Step 1: Linear projections
    Q_proj = Q @ W_q  # [batch, seq_len_q, d_model]
    K_proj = K @ W_k  # [batch, seq_len_k, d_model]
    V_proj = V @ W_v  # [batch, seq_len_k, d_model]
    
    # Step 2: Split into multiple heads
    # Reshape: [batch, seq_len, d_model] → [batch, seq_len, num_heads, d_k]
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k)
    
    # Transpose: [batch, seq_len, num_heads, d_k] → [batch, num_heads, seq_len, d_k]
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)
    
    # Step 3: Scaled dot-product attention
    # Compute scores: Q @ K^T
    K_heads_T = K_heads.transpose(0, 1, 3, 2)  # [batch, num_heads, d_k, seq_len_k]
    scores = Q_heads @ K_heads_T  # [batch, num_heads, seq_len_q, seq_len_k]
    
    # Scale
    scores = scores / np.sqrt(d_k)
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)  # [batch, num_heads, seq_len_q, seq_len_k]
    
    # Apply to values
    output = attention_weights @ V_heads  # [batch, num_heads, seq_len_q, d_k]
    
    # Step 4: Concatenate heads
    # Transpose: [batch, num_heads, seq_len_q, d_k] → [batch, seq_len_q, num_heads, d_k]
    output = output.transpose(0, 2, 1, 3)
    
    # Reshape: [batch, seq_len_q, num_heads, d_k] → [batch, seq_len_q, d_model]
    output = output.reshape(batch_size, seq_len_q, d_model)
    
    # Step 5: Output projection
    output = output @ W_o  # [batch, seq_len_q, d_model]
    
    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
    
    This is applied independently to each position (token) in the sequence.
    Typically: d_model → d_ff → d_model (e.g., 512 → 2048 → 512)
    
    Args:
        x: Input [batch, seq_len, d_model]
        W1: First layer weights [d_model, d_ff]
        b1: First layer bias [d_ff]
        W2: Second layer weights [d_ff, d_model]
        b2: Second layer bias [d_model]
        
    Returns:
        Output [batch, seq_len, d_model]
    """
    # First linear layer + ReLU activation
    # x: [batch, seq_len, d_model]
    # W1: [d_model, d_ff]
    # Result: [batch, seq_len, d_ff]
    hidden = x @ W1 + b1  # Broadcasting adds b1 to each position
    
    # ReLU activation: max(0, x)
    hidden = np.maximum(0, hidden)  # [batch, seq_len, d_ff]
    
    # Second linear layer
    # hidden: [batch, seq_len, d_ff]
    # W2: [d_ff, d_model]
    # Result: [batch, seq_len, d_model]
    output = hidden @ W2 + b2
    
    return output


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    
    Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm (residual connection + layer normalization)
    
    Args:
        x: Input [batch, seq_len, d_model]
        W_q, W_k, W_v, W_o: Attention weights
        W1, b1, W2, b2: FFN weights and biases
        gamma1, beta1: Layer norm parameters for first Add & Norm
        gamma2, beta2: Layer norm parameters for second Add & Norm
        num_heads: Number of attention heads
        
    Returns:
        Output [batch, seq_len, d_model]
    """
    # Sub-layer 1: Multi-Head Self-Attention
    # In self-attention, Q = K = V = x
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # Add & Norm 1: Residual connection + Layer normalization
    # x + sublayer(x)
    x = x + attn_output  # Residual connection
    x = layer_norm(x, gamma1, beta1)  # Layer normalization
    
    # Sub-layer 2: Feed-Forward Network
    ffn_output = feed_forward(x, W1, b1, W2, b2)
    
    # Add & Norm 2: Residual connection + Layer normalization
    x = x + ffn_output  # Residual connection
    x = layer_norm(x, gamma2, beta2)  # Layer normalization
    
    return x