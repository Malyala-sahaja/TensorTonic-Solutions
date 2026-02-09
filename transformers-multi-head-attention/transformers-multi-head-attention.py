import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Get dimensions
    batch_size, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    
    # Calculate dimension per head
    d_k = d_model // num_heads
    
    # Step 1: Linear projections
    # Q @ W_q: [batch, seq_len_q, d_model] @ [d_model, d_model] 
    #        = [batch, seq_len_q, d_model]
    Q_proj = Q @ W_q  # [batch, seq_len_q, d_model]
    K_proj = K @ W_k  # [batch, seq_len_k, d_model]
    V_proj = V @ W_v  # [batch, seq_len_k, d_model]
    
    # Step 2: Split into multiple heads
    # Reshape from [batch, seq_len, d_model] to [batch, seq_len, num_heads, d_k]
    # Then transpose to [batch, num_heads, seq_len, d_k]
    
    # Reshape: [batch, seq_len_q, d_model] → [batch, seq_len_q, num_heads, d_k]
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k)
    
    # Transpose: [batch, seq_len, num_heads, d_k] → [batch, num_heads, seq_len, d_k]
    Q_heads = Q_heads.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len_q, d_k]
    K_heads = K_heads.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len_k, d_k]
    V_heads = V_heads.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len_k, d_k]
    
    # Step 3: Scaled dot-product attention for each head
    # Compute attention scores: Q @ K^T
    # Q_heads: [batch, num_heads, seq_len_q, d_k]
    # K_heads.T: [batch, num_heads, d_k, seq_len_k]
    # Result: [batch, num_heads, seq_len_q, seq_len_k]
    
    # Transpose last two dimensions of K_heads
    K_heads_T = K_heads.transpose(0, 1, 3, 2)  # [batch, num_heads, d_k, seq_len_k]
    
    # Compute scores
    scores = Q_heads @ K_heads_T  # [batch, num_heads, seq_len_q, seq_len_k]
    
    # Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)  # [batch, num_heads, seq_len_q, seq_len_k]
    
    # Apply attention weights to values
    # attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
    # V_heads: [batch, num_heads, seq_len_k, d_k]
    # Result: [batch, num_heads, seq_len_q, d_k]
    output = attention_weights @ V_heads  # [batch, num_heads, seq_len_q, d_k]
    
    # Step 4: Concatenate heads
    # Transpose back: [batch, num_heads, seq_len_q, d_k] → [batch, seq_len_q, num_heads, d_k]
    output = output.transpose(0, 2, 1, 3)  # [batch, seq_len_q, num_heads, d_k]
    
    # Reshape to concatenate: [batch, seq_len_q, num_heads, d_k] → [batch, seq_len_q, d_model]
    output = output.reshape(batch_size, seq_len_q, d_model)
    
    # Step 5: Final linear projection
    # [batch, seq_len_q, d_model] @ [d_model, d_model] = [batch, seq_len_q, d_model]
    output = output @ W_o
    
    return output