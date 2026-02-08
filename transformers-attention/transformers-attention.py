import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Step 1: Get the dimension of keys (for scaling)
    d_k = Q.size(-1)  # Last dimension of Q
    
    # Step 2: Compute attention scores (QK^T)
    # Q: [batch, seq_len_q, d_k]
    # K^T: [batch, d_k, seq_len_k] (transposed)
    # Result: [batch, seq_len_q, seq_len_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 3: Scale by sqrt(d_k)
    # Prevents scores from becoming too large
    scores = scores / math.sqrt(d_k)
    
    # Step 4: Apply mask (if provided)
    # if mask is not None:
    #     # Where mask is 0, set scores to very negative number
    #     # This makes softmax output ~0 for those positions
    #     scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 5: Apply softmax to get attention weights
    # Converts scores to probabilities (sum to 1)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 6: Apply attention weights to values
    # attention_weights: [batch, seq_len_q, seq_len_k]
    # V: [batch, seq_len_k, d_v]
    # Result: [batch, seq_len_q, d_v]
    output = torch.matmul(attention_weights, V)
    
    return output