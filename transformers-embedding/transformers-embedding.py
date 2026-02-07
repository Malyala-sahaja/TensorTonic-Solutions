import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Create the embedding layer
    # This allocates a [vocab_size, d_model] matrix of learnable parameters
    embedding = nn.Embedding(vocab_size, d_model)
    
    # Initialize with scaled normal distribution
    # Standard deviation = 1/sqrt(d_model) helps with training stability
    # This is a common initialization strategy (similar to Xavier)
    nn.init.normal_(embedding.weight, mean=0.0, std=d_model**-0.5)
    
    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Lookup: For each token index, retrieve its embedding vector
    # Input shape: [batch_size, seq_len] or [seq_len]
    # Output shape: [batch_size, seq_len, d_model] or [seq_len, d_model]
    embedded = embedding(tokens)
    
    # Scale by sqrt(d_model) as per "Attention Is All You Need" paper
    # This prevents embeddings from dominating over positional encodings
    scaled_embedded = embedded * math.sqrt(d_model)
    
    return scaled_embedded