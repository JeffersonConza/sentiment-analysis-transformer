"""
Configuration module for SentimentScope Transformer model and training.
"""
import torch

DEFAULT_CONFIG = {
    "vocabulary_size": 30522,  # bert-base-uncased default vocabulary size
    "num_classes": 2,          # Binary classification: 0 (Negative), 1 (Positive)
    "d_embed": 128,            # Embedding dimension
    "context_size": 128,       # Maximum sequence length (tokens)
    "layers_num": 4,           # Number of transformer blocks
    "heads_num": 4,            # Number of attention heads
    "head_size": 32,           # Dimension per head (4 * 32 = 128)
    "dropout_rate": 0.1,       # Dropout regularization rate
    "use_bias": True,          # Linear layer bias
    "causal": True             # True to match the trained sentiment_transformer.pt checkpoint
}


def get_device():
    """Detect and return available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
