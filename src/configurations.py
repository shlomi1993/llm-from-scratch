"""
GPT Model Configuration

This module defines configurations for GPT model architectures.

The configurations are based on the original GPT-2 paper specifications and include parameters for embedding dimensions,
layer counts, attention heads, vocabulary size, context length, dropout rates, and bias settings.

References:
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- https://scholar.google.com/citations?view_op=view_citation&hl=en&user=dOad5HoAAAAJ&citation_for_view=dOad5HoAAAAJ:YsMSGLbcyi4C
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class GptConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass defines all the hyperparameters needed to configure a GPT model, including architectural parameters,
    training settings, and model-specific options.

    Attributes:
        emb_dim (int): Embedding dimension (hidden size) of the model
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads in each layer
        vocab_size (int): Size of the vocabulary (default: 50257, GPT-2 vocab size)
        context_length (int): Maximum sequence length the model can handle (default: 1024)
        drop_rate (float): Dropout probability for regularization (default: 0.1)
        qkv_bias (bool): Whether to use bias in Query-Key-Value projections (default: False)

    Note:
        The embedding dimension should be divisible by the number of heads for proper multi-head attention computation.
    """
    emb_dim: int                # Embedding dimension
    n_layers: int               # Number of layers
    n_heads: int                # Number of attention heads
    vocab_size: int = 50257     # Vocabulary size
    context_length: int = 1024  # Context length
    drop_rate: float = 0.1      # Dropout rate
    qkv_bias: bool = False      # Query-Key-Value bias


# GPT2-Small (124M parameters)
GPT_CONFIG_124M = GptConfig(
    emb_dim=768,
    n_layers=12,
    n_heads=12
)


# GPT2-Medium (355M parameters)
GPT_CONFIG_355M = GptConfig(
    emb_dim=1024,
    n_layers=24,
    n_heads=16
)


# GPT2-Large (774M parameters)
GPT_CONFIG_774M = GptConfig(
    emb_dim=1280,
    n_layers=36,
    n_heads=20
)


# GPT2-XL (1558M parameters)
GPT_CONFIG_1558M = GptConfig(
    emb_dim=1600,
    n_layers=48,
    n_heads=25
)
