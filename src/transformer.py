"""
Transformer Block

This module implements a single transformer decoder block, which is the fundamental building component of GPT models.
Each transformer block consists of a multi-head attention layer followed by a position-wise feed-forward network, with
residual connections and layer normalization around each sub-layer.

The implementation follows the "Pre-LN" (Pre-Layer Normalization) variant where layer normalization is applied before
the attention and feed-forward computations, which has been shown to improve training stability compared to the original
"Post-LN" design.
"""

import torch.nn as nn

from torch import Tensor

from .normalization import LayerNorm
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .configurations import GptConfig


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block.

    This class implements a transformer decoder block used in GPT models. The block consists of masked multi-head
    self-attention followed by a position-wise feed-forward network, with residual connections and layer normalization
    around each sub-layer.

    Args:
        config (GptConfig): Configuration containing model parameters.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the transformer block.

        Args:
            config (GptConfig): Model configuration
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the transformer block to the input.

        Implements the Pre-LN transformer block with residual connections:
        1. Apply layer normalization and multi-head attention with residual connection
        2. Apply layer normalization and feed-forward network with residual connection

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, emb_dim]

        Returns:
            Tensor: Output tensor with the same shape as input [batch_size, seq_len, emb_dim]

        Note:
            The function applies causal masking in the attention mechanism, ensuring that each position can only attend
            to previous positions and itself, which is essential for autoregressive language modeling.
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
