import torch.nn as nn

from torch import Tensor

from .attention.multihead import MultiheadAttentionCached
from .config import GptConfig
from .feed_forward import FeedForward
from .normalization import LayerNorm


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head self-attention and feed-forward layers,
    each followed by layer normalization and residual connections.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the Transformer block with attention and feed-forward layers.

        Args:
            config (GptConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.att = MultiheadAttentionCached.from_config(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, emb_size).
            use_cache (bool, optional): Whether to use caching for attention. Default is False.

        Returns:
            Tensor: Output tensor after passing through the Transformer block.
        """

        # Shortcut connection for attention block  # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block  # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
