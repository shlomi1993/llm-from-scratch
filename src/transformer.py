import torch.nn as nn

from torch import Tensor

from .attention import MultiheadAttentionCached
from .config import GptConfig
from .feed_forward import FeedForward
from .normalization import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(self, config: GptConfig) -> None:
        super().__init__()
        self.att = MultiheadAttentionCached.from_config(config)
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
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
