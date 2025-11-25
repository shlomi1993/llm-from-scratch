"""
Transformer block implementations for GPT-style language models.

This module contains transformer block implementations that serve as the core building blocks for GPT-style
autoregressive language models. It provides both standard and cached variants with different attention mechanisms.

The transformer blocks implement the Pre-LN architecture with residual connections, layer normalization, and dropout for
stable training and inference.
"""

import torch.nn as nn

from torch import Tensor

from .normalization import LayerNorm
from .attention import MultiHeadAttention, MultiHeadAttentionCached, GroupedQueryAttention
from .feed_forward import FeedForward
from .configurations import GptConfig


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention.

    Implements the Pre-LN (Pre-Layer Normalization) transformer architecture which applies layer normalization before
    the attention and feed-forward sub-layers rather than after. This design choice improves training stability and
    gradient flow compared to the original Post-LN architecture.

    Args:
        config (GptConfig): Configuration object containing model hyperparameters.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the TransformerBlock with the given configuration.

        Args:
            config (GptConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            n_heads=config.n_heads,
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


class TransformerBlockCached(nn.Module):
    """
    Advanced Transformer block with KV-cache support and multiple attention types.

    This enhanced transformer block supports efficient autoregressive generation through KV caching and can use either
    standard Multi-Head Attention (MHA) or Grouped Query Attention (GQA) based on the configuration.

    This block is optimized for autoregressive generation and inference. For training, consider using the standard
    TransformerBlock unless you specifically need the memory benefits of GQA.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the cached transformer block with attention and feed-forward layers.

        Sets up the transformer block based on the configuration, automatically selecting the appropriate attention
        mechanism (GQA vs MHA) and initializing all components with proper parameters for cached operation.

        Args:
            config (GptConfig): Configuration object containing model hyperparameters.

        Attention Selection Logic:
            - config.n_kv_groups > 1: Uses GroupedQueryAttention (GQA)
              - Groups multiple query heads with fewer key-value heads
              - Reduces memory footprint for large models
              - Maintains performance while improving efficiency
            - config.n_kv_groups == 1: Uses MultiHeadAttentionCached (MHA)
              - Standard multi-head attention with sliding window cache
              - Full attention computation with memory optimization
              - Ideal for smaller models and training scenarios

        Note:
            The cache-enabled attention mechanisms support efficient autoregressive generation by reusing previously
            computed key-value pairs, reducing computational complexity from O(n²) to O(n) for new token generation.
        """
        super().__init__()
        if config.n_kv_groups > 1:
            self.att = GroupedQueryAttention(
                d_in=config.emb_dim,
                d_out=config.emb_dim,
                n_heads=config.n_heads,
                num_kv_groups=config.n_kv_groups,
                dropout=config.drop_rate,
                qkv_bias=config.qkv_bias)
        else:
            self.att = MultiHeadAttentionCached(
                d_in=config.emb_dim,
                d_out=config.emb_dim,
                context_length=config.context_length,
                n_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
            window_size=config.kv_window_size or config.context_length
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass through the cached transformer block.

        Performs the same Pre-LN transformer computation as the standard block but with support for KV caching in the
        attention mechanism. The cache enables efficient autoregressive generation by reusing computed key-value pairs
        from previous tokens.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, emb_dim]
            use_cache (bool, optional): Whether to use KV caching. Defaults to False.
                - True: Enable caching for autoregressive generation
                - False: Standard attention without caching (e.g., for training)

        Returns:
            Tensor: Output tensor with same shape as input [batch_size, seq_len, emb_dim]

        Notes:
            1. The cache persists across forward calls until explicitly reset, by calling the reset_kv_cache() method.
            2. Cached mode: O(1) attention for each new token.
            3. Non-cached mode: O(n²) attention for sequence length n.
            4. GQA variant uses fewer memory for key-value storage.
            5. Sliding window maintains constant memory regardless of sequence length.
        """
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
