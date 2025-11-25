"""
Multi-Head Attention implementation using PyTorch's built-in MultiheadAttention module.
"""

import torch

from torch import nn, Tensor


class MHAPyTorchClass(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's built-in MultiheadAttention module.

    This module implements multi-head attention by leveraging PyTorch's nn.MultiheadAttention class, which provides
    a complete implementation of multi-head attention with various optimization options. This approach offers high-level
    abstraction and is well-tested, making it suitable for production use when you want to use PyTorch's official
    implementation rather than custom implementations.

    Key features:
    1. Uses PyTorch's official nn.MultiheadAttention module
    2. Configurable attention weight output for analysis
    3. Built-in optimization and numerical stability features
    4. Explicit causal masking support
    5. Additional output projection layer for enhanced representational capacity
    6. Comprehensive bias options for QKV projections
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, context_length: int, dropout: float = 0.0,
                 qkv_bias: bool = False, need_weights: bool = True) -> None:
        """
        Initialize the MHAPyTorchClass module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension (must be divisible by n_heads)
            n_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
            need_weights (bool, optional): Whether to return attention weights (for analysis). Defaults to True.

        Note:
            The d_in parameter is included for API consistency but is not directly used since
            nn.MultiheadAttention expects the input dimension to match embed_dim.
        """
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            n_heads=n_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PyTorch MultiheadAttention-based mechanism.

        Computes multi-head attention using PyTorch's nn.MultiheadAttention by:
        1. Preparing causal mask for the current sequence length
        2. Calling PyTorch's multihead_attn with query=key=value=x (self-attention)
        3. Applying additional output projection for enhanced representation

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_out)
                       Note: Input dimension must match embed_dim from initialization

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            This implementation uses PyTorch's nn.MultiheadAttention in self-attention mode where
            query, key, and value are all the same input tensor. The causal mask is dynamically
            adjusted based on the actual sequence length to ensure proper autoregressive behavior.
        """
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask, need_weights=self.need_weights)

        output = self.proj(attn_output)

        return output
