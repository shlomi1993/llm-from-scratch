"""
Multi-Head Attention implementation using PyTorch's built-in scaled_dot_product_attention.
"""

import torch

from torch import nn, Tensor


class MHAPyTorchScaledDotProduct(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's built-in scaled_dot_product_attention.

    This module implements multi-head attention by leveraging PyTorch's optimized
    nn.functional.scaled_dot_product_attention function, which provides hardware-accelerated attention computation with
    automatic optimization for different backends (FlashAttention, memory-efficient attention, etc.). This
    implementation offers the best performance and is the recommended approach for production use.

    Key features:
    1. Uses PyTorch's optimized scaled_dot_product_attention for maximum performance
    2. Automatic backend selection (FlashAttention, memory-efficient, etc.)
    3. Built-in causal masking support
    4. Efficient memory usage and computation
    5. Combined QKV projection for reduced memory bandwidth
    6. Output projection layer for learned combination of heads
    7. Training-aware dropout handling
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchScaledDotProduct module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length (for compatibility, not directly used)
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PyTorch scaled dot-product attention mechanism.

        Computes multi-head attention using PyTorch's optimized scaled_dot_product_attention by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Using PyTorch's scaled_dot_product_attention with causal masking
        4. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            This implementation leverages PyTorch's optimized attention kernels which automatically select the best
            backend (FlashAttention, memory-efficient attention, etc.) based on the input shapes and available hardware.
            The is_causal=True parameter enables automatic causal masking without requiring explicit mask creation.
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # Use Dropout only during training
        use_dropout = 0. if not self.training else self.dropout

        # Leverage PyTorch's built-in scaled_dot_product_attention with causal masking
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec
