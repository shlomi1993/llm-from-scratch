"""
Multi-Head Attention implementation using PyTorch's scaled_dot_product_attention without FlashAttention.
"""

import torch

from torch import nn, Tensor


class MultiheadAttentionPyTorchSdpaWithoutFlash(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's scaled_dot_product_attention without FlashAttention.

    This module implements multi-head attention using PyTorch's scaled_dot_product_attention function while explicitly
    disabling FlashAttention optimizations. This can be useful for debugging, compatibility testing, or when specific
    attention computation behavior is required. Unlike the regular PyTorch scaled dot-product attention, this
    implementation uses explicit masking rather than the is_causal parameter.

    Key features:
    1. Uses PyTorch's scaled_dot_product_attention with explicit masking
    2. Disables FlashAttention for consistent behavior across different hardware
    3. Combined QKV projection for efficient memory usage
    4. Explicit causal masking with registered buffer
    5. Output projection layer for learned combination of heads
    6. Training-aware dropout handling
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchSDPAWithoutFlash module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by n_heads)
            n_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by n_heads
        """
        super().__init__()

        assert d_out % n_heads == 0, "d_out is indivisible by n_heads"

        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PyTorch SDPA without FlashAttention.

        Computes multi-head attention using PyTorch's scaled_dot_product_attention with explicit masking by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Using PyTorch's scaled_dot_product_attention with explicit causal mask
        4. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)

        # (b, num_tokens, 3, n_heads, head_dim) --> (3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, n_heads, num_tokens, head_dim) -> 3 times (b, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # Use Dropout only during training
        use_dropout = 0. if not self.training else self.dropout

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # Leverage PyTorch's built-in scaled_dot_product_attention with explicit mask
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec
