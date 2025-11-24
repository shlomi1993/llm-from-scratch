"""
Multi-Head Attention implementation with combined QKV projections.
"""

import torch

from torch import nn, Tensor


class MultiHeadAttentionCombinedQKV(nn.Module):
    """
    Efficient Multi-Head Attention implementation with combined QKV projections.

    This module implements multi-head attention using a highly optimized approach where query, key, and value
    projections are computed in a single linear transformation. This reduces the number of matrix multiplications and
    improves computational efficiency compared to separate QKV projections. The implementation uses advanced tensor
    reshaping and permutation operations to efficiently separate and process multiple attention heads simultaneously.

    Key features:
    1. Single linear layer computes all QKV projections at once (3 * d_out dimensions)
    2. Advanced tensor permutations for efficient head separation
    3. Causal masking for autoregressive language modeling
    4. Output projection layer for learned combination of heads
    5. Dropout regularization for attention weights
    6. Memory and computationally efficient for large-scale models
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttentionCombinedQKV module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
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

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the combined QKV multi-head attention mechanism.

        Computes multi-head attention using combined QKV projection by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Computing scaled dot-product attention for all heads simultaneously
        4. Applying causal mask to prevent attention to future positions
        5. Applying dropout regularization to attention weights
        6. Combining attention outputs and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            For inputs where `num_tokens` exceeds `context_length`, this will result in errors in the mask creation.
            This implementation uses advanced tensor operations for maximum efficiency, including permute and unbind
            operations for optimal memory layout and computation.
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec
