"""
Multi-head attention module with sliding window attention (SWA) and KV cache support.

This module implements Sliding Window Attention, which limits the attention span to a fixed window size, providing
memory-efficient attention for long sequences while maintaining causal properties.
"""

import torch
import torch.nn as nn

from torch import Tensor


class MultiheadAttentionWithSwa(nn.Module):
    """
    Multi-Head Attention with Sliding Window Attention (SWA) and KV cache support.

    This implementation combines multi-head attention with a sliding window mechanism that restricts each token to
    attend only to tokens within a fixed-size window. This approach significantly reduces memory consumption for long
    sequences while maintaining strong performance, as most important context is typically local.

    Key Features:
    - Sliding window attention for memory efficiency
    - KV cache support for autoregressive generation
    - Automatic cache trimming to window size
    - Causal masking combined with window constraints
    - Configurable window size for flexibility

    The attention pattern ensures:
    1. Causal constraint: Token i can only attend to tokens j where j <= i
    2. Window constraint: Token i can only attend to the last W tokens (where W is window_size)

    Note:
        When sliding_window_size is None, the attention degenerates to full causal attention without windowing.
        The cache is automatically trimmed when it exceeds the window size to maintain constant memory.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float, n_heads: int, qkv_bias: bool = False,
                 sliding_window_size: int = None) -> None:
        """
        Initialize the Multi-Head Attention with Sliding Window Attention.

        Sets up the linear projections for queries, keys, and values, along with the sliding window cache mechanism.
        The cache dynamically grows and is trimmed to the window size when it exceeds the limit.

        Args:
            d_in (int): Input dimension of the embeddings
            d_out (int): Output dimension (must be divisible by num_heads)
            dropout (float): Dropout probability applied to attention weights
            num_heads (int): Number of parallel attention heads
            qkv_bias (bool): Whether to include bias terms in Q, K, V projections. Defaults to False.
            sliding_window_size (int, optional): Size of the sliding attention window. If None, uses full causal
                attention. When set, limits each token to attend only to the most recent sliding_window_size tokens.
                Defaults to None.

        Raises:
            AssertionError: If d_out is not divisible by n_heads
        """
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size

        # KV cache-related code
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass of Multi-Head Attention with Sliding Window and KV caching.

        Performs multi-head attention computation with sliding window constraints and optional KV caching. The sliding
        window limits attention to the most recent W tokens, where W is sliding_window_size. When caching is enabled,
        keys and values accumulate across forward passes and are automatically trimmed to maintain the window size.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_tokens, d_in]
            use_cache: Whether to use and update the KV cache. Defaults to False.
                - True: Enable caching for autoregressive generation (accumulates K/V across calls)
                - False: Standard attention without caching (stateless operation)

        Returns:
            Tensor: Output tensor of shape [batch_size, num_tokens, d_out] containing the attention-weighted
            representation of the input.
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # KV cache-related
        if use_cache:
            old_len = 0 if self.cache_k is None else self.cache_k.size(1)
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            # Left-trim to sliding window if configured
            if self.sliding_window_size is not None:
                if self.cache_k.size(1) > self.sliding_window_size:
                    self.cache_k = self.cache_k[:, -self.sliding_window_size:, :, :]
                    self.cache_v = self.cache_v[:, -self.sliding_window_size:, :, :]
            # Compute absolute start positions for mask
            total_len = old_len + num_tokens
            k_len_now = self.cache_k.size(1)
            dropped = max(0, total_len - k_len_now)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # causal + sliding-window mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        # Determine absolute positions for q and k
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)
        # Sliding window width
        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self) -> None:
        """
        Reset the KV cache and position pointer.

        Clears the cached keys and values, effectively starting fresh for a new sequence.

        This method should be called:
        - Before starting a new sequence generation
        - When switching between different sequences
        - To clear memory and start fresh
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
