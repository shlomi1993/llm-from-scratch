"""
Multi-Head Attention with KV cache support.
"""

import torch

from torch import nn, Tensor


class MultiheadAttentionCached(nn.Module):
    """
    Multi-Head Attention with KV cache support and sliding window optimization.

    This implementation provides efficient memory management for autoregressive inference through a sliding window cache
    mechanism. When the cache exceeds the window size, older tokens are automatically shifted out to make room for new
    ones, maintaining constant memory usage regardless of sequence length.

    Key Features:
    - Sliding window KV cache for memory efficiency
    - Automatic cache overflow handling
    - Causal masking for autoregressive generation
    - Configurable window size and maximum sequence length
    - Production-ready for long sequence generation

    Note:
        The sliding window cache maintains the most recent `window_size` tokens. When the cache overflows, older tokens
        are discarded, which may affect attention to very distant context.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False,
                 max_seq_len: int = None, window_size: int = None) -> None:
        """
        Initialize the Multi-Head Attention with KV cache.

        Sets up the linear projections for queries, keys, and values, along with the sliding window cache buffers. The
        cache is dynamically allocated during the first forward pass to match the batch size.

        Args:
            d_in: Input dimension of the embeddings
            d_out: Output dimension (must be divisible by n_heads)
            context_length: Maximum context length for standard attention
            dropout: Dropout probability applied to attention weights
            n_heads: Number of parallel attention heads
            qkv_bias: Whether to include bias terms in Q, K, V projections
            max_seq_len: Maximum sequence length to support (defaults to context_length)
            window_size: Cache window size for sliding window (defaults to max_seq_len)

        Raises:
            AssertionError: If d_out is not divisible by n_heads

        Note:
            Cache buffers (cache_k, cache_v) are registered as non-persistent buffers and are allocated lazily during
            the first forward pass.
        """
        super().__init__()
        assert d_out % n_heads == 0 == 0, "emb_dim must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        # KV cache parameters
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass of Multi-Head Attention with optional KV caching.

        Performs multi-head attention computation with support for KV caching and sliding window optimization. When
        caching is enabled, keys and values are stored in a circular buffer for efficient autoregressive generation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            use_cache: Whether to use KV caching for autoregressive generation

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, n_tokens, d_in = x.shape

        # Compute Q, K, V
        keys_new = self.W_key(x)  # Shape: (batch_size, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # Reshape for multi-head attention
        keys_new = keys_new.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        values_new = values_new.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        # Transpose for attention computation
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Handle KV caching
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != batch_size:
                self.cache_k = torch.zeros(batch_size, self.n_heads, self.window_size, self.head_dim, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # pointer to next free slot

            # Handle cache overflow
            if self.ptr_cur + n_tokens > self.window_size:
                overflow = self.ptr_cur + n_tokens - self.window_size
                # Shift cache left
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow

            # Update cache
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + n_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + n_tokens, :] = values_new
            self.ptr_cur += n_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        K = attn_scores.size(-1)
        if n_tokens == K:
            # No cache - use triangular mask
            causal_mask = torch.triu(torch.ones(n_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # With cache - offset diagonal
            offset = K - n_tokens
            row_idx = torch.arange(n_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx

        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        # Apply attention
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        context_vec = (attn_weights @ values).transpose(1, 2)  # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(batch_size, n_tokens, self.d_out)
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
