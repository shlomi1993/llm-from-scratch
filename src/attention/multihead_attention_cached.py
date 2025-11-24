"""
Multi-Head Attention with KV cache support.
"""

import torch

from torch import nn, Tensor

from configurations import GptConfig


class MultiHeadAttentionCached(nn.Module):
    """
    Multi-Head Attention with KV cache support.

    Implements sliding window cache optimization and efficient memory management for autoregressive inference.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the cached multi-head attention module.

        Args:
            config (GptConfig): Model configuration containing attention parameters
        """
        super().__init__()
        assert config.emb_dim % config.n_heads == 0, "emb_dim must be divisible by n_heads"

        self.d_out = config.emb_dim
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        self.W_query = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        # KV cache parameters
        self.max_seq_len = config.context_length
        self.window_size = getattr(config, 'kv_window_size', config.context_length)

        # Register cache buffers
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass with optional KV caching.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, emb_dim]
            use_cache (bool): Whether to use KV caching

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, emb_dim]
        """
        b, num_tokens, d_in = x.shape

        # Compute Q, K, V
        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # Reshape for multi-head attention
        keys_new = keys_new.view(b, num_tokens, self.n_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)

        # Transpose for attention computation
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Handle KV caching
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros(b, self.n_heads, self.window_size, self.head_dim, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0

            # Handle cache overflow
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                # Shift cache left
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow

            # Update cache
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = values_new
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        K = attn_scores.size(-1)
        if num_tokens == K:
            # No cache - use triangular mask
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # With cache - offset diagonal
            offset = K - num_tokens
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx

        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        # Apply attention
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

    def reset_cache(self) -> None:
        """
        Reset the KV cache.
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_cur = 0
