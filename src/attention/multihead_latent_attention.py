"""
Multi-Head Latent Attention (MLA) implementation with compressed KV cache.

This module implements Multi-Head Latent Attention, which compresses key-value representations
into a lower-dimensional latent space before caching, significantly reducing memory usage
while maintaining model performance.
"""

import torch
import torch.nn as nn

from torch import Tensor


class MultiheadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention with compressed KV cache.

    This implementation compresses key and value representations into a lower-dimensional latent space before storing in
    the KV cache, reducing memory requirements for long-context generation. The latent representations are then
    up-projected back to full dimensionality for attention computation.

    The MLA approach:
    1. Projects queries directly to per-head dimension (like standard MHA)
    2. Compresses keys/values to a latent dimension via W_DKV
    3. Caches the compressed latent representations
    4. Up-projects cached latents to full key/value dimensions via W_UK and W_UV
    5. Performs standard scaled dot-product attention with causal masking

    This is particularly beneficial for:
    - Long-context generation where KV cache size becomes prohibitive
    - Models with many attention heads where cache memory dominates
    - Scenarios requiring efficient memory usage without significant quality loss

    References:
        Inspired by DeepSeek's Multi-Head Latent Attention mechanism.
        https://huggingface.co/bird-of-paradise/deepseek-mla
    """

    def __init__(self, d_in: int, d_out: int, dropout: float, num_heads: int, qkv_bias: bool = False, latent_dim: int = None) -> None:
        """
        Initialize the MultiheadLatentAttention module.

        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Total output embedding dimension (must be divisible by num_heads).
            dropout (float): Dropout probability for attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
            latent_dim (int, optional): Dimension of the compressed latent space for KV cache. If None, defaults to max(16, d_out // 8).

        Raises:
            AssertionError: If d_out is not divisible by num_heads.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Initialize dimensions and latent dimension
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        # Projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)            # per-head Q
        self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)    # down to latent C
        self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)    # latent -> per-head K
        self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)    # latent -> per-head V

        # Output projection
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Latent-KV cache
        self.register_buffer("cache_c_kv", None, persistent=False)
        self.ptr_current_pos = 0

    def reset_cache(self) -> None:
        """
        Reset the KV cache and position pointer.

        This method clears the cached latent representations and resets the position counter, preparing the module for a
        new generation sequence.
        """
        self.cache_c_kv = None
        self.ptr_current_pos = 0

    @staticmethod
    def _reshape_to_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
        """
        Reshape tensor from (batch, seq_len, d_out) to (batch, num_heads, seq_len, head_dim).

        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_out).
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.

        Returns:
            Tensor reshaped to (batch_size, num_heads, num_tokens, head_dim).
        """
        # (b, T, d_out) -> (b, num_heads, T, head_dim)
        bsz, num_tokens, _ = x.shape
        return x.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2).contiguous()

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Compute multi-head latent attention.

        This method performs the following steps:
        1. Projects input to queries (full dimension) and latent representations
        2. Updates the latent KV cache if use_cache is True
        3. Up-projects latent representations to full key and value dimensions
        4. Reshapes Q, K, V to multi-head format
        5. Computes scaled dot-product attention with causal masking
        6. Combines heads and applies output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in).
            use_cache (bool, optional): Whether to use KV caching for efficient generation. When True, the latent
                representations are cached and reused across timesteps. Defaults to False.

        Returns:
            Tensor: Output tensor of shape (batch_size, num_tokens, d_out) after multi-head latent attention and output
                projection.

        Note:
            When use_cache is True, this method maintains state across calls via self.cache_c_kv and
            self.ptr_current_pos. Call reset_cache() before starting a new generation sequence.
        """
        b, num_tokens, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        # 1) Project to queries (per-token, per-head) and new latent chunk
        queries_all = self.W_query(x)  # (b, T, d_out)
        latent_new = self.W_DKV(x)  # (b, T, latent_dim)

        # 2) Update latent cache and choose latent sequence to up-project
        if use_cache:
            if self.cache_c_kv is None:
                latent_total = latent_new
            else:
                latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1)
            self.cache_c_kv = latent_total
        else:
            latent_total = latent_new

        # 3) Up-project latent to per-head keys/values (then split into heads)
        keys_all = self.W_UK(latent_total)   # (b, T_k_total, d_out)
        values_all = self.W_UV(latent_total)   # (b, T_k_total, d_out)

        # 4) Reshape to heads
        queries = self._reshape_to_heads(queries_all, num_heads, head_dim)
        keys = self._reshape_to_heads(keys_all, num_heads, head_dim)
        values = self._reshape_to_heads(values_all, num_heads, head_dim)

        # 5) Scaled dot-product attention with causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))

        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

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
