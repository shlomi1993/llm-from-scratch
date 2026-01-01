import torch
import torch.nn as nn

from torch import Tensor


class GroupedQueryAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, dropout: float, n_heads: int, num_kv_groups: int,
                 dtype: torch.dtype = None, qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        assert n_heads % num_kv_groups == 0, "n_heads must be divisible by num_kv_groups"

        # Calculate group size for KV heads
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        # Projections for keys and values with reduced number of heads
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = n_heads // num_kv_groups

        # Projection for queries remains the same as in standard MHA
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # KV cache parameters
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, n_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Handle KV caching
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys_base, values_base = self.cache_k, self.cache_v

        # If not using cache, reset cache and position pointer
        else:
            keys_base, values_base = keys_new, values_new
            if self.cache_k is not None or self.cache_v is not None:
                self.cache_k, self.cache_v = None, None
                self.ptr_current_pos = 0

        # Expand keys and values to match the number of heads
        # Shape: (b, n_heads, num_tokens, head_dim)
        keys = keys_base.repeat_interleave(self.group_size, dim=1)  # Shape: (b, n_heads, num_tokens, head_dim)
        values = values_base.repeat_interleave(self.group_size, dim=1)  # Shape: (b, n_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, n_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # causal mask
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
        mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
