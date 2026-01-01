import torch
import torch.nn as nn

from torch import Tensor


class MultiheadAttentionWithSwa(nn.Module):

    def __init__(self, d_in: int, d_out: int, dropout: float, n_heads: int, qkv_bias: bool = False,
                 sliding_window_size: int = None) -> None:
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
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
