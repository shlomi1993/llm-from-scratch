import torch
import torch.nn as nn

from torch import Tensor


class MultiheadLatentAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, dropout: float, num_heads: int, qkv_bias: bool = False, latent_dim: int = None) -> None:
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

    @staticmethod
    def _reshape_to_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
        # (b, T, d_out) -> (b, num_heads, T, head_dim)
        bsz, num_tokens, _ = x.shape
        return x.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2).contiguous()

    def reset_cache(self) -> None:
        self.cache_c_kv = None
        self.ptr_current_pos = 0

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
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
