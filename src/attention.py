import torch

from torch import nn, Tensor

from .config import GptConfig


class SelfAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    @staticmethod
    def from_config(config: GptConfig) -> 'SelfAttention':
        return SelfAttention(d_in=config.emb_dim, d_out=config.emb_dim, qkv_bias=config.qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def from_config(config: GptConfig) -> 'CausalAttention':
        return CausalAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose

        # New, _ ops are in-place
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec


class MultiheadAttentionWrapper(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        attention_heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(n_heads)]
        self.heads = nn.ModuleList(attention_heads)

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionWrapper':
        return MultiheadAttentionWrapper(
            d_in=config.emb_dim,
            d_out=config.emb_dim // config.n_heads,
            context_length=config.context_length,
            dropout=config.drop_rate,
            n_heads=config.n_heads,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiheadAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionCached':
        return MultiheadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            n_heads=config.n_heads,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `n_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, n_heads, head_dim)
        keys = keys.view(b, num_tokens, self.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)

        # Transpose: (b, num_tokens, n_heads, head_dim) -> (b, n_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply scaled softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


class MultiheadAttentionCached(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False,
                 max_seq_len: int = None, window_size: int = None) -> None:

        super().__init__()
        assert d_out % n_heads == 0 == 0, "emb_dim must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        # Linear projections
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

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionCached':
        return MultiheadAttentionCached(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            n_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
            window_size=config.kv_window_size or config.context_length
        )

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
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
        self.cache_k, self.cache_v = None, None
