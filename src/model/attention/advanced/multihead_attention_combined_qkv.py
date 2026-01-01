import torch

from torch import nn, Tensor


class MultiheadAttentionCombinedQKV(nn.Module):

    def __init__(self, d_in: int, d_out: int, n_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out is indivisible by n_heads"

        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)

        # (b, num_tokens, 3, n_heads, head_dim) --> (3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, n_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, n_heads, num_tokens, head_dim) --> (b, n_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, n_heads, num_tokens, num_tokens) --> (b, n_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, n_heads, num_tokens, head_dim) --> (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, n_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec
