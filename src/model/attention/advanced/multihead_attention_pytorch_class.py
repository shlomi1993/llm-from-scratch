import torch

from torch import nn, Tensor


class MultiheadAttentionPyTorchClass(nn.Module):

    def __init__(self, d_in: int, d_out: int, n_heads: int, context_length: int, dropout: float = 0.0,
                 qkv_bias: bool = False, need_weights: bool = True) -> None:
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask, need_weights=self.need_weights)

        output = self.proj(attn_output)

        return output
