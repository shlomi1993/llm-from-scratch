import torch

from torch import nn, Tensor
from torch.nn.attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


def causal(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx >= kv_idx


class MultiheadAttentionPyTorchFlexAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, n_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        super().__init__()

        # Check PyTorch version (FlexAttention requires PyTorch 2.5+)
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 5):
            raise RuntimeError("MHAPyTorchFlexAttention requires PyTorch 2.5+ with CUDA or MPS support")

        assert d_out % n_heads == 0, "d_out is indivisible by n_heads"

        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        # Create block mask on CPU (FlexAttention only supports CPU, CUDA, or HPU)
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length, device='cpu')

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        original_device = x.device

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)

        # (b, num_tokens, 3, n_heads, head_dim) --> (3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, n_heads, num_tokens, head_dim) -> 3 times (b, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # FlexAttention only supports CPU, CUDA, and HPU devices
        # Move to CPU if on unsupported device (like MPS)
        compute_device = 'cpu' if original_device.type not in ['cpu', 'cuda', 'hpu'] else original_device

        if original_device.type not in ['cpu', 'cuda', 'hpu']:
            queries = queries.to('cpu')
            keys = keys.to('cpu')
            values = values.to('cpu')

        # Create block mask with correct dimensions for current sequence length
        attn_mask = create_block_mask(causal, B=None, H=None, Q_LEN=num_tokens, KV_LEN=num_tokens, device=compute_device)

        # For testing/debugging purposes, disable the compile warning
        # In production, you would want to use torch.compile(flex_attention) on CUDA
        original_debug_setting = getattr(torch.nn.attention.flex_attention, '_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG', False)
        torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

        try:
            # Leverage PyTorch's built-in FlexAttention with the block mask
            context_vec = flex_attention(queries, keys, values, block_mask=attn_mask)
        finally:
            # Restore original debug setting
            torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = original_debug_setting

        # Move back to original device if necessary
        if original_device.type not in ['cpu', 'cuda', 'hpu']:
            context_vec = context_vec.to(original_device)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec
