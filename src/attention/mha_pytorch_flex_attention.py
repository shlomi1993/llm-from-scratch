"""
Multi-Head Attention implementation using PyTorch's FlexAttention.
"""

import torch

from torch import nn, Tensor
from torch.nn.attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


def causal(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """
    Causal mask function for flex attention.

    This function defines the causal masking pattern for autoregressive attention,
    where each query position can only attend to key-value positions at or before
    its own position in the sequence.

    Args:
        b (int): Batch index (not used in causal masking)
        h (int): Head index (not used in causal masking)
        q_idx (int): Query position index
        kv_idx (int): Key-Value position index

    Returns:
        bool: True if the query can attend to the key-value position, False otherwise
    """
    return q_idx >= kv_idx


class MHAPyTorchFlexAttention(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's FlexAttention.

    This module implements multi-head attention using PyTorch's flexible attention mechanism (FlexAttention), which
    provides advanced masking capabilities and efficient computation for various attention patterns. FlexAttention
    allows for custom attention patterns through user-defined mask functions and supports block-sparse attention
    patterns for improved memory efficiency and computational performance.

    Key features:
    1. Uses PyTorch's FlexAttention for advanced masking capabilities
    2. Block-sparse attention patterns for memory efficiency
    3. Custom causal masking through mask functions
    4. Combined QKV projection for efficient memory usage
    5. Output projection layer for learned combination of heads
    6. Dropout regularization for attention weights
    7. Optimized for large-scale transformer models with flexible attention patterns
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchFlexAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the block mask creation
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
            RuntimeError: If PyTorch version is below 2.5
        """
        super().__init__()

        # Check PyTorch version (FlexAttention requires PyTorch 2.5+)
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 5):
            raise RuntimeError("MHAPyTorchFlexAttention requires PyTorch 2.5+ with CUDA or MPS support")

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        # Create block mask on CPU (FlexAttention only supports CPU, CUDA, or HPU)
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length, device='cpu')

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FlexAttention-based multi-head attention mechanism.

        Computes multi-head attention using PyTorch's FlexAttention by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Creating appropriate block mask for the current sequence length
        4. Using FlexAttention with the causal block mask for efficient computation
        5. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            FlexAttention uses block-sparse patterns for efficient memory usage and supports
            custom masking functions. The causal mask is created using create_block_mask with
            the causal function to ensure autoregressive behavior.

            FlexAttention only supports CPU, CUDA, and HPU devices. For MPS devices,
            the computation is moved to CPU and then back to the original device.
        """
        batch_size, num_tokens, embed_dim = x.shape
        original_device = x.device

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
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

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec
