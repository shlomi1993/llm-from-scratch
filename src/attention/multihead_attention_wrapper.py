"""
Multi-Head Attention wrapper implementation.
"""

import torch

from torch import nn, Tensor

from .causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
    Multi-Head Attention wrapper for parallel attention computation.

    This module implements multi-head attention by running multiple CausalAttention heads in parallel and concatenating
    their outputs. Each head learns different representation subspaces, allowing the model to jointly attend to
    information from different representation subspaces at different positions.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttentionWrapper module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension per head
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            n_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        attention_heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(n_heads)]
        self.heads = nn.ModuleList(attention_heads)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Computes multi-head attention by:
        1. Running each attention head on the input in parallel
        2. Concatenating all head outputs along the feature dimension

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Concatenated attention outputs of shape (batch_size, num_tokens, d_out * n_heads)
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)
