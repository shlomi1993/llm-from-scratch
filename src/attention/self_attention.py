"""
Self-attention mechanism implementation.
"""

import torch

from torch import nn, Tensor


class SelfAttention(nn.Module):
    """
    Self-attention mechanism implementation for transformer models.

    This module implements the scaled dot-product self-attention mechanism, which allows each position in a sequence to
    attend to all positions in the same sequence. It computes attention weights based on the similarity between query
    and key vectors, then uses these weights to create a weighted combination of value vectors.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """
        Initialize the SelfAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the self-attention mechanism.

        Computes self-attention by:
        1. Projecting input to query, key, and value vectors
        2. Computing attention scores as query-key dot products
        3. Applying scaled softmax to get attention weights
        4. Computing weighted sum of values using attention weights

        Args:
            x (Tensor): Input tensor of shape (seq_len, d_in)

        Returns:
            Tensor: Context vectors of shape (seq_len, d_out)
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
