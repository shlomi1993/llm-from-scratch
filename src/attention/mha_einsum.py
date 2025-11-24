"""
Multi-Head Attention implementation using Einstein summation operations.
"""

import torch
import math

from torch import nn, Tensor


class MHAEinsum(nn.Module):
    """
    Multi-Head Attention implementation using Einstein summation (einsum) operations.

    This module implements multi-head attention using einsum operations for tensor contractions, providing an
    alternative implementation that demonstrates the mathematical operations more explicitly. The einsum notation makes
    the tensor dimension manipulations clearer and can be more readable for understanding the underlying mathematics of attention mechanisms.

    Key features:
    1. Uses einsum operations for explicit tensor contractions
    2. Manual parameter initialization with Kaiming uniform for better training dynamics
    3. Supports optional biases for QKV projections
    4. Causal masking for autoregressive language modeling
    5. Output projection layer for learned combination of heads
    6. Dropout regularization for attention weights
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAEinsum module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV projections. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform initialization.

        This method initializes the query, key, and value weight matrices using Kaiming uniform initialization, which is
        suitable for layers with ReLU-like activations. The bias terms (if used) are initialized uniformly within a
        bound derived from the fan-in of the weights.
        """
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the einsum-based multi-head attention mechanism.

        Computes multi-head attention using einsum operations by:
        1. Computing Q, K, V projections using einsum for explicit tensor contractions
        2. Adding optional biases to the projections
        3. Reshaping and transposing to separate heads
        4. Computing scaled dot-product attention using einsum operations
        5. Applying causal mask to prevent attention to future positions
        6. Applying dropout regularization to attention weights
        7. Aggregating context vectors using einsum operations
        8. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            The einsum operations use the following notation:
            - 'bnd,do->bno': batch matrix multiplication for linear projections
            - 'bhnd,bhmd->bhnm': scaled dot-product attention computation
            - 'bhnm,bhmd->bhnd': context vector aggregation
            where b=batch, n=sequence_length, d=feature_dim, h=heads, m=sequence_length, o=output_dim
        """
        b, n, _ = x.shape

        # Calculate Q, K, V using einsum, first perform linear transformations
        Q = torch.einsum("bnd,do->bno", x, self.W_query)
        K = torch.einsum("bnd,do->bno", x, self.W_key)
        V = torch.einsum("bnd,do->bno", x, self.W_value)

        # Add biases if they are used
        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        # Reshape for multi-head attention
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)

        # Apply mask
        mask = self.mask[:n, :n]
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate the attended context vectors
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # Combine heads and project the output
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
