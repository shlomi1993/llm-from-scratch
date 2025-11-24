"""
Efficient Multi-Head Attention implementation with unified QKV projections.
"""

import torch

from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention implementation with unified QKV projections.

    This module implements multi-head attention using a more efficient approach where all heads are computed
    simultaneously through tensor reshaping and transposition operations. Unlike the wrapper approach, this
    implementation uses unified linear projections for all heads and then splits the computation, making it more memory
    and computationally efficient. The module includes causal masking for autoregressive language modeling and an output
    projection layer for additional representational capacity.

    The key difference from MultiHeadAttentionWrapper is that this implementation:
    1. Projects to the full d_out dimension and then splits into heads
    2. Computes all heads simultaneously using tensor operations
    3. Includes an output projection layer for learned combination of heads
    4. Is more memory efficient and faster for large models
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the efficient multi-head attention mechanism.

        Computes multi-head attention by:
        1. Projecting input to unified query, key, and value tensors
        2. Reshaping and transposing to separate heads: (batch, tokens, heads, head_dim)
        3. Computing scaled dot-product attention for all heads simultaneously
        4. Applying causal mask to prevent attention to future positions
        5. Applying dropout regularization to attention weights
        6. Combining attention outputs and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, this will result in errors
            in the mask creation. In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs do
            not exceed `context_length` before reaching this forward method.
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
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

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
