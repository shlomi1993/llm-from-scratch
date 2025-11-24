"""
Causal self-attention mechanism implementation with masking and dropout.
"""

import torch

from torch import nn, Tensor


class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention mechanism implementation for transformer models.

    This module implements the scaled dot-product self-attention mechanism with causal masking, which prevents positions
    from attending to future positions in the sequence. This is essential for autoregressive language models where each
    position should only have access to previous tokens. The module also includes dropout regularization to prevent
    overfitting.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        """
        Initialize the CausalAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the causal self-attention mechanism.

        Computes causal self-attention by:
        1. Projecting input to query, key, and value vectors
        2. Computing attention scores as query-key dot products
        3. Applying causal mask to prevent attention to future positions
        4. Applying scaled softmax to get attention weights
        5. Applying dropout regularization to attention weights
        6. Computing weighted sum of values using attention weights

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Context vectors of shape (batch_size, num_tokens, d_out)

        Note:
            For inputs where `num_tokens` exceeds `context_length`, this will result in errors in the mask creation.
            In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs do not exceed
            `context_length` before reaching this forward method.
        """
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose

        # New, _ ops are in-place
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
