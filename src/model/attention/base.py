import torch

from torch import nn, Tensor

from src.model.config import GptConfig


class SelfAttention(nn.Module):
    """
    A basic self-attention mechanism without causality constraints.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """
        Initialize the Self-Attention layer.

        Args:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            qkv_bias (bool, optional): Whether to use bias in QKV projections. Default is False.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    @staticmethod
    def from_config(config: GptConfig) -> 'SelfAttention':
        """
        Create a SelfAttention layer from a GptConfig object.
        """
        return SelfAttention(d_in=config.emb_dim, d_out=config.emb_dim, qkv_bias=config.qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Self-Attention layer. Applies attention mechanism to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (num_tokens, emb_size).

        Returns:
            Tensor: Output tensor after applying self-attention.
        """
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """
    A causal self-attention mechanism that prevents attending to future tokens.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        """
        Initialize the Causal Attention layer.

        Args:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            context_length (int): Maximum context length for attention.
            dropout (float): Dropout rate for attention weights.
            qkv_bias (bool, optional): Whether to use bias in QKV projections. Default is False.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def from_config(config: GptConfig) -> 'CausalAttention':
        """
        Create a CausalAttention layer from a GptConfig object.
        """
        return CausalAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Causal Attention layer. Applies causal attention mechanism to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (num_tokens, emb_size).

        Returns:
            Tensor: Output tensor after applying causal attention.
        """
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)

        attn_scores: Tensor = queries @ keys.transpose(1, 2)  # Changed transpose

        # New, _ ops are in-place
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
