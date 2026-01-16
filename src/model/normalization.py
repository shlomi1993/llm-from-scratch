import torch
import torch.nn as nn

from torch import Tensor


class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    Normalizes the input tensor across the last dimension to have zero mean and unit variance,
    then applies learnable scaling and shifting.
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initialize the LayerNorm module.

        Args:
            emb_dim (int): The dimensionality of the input tensor.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for layer normalization. Applies normalization to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, emb_size).

        Returns:
            Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
