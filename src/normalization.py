"""
Layer Normalization

This module implements Layer Normalization, a technique used to normalize the inputs across the feature dimension in
neural networks. Layer normalization is particularly effective in transformer architectures where it helps stabilize
training and improve convergence.

Unlike Batch Normalization which normalizes across the batch dimension, Layer Normalization normalizes across the
feature dimension for each sample independently. This makes it particularly suitable for sequence models where batch
sizes may vary and the sequence structure should be preserved.
"""

import torch
import torch.nn as nn

from torch import Tensor


class LayerNorm(nn.Module):
    """
    Layer Normalization implementation.

    Layer Normalization normalizes inputs across the feature dimension, computing mean and variance for each sample
    independently. This is particularly useful in transformer architectures where it helps stabilize training and
    improve model performance.

    The normalization is computed as:
    LayerNorm(x) = γ * (x - μ) / σ + β

    where:
    - μ is the mean across the last dimension
    - σ is the standard deviation across the last dimension
    - γ (scale) and β (shift) are learnable parameters

    Args:
        emb_dim (int): The embedding dimension (number of features to normalize)
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initialize Layer Normalization.

        Args:
            emb_dim (int): Number of features in the input (embedding dimension)
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [..., emb_dim] where ... represents any number of batch dimensions

        Returns:
            Tensor: Normalized tensor with the same shape as input

        Note:
            Normalization is applied across the last dimension (feature dimension). The output will have mean ≈ 0 and
            variance ≈ 1 across the last dimension for each sample, before applying the learnable scale and shift.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
