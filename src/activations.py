"""
Activation Functions

This module implements custom activation functions for neural networks.
"""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth differentiable activation function that outperforms ReLU in many contexts, particularly in
    transformer architectures.

    It's defined as:
    GELU(x) = x * Φ(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    where Φ(x) is the cumulative distribution function of a standard Gaussian.

    This implementation uses the tanh approximation which is computationally efficient while maintaining high accuracy.

    Reference:
        https://arxiv.org/abs/1606.08415 - Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the GELU activation function element-wise.

        Args:
            x (Tensor): Input tensor of any shape

        Returns:
            Tensor: Output tensor with the same shape as input, with GELU activation applied element-wise
        """
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
