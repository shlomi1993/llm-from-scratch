
"""
Feed-Forward Network

This module implements the position-wise feed-forward network used in transformer architectures. The feed-forward
network applies two linear transformations with a GELU activation function in between, following the standard
transformer design.

The network expands the input dimension by a factor of 4 (as is standard in transformer architectures), applies GELU
activation, and then projects back to the original dimension. This provides the non-linear transformation capacity
within each transformer layer.
"""

import torch
import torch.nn as nn

from torch import Tensor

from .activations import GELU
from .configurations import GptConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network for Transformer architectures.

    This class implements the feed-forward network used in each transformer layer, which consists of two linear
    transformations with a GELU activation function.

    The network follows the standard transformer design:

    FFN(x) = GELU(x * W1 + b1) * W2 + b2

    where the first linear layer expands the dimension by a factor of 4, and the second layer projects back to the
    original dimension.

    Args:
        config (GptConfig): Configuration containing model parameters.

    Attributes:
        layers (nn.Sequential): Sequential container with linear layers and GELU activation
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the feed-forward network.

        Args:
            config (GptConfig): Configuration with embedding dimension
        """
        super().__init__()
        emb_dim = config.emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  # Expand by factor of 4
            GELU(),                           # Non-linear activation
            nn.Linear(4 * emb_dim, emb_dim),  # Project back to original dimension
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to the input.

        Args:
            x (Tensor): Input tensor of shape [..., emb_dim] where ... represents any number of batch dimensions

        Returns:
            Tensor: Output tensor with the same shape as input

        Note:
            The feed-forward network is applied to the last dimension (embedding dimension) while preserving all other
            dimensions. This allows it to work with inputs of shape [batch_size, seq_len, emb_dim] as commonly used in
            transformers.
        """
        return self.layers(x)
