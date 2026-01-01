import torch
import torch.nn as nn

from torch import Tensor

from .activation import GELU
from .config import GptConfig


class FeedForward(nn.Module):

    def __init__(self, config: GptConfig) -> None:

        super().__init__()
        emb_dim = config.emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  # Expand by factor of 4
            GELU(),                           # Non-linear activation
            nn.Linear(4 * emb_dim, emb_dim),  # Project back to original dimension
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        return self.layers(x)


