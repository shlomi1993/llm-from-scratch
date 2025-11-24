import pytest
import torch

from configurations import GptConfig


@pytest.fixture
def sample_inputs():
    """
    Sample input tensor representing word embeddings for testing.
    """
    return torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )


@pytest.fixture
def sample_batch_inputs():
    """
    Sample batch input tensor for testing with batch dimension.
    """
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    return torch.stack((inputs, inputs), dim=0)


@pytest.fixture
def sample_config():
    """
    Sample configuration for testing attention modules.
    """
    return GptConfig(emb_dim=64, n_layers=2, n_heads=4)


@pytest.fixture
def sample_configs():
    """
    Multiple sample configurations for testing.
    """
    return [
        GptConfig(emb_dim=64, n_layers=2, n_heads=2, vocab_size=1000, context_length=32, drop_rate=0.0, qkv_bias=False),
        GptConfig(emb_dim=128, n_layers=4, n_heads=4, vocab_size=2000, context_length=64, drop_rate=0.0, qkv_bias=False)
    ]
