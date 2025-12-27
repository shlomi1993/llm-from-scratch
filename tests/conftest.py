import os
import pytest
import sys
import tiktoken
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.config import GptConfig
from src.gpt import GptModel
from src.utils import get_device


# Add the root directory to Python path for tools package
root_path = os.path.join(os.path.dirname(__file__), '..')
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return get_device()


@pytest.fixture(scope="session")
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="session")
def the_verdict_dataset() -> str:
    with open("datasets/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

@pytest.fixture
def dummy_loader():
    batch_size = 2
    seq_len = 3
    n_batches = 2
    x = torch.randint(0, 10, (n_batches * batch_size, seq_len))
    y = x.clone()
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def sample_config() -> GptConfig:
    return GptConfig(
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        context_length=32,
        drop_rate=0.1,
        qkv_bias=False
    )


@pytest.fixture
def sample_model(sample_config: GptConfig) -> GptModel:
    torch.manual_seed(42)
    return GptModel(sample_config)
