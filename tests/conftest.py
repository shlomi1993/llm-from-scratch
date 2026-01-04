import os
import pytest
import sys
import tiktoken
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.download import FILES_TO_DOWNLOAD, download_gpt2
from src.utils.device import get_device


# Add the root directory to Python path for tools package
root_path = os.path.join(os.path.dirname(__file__), '..')
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def set_torch_seed():
    torch.manual_seed(123)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return get_device()


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
    torch.manual_seed(123)
    return GptModel(sample_config)


@pytest.fixture(scope="session")
def ref_model_dir() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_dir = os.path.join(base_dir, "reference_gpt2_models", "124M")

    needs_download = False
    if not os.path.isdir(model_dir):
        needs_download = True
    else:
        for fname in FILES_TO_DOWNLOAD:
            fpath = os.path.join(model_dir, fname)
            if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                needs_download = True
                break

    if needs_download:
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2("124M", model_dir)

    return model_dir


@pytest.fixture
def pretrained_model() -> dict:
    gpt_config_124m_256ctx = GptConfig(
        vocab_size=50257,    # Vocabulary size
        context_length=256,  # Shortened context length (orig: 1024)
        emb_dim=768,         # Embedding dimension
        n_heads=12,          # Number of attention heads
        n_layers=12,         # Number of layers
        drop_rate=0.1,       # Dropout rate
        qkv_bias=False       # Query-key-value bias
    )
    torch.manual_seed(123)
    model = GptModel(gpt_config_124m_256ctx)
    model.eval();  # Disable dropout during inference
    return model
