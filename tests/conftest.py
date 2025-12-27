import os
import pytest
import sys
import tiktoken
import torch


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
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="session")
def the_verdict_dataset() -> str:
    with open("datasets/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text
