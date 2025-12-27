import os
import pytest
import sys
import tiktoken
import torch

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
