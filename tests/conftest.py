import os
import pytest
import sys
import torch

from pathlib import Path
from shutil import get_terminal_size


# Add the root directory to Python path for tools package
root_path = os.path.join(os.path.dirname(__file__), '..')
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def print_test_title(request: pytest.FixtureRequest) -> None:
    sep = "=" * get_terminal_size().columns
    print(f"\n\n\033[94m{sep}\nRunning test: {request.node.name}\n{sep}\n\033[0m")


@pytest.fixture(autouse=True)
def set_torch_seed():
    torch.manual_seed(123)


@pytest.fixture
def chapters_path(tmp_path: Path) -> Path:
    """
    Returns the absolute path to the chapters directory, required for locating chapter scripts.
    """
    chapters_dir = Path(__file__).parent.parent / "chapters"
    models_dir = Path(__file__).parent.parent / "models"
    symlink_path = tmp_path / "gpt2"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(models_dir, target_is_directory=True)
    return chapters_dir
