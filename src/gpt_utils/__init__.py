"""
GPT Utilities Package

This package provides utility functions for downloading GPT-2 models, loading their weights into PyTorch models, and
training them.
"""

from .download import download_gpt2, FILES_TO_DOWNLOAD
from .load import load_weights_into_gpt
from .train import (
    calc_loss_batch,
    calc_loss_loader,
    train_test_split,
    evaluate_model,
    generate_and_print_sample,
    train_model,
    plot_losses
)

__all__ = [
    "download_gpt2",
    "FILES_TO_DOWNLOAD",
    "load_weights_into_gpt",
    "calc_loss_batch",
    "calc_loss_loader",
    "train_test_split",
    "evaluate_model",
    "generate_and_print_sample",
    "train_model",
    "plot_losses"
]
