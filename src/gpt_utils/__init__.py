"""
GPT Utilities Package

This package provides utility functions for downloading GPT-2 models, loading their weights into PyTorch models, and
training them.
"""

from .download import download_gpt2, FILES_TO_DOWNLOAD
from .generate import load_weights_into_gpt, run_model_generation_flow, run_model_interactive_flow
from .train import (
    calc_loss_batch,
    calc_loss_loader,
    train_test_split,
    evaluate_model,
    generate_and_print_sample,
    train_model,
    plot_losses,
    run_model_training_flow,
)

__all__ = [
    # Download flow
    "download_gpt2",
    "FILES_TO_DOWNLOAD",

    # Training flow
    "calc_loss_batch",
    "calc_loss_loader",
    "train_test_split",
    "evaluate_model",
    "generate_and_print_sample",
    "train_model",
    "plot_losses",
    "run_model_training_flow",

    # Generation flow
    "load_weights_into_gpt",
    "run_model_generation_flow",
    "run_model_interactive_flow",
]
