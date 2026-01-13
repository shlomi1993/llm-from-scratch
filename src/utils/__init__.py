"""
Utils Package

This package contains utility functions and classes for device management, logging, tokenization, and visualization
"""

from .checkpoint import load_model, save_model
from .device import Device, get_device
from .logger import g_logger
from .losses import LossFunc, calc_loss_batch, calc_loss_last_token, calc_loss_loader, calc_losses
from .ollama import OllamaEvaluator, format_input, coding_format_input
from .tokenization import tokenizer
from .tokenization.tokenizer import TOKENIZER, EOT, PAD_IDX, IGNORE_IDX, encode, decode, text_to_token_ids, token_ids_to_text
from .visualization import plot_metrics


__all__ = [
    "load_model",
    "save_model",
    "Device",
    "get_device",
    "g_logger",
    "LossFunc",
    "calc_loss_batch",
    "calc_loss_last_token",
    "calc_loss_loader",
    "calc_losses",
    "OllamaEvaluator",
    "format_input",
    "coding_format_input",
    "tokenizer",
    "TOKENIZER",
    "EOT",
    "PAD_IDX",
    "IGNORE_IDX",
    "encode",
    "decode",
    "text_to_token_ids",
    "token_ids_to_text",
    "plot_metrics",
]
