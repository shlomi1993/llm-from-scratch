"""
Utils Package

This package contains utility functions and classes for device management, logging, tokenization, and visualization
"""

from .device import Device, get_device
from .ollama import OllamaEvaluator, format_input
from .tokenization import tokenizer
from .tokenization.tokenizer import TOKENIZER, encode, decode, text_to_token_ids, token_ids_to_text
from .visualization import plot_metrics


__all__ = [
    "Device",
    "get_device",
    "OllamaEvaluator",
    "format_input",
    "tokenizer",
    "TOKENIZER",
    "encode",
    "decode",
    "text_to_token_ids",
    "token_ids_to_text",
    "plot_metrics",
]
