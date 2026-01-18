"""
Tokenization module for encoding and decoding text using the GPT-2 tokenizer.

This module provides a wrapper around the tiktoken GPT-2 tokenizer, exposing convenience methods for encoding text to
token IDs and decoding token IDs back to text.
"""

from .tokenizer import EOT_TOK, EOT_IDX, IGNORE_IDX, g_tokenizer


__all__ = [
    "EOT_TOK",
    "EOT_IDX",
    "IGNORE_IDX",
    "g_tokenizer",
]
