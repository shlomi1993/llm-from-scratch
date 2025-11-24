"""
GPT Model Module

This module provides access to multiple GPT model implementations.

Classes:
    GptModelBasic: Standard GPT implementation without optimization features (from gpt_basic.py)
    GptModelCached: Enhanced GPT implementation with KV cache support (from gpt_cached.py)
    GptModel: Default alias for GptModelCached
"""

from .gpt_basic import GptModelBasic
from .gpt_cached import GptModelCached

# Default import alias points to cached version
GptModel = GptModelCached

__all__ = ["GptModelBasic", "GptModelCached", "GptModel"]
