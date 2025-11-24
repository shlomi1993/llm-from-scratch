"""
GPT KV-Cached Model Implementation

This module implements the cached version of the GPT (Generative Pre-trained Transformer) model with KV caching for
improved inference efficiency. The model features token and positional embeddings, optimized transformer blocks, layer
normalization, and efficient text generation capabilities using KV cache optimization.
"""

import torch
import torch.nn as nn

from torch import Tensor

from ..configurations import GptConfig
from ..normalization import LayerNorm
from ..transformer import TransformerBlockCached


class GptModelCached(nn.Module):
    """
    GPT Model implementation with KV cache support for efficient inference.

    This class implements an enhanced version of the GPT-2 decoder-only transformer model that includes key-value
    caching optimizations for improved inference performance. The model maintains cached key and value tensors across
    generation steps, significantly reducing computational overhead during autoregressive text generation.

    Features:
        - KV cache support for efficient incremental inference
        - Sliding window cache with configurable window size
        - Optimized positional embedding handling for cached inference
        - Memory-efficient cache overflow management
        - Compatible with both cached and non-cached inference modes

    Args:
        config (GptConfig): Configuration object containing model hyperparameters.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the GPT model with KV cache support.

        Args:
            config (GptConfig): Model configuration containing hyperparameters
        """
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        # Use ModuleList instead of Sequential to support caching
        self.trf_blocks = nn.ModuleList([TransformerBlockCached(config) for _ in range(config.n_layers)])

        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # Position tracking for KV cache
        self.ptr_current_pos = 0

    def forward(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass through the GPT model with optional KV caching.

        Args:
            in_idx (Tensor): Input token indices of shape [batch_size, seq_len]
            use_cache (bool, optional): Whether to use KV caching for efficient inference. Defaults to False.

        Returns:
            Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # Handle positional embeddings for caching
        if use_cache:
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)

        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # Pass through transformer blocks with caching support
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        """
        Reset the KV cache for all transformer blocks and position tracking.

        This method should be called before starting a new generation sequence to ensure clean cache state.
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0

    def generate_text(self, idx: Tensor, max_new_tokens: int, context_size: int = None) -> Tensor:
        """
        Generate text using KV cache optimization for efficient inference.

        This method uses cached key-value pairs to significantly speed up autoregressive generation. After processing
        the initial context, only new tokens need full computation while leveraging previously cached attention states.

        Args:
            idx (Tensor): Initial context tokens of shape [batch_size, seq_len]
            max_new_tokens (int): Maximum number of new tokens to generate
            context_size (int, optional): Maximum context length. Defaults to model's context length.

        Returns:
            Tensor: Extended sequence with generated tokens of shape [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        ctx_len = context_size or self.config.context_length

        with torch.no_grad():
            # Initialize cache with full prompt
            self.reset_kv_cache()
            logits = self(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # Pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)

                # Append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)

                # Feed model only the new token (leveraging cache)
                logits = self(next_idx, use_cache=True)

        return idx
