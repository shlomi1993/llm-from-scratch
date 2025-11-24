"""
GPT Basic Model Implementation

This module implements the GPT (Generative Pre-trained Transformer) model architecture similar to the OpenAI GPT-2 model.
It includes a standard implementation without optimization features. The model features token and positional embeddings,
transformer blocks, layer normalization, and text generation capabilities using both greedy and softmax-based sampling
strategies.
"""

import torch
import torch.nn as nn

from torch import Tensor

from ..transformer import TransformerBlock
from ..normalization import LayerNorm
from ..configurations import GptConfig


class GptModelBasic(nn.Module):
    """
    GPT Model implementation based on the Transformer architecture.

    This class implements a decoder-only transformer model similar to GPT-2, featuring token and positional embeddings,
    multiple transformer blocks, layer normalization, and a language modeling head.

    Args:
        config (GptConfig): Configuration object containing model hyperparameters including embedding dimension, number
            of layers, heads, vocabulary size, context length, and dropout rate.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the GPT model with the given configuration.

        Args:
            config (GptConfig): Model configuration containing hyperparameters
        """
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx: Tensor) -> Tensor:
        """
        Forward pass through the GPT model.

        Args:
            in_idx (Tensor): Input token indices of shape [batch_size, seq_len]

        Returns:
            Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def _generate_text_simple(self, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
        """
        Generate text using greedy sampling (argmax) strategy.

        This method generates new tokens by always selecting the token with the highest logit value at each step. The
        context is cropped if it exceeds the specified context size to maintain computational efficiency.

        Args:
            idx (Tensor): Initial context tokens of shape [batch_size, seq_len]
            max_new_tokens (int): Maximum number of new tokens to generate
            context_size (int): Maximum context length to consider for generation

        Returns:
            Tensor: Extended sequence with generated tokens of shape
                         [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            # Note that the -1 is for keeping the dimensions correct for concatenation later
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx

    def _generate_text_softmax(self, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
        """
        Generate text using softmax-based sampling strategy.

        This method generates new tokens by first applying softmax to convert logits to probabilities, then selecting
        the token with the highest probability. While this method produces the same results as generate_text_simple for
        greedy sampling, it demonstrates the softmax step explicitly.

        Args:
            idx (Tensor): Initial context tokens of shape [batch_size, seq_len]
            max_new_tokens (int): Maximum number of new tokens to generate
            context_size (int): Maximum context length to consider for generation

        Returns:
            Tensor: Extended sequence with generated tokens of shape [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx

    def generate_text(self, idx: Tensor, max_new_tokens: int, context_size: int, use_softmax: bool = False) -> Tensor:
        """
        Generate text using either greedy or softmax-based sampling strategy.

        This method allows the user to choose between two text generation strategies: greedy sampling (argmax) or
        softmax-based sampling. The context is cropped if it exceeds the specified context size to maintain computational
        efficiency.

        Args:
            idx (Tensor): Initial context tokens of shape [batch_size, seq_len]
            max_new_tokens (int): Maximum number of new tokens to generate
            context_size (int): Maximum context length to consider for generation
            use_softmax (bool): If True, use softmax-based sampling; otherwise, use greedy sampling (default: False)

        Returns:
            Tensor: Extended sequence with generated tokens of shape [batch_size, seq_len + max_new_tokens]
        """
        if use_softmax == False:
            return self._generate_text_simple(idx, max_new_tokens, context_size)
        else:
            return self._generate_text_softmax(idx, max_new_tokens, context_size)
