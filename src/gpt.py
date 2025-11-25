"""
GPT model implementation.

This module defines the GPTModel class, which implements a GPT-style autoregressive language model with token and
position embeddings, multiple transformer blocks, and an output head for generating vocabulary logits. The model
supports both standard and cached generation modes for efficient text generation.
"""
import torch
import torch.nn as nn

from torch import Tensor

from .transformer import TransformerBlock
from .normalization import LayerNorm
from .configurations import GptConfig


class GptModel(nn.Module):

    def __init__(self, config: GptConfig) -> None:
        super().__init__()

        # Embedding layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)

        # Dropout layer
        self.drop_emb = nn.Dropout(config.drop_rate)

        # Transformer blocks
        self.trf_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.current_pos = 0

        # Final normalization and output head
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # Position ids
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        # Combine token and position embeddings
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)

        # Output logits
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0

    def generate_text_simple(self, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
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

    def generate_text_softmax(self, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
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

    def generate_text_simple_cached(self, idx: Tensor, max_new_tokens: int, context_size: int ,use_cache: bool =True):
        self.eval()
        ctx_len = context_size or self.pos_emb.num_embeddings

        with torch.no_grad():
            if use_cache:
                # Init cache with full prompt
                self.reset_kv_cache()
                logits = self(idx[:, -ctx_len:], use_cache=True)

                for _ in range(max_new_tokens):
                    # a) pick the token with the highest log-probability (greedy sampling)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    # b) append it to the running sequence
                    idx = torch.cat([idx, next_idx], dim=1)
                    # c) feed model only the new token
                    logits = self(next_idx, use_cache=True)
            else:
                for _ in range(max_new_tokens):
                    logits = self(idx[:, -ctx_len:], use_cache=False)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    idx = torch.cat([idx, next_idx], dim=1)

        return idx
