import torch
import torch.nn as nn

from torch import Tensor

from .config import GptConfig
from .normalization import LayerNorm
from .transformer import TransformerBlock
from src.utils.tokenization.tokenizer import g_tokenizer


class GptModel(nn.Module):
    """
    GPT2 language model architecture implementation.
    Based on the architecture described in "Language Models are Few-Shot Learners" (Brown et al., 2020).
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize an instance of a GPT2-based model to train

        Args:
            config (GptConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self._config = config

        # Embedding layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)

        # Dropout layer
        self.drop_emb = nn.Dropout(config.drop_rate)

        # Transformer blocks
        self.trf_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ptr_current_pos = 0  # Position tracking for cached attention

        # Final normalization and output head
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    @property
    def config(self) -> GptConfig:
        """
        Get the model configuration while preventing modifications (the instance is frozen).
        """
        return self._config

    def forward(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass of the GPT model.

        Args:
            in_idx (Tensor): Input token indices of shape (batch_size, sequence_length).
            use_cache (bool, optional): Whether to use cached key-value pairs for attention. Defaults to False.

        Returns:
            Tensor: Output logits of shape (batch_size, sequence_length, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # Position ids
        if use_cache:
            if self.ptr_current_pos + seq_len > self.pos_emb.num_embeddings:
                raise ValueError(f"Position embedding overflow. Tried to read {self.ptr_current_pos + seq_len} which exceeded size of {self.pos_emb.num_embeddings}")

            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds: Tensor = self.pos_emb(pos_ids)

        # Combine token and position embeddings
        x = tok_embeds + pos_embeds.unsqueeze(0)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # Pass through transformer blocks
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        # Final normalization
        x = self.final_norm(x)

        # Output logits
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        """
        Reset the key-value cache for all transformer blocks.
        This is typically called at the beginning of a new generation sequence.
        """
        for blk in self.trf_blocks:
            blk: TransformerBlock
            blk.att.reset_cache()
        self.ptr_current_pos = 0

    def generate_naive(self, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
        """
        Generate new tokens using a naive approach without caching.

        Args:
            idx (Tensor): Input token indices of shape (batch_size, sequence_length).
            max_new_tokens (int): Number of new tokens to generate.
            context_size (int): Maximum context size for generation.

        Returns:
            Tensor: Generated token indices of shape (batch_size, sequence_length + max_new_tokens).
        """
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size (e.g., if LLM supports only 5 tokens, and
            # the context size is 10), then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Optional: Apply softmax to get probabilities
            # logits = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest logits value
            # Note that the -1 is for keeping the dimensions correct for concatenation later
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, next_idx), dim=1)  # (batch, n_tokens + 1)

        return idx

    def generate_cached(self, idx: Tensor, max_new_tokens: int, context_size: int, use_cache: bool = True):
        """
        Generate new tokens using cached key-value pairs for attention.

        Args:
            idx (Tensor): Input token indices of shape (batch_size, sequence_length).
            max_new_tokens (int): Number of new tokens to generate.
            context_size (int): Maximum context size for generation.
            use_cache (bool, optional): Whether to use cached key-value pairs for attention. Defaults to True.
        """
        self.eval()
        ctx_len = context_size or self.pos_emb.num_embeddings
        with torch.no_grad():
            if use_cache:
                self.reset_kv_cache()  # Init cache with full prompt
                logits: Tensor = self(idx[:, -ctx_len:], use_cache)
                for _ in range(max_new_tokens):
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)  # a) pick the token with the highest log-probability (greedy sampling)
                    idx = torch.cat([idx, next_idx], dim=1)                # b) append it to the running sequence
                    logits = self(next_idx, use_cache)                     # c) feed model only the new token
            else:
                for _ in range(max_new_tokens):
                    logits = self(idx[:, -ctx_len:], use_cache)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def generate(self, idx: Tensor, max_new_tokens: int, context_size: int, temperature: float = 0.0, top_k: int = None,
                 eos_id: int = None, live: bool = False) -> Tensor:
        """
        Generate new tokens using sampling strategies such as temperature scaling and top-k sampling.

        Args:
            idx (Tensor): Input token indices of shape (batch_size, sequence_length).
            max_new_tokens (int): Number of new tokens to generate.
            context_size (int): Maximum context size for generation.
            temperature (float, optional): Temperature for scaling logits. Defaults to 0.0 (greedy sampling).
            top_k (int, optional): If specified, only consider the top_k logits for sampling. Defaults to None.
            eos_id (int, optional): If specified, generation stops when this token ID is generated. Defaults to None.
            live (bool, optional): If True, prints generated tokens in real-time. Defaults to False.
        """
        for _ in range(max_new_tokens):

            # Trim context
            idx_cond = idx[:, -context_size:]

            # Forward pass
            with torch.no_grad():
                logits: Tensor = self(idx_cond)

            # Get logits for the last time step
            logits = logits[:, -1, :]

            # Apply top-k
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            # Apply temperature or greedy selection
            if temperature > 0.0:
                logits = logits / temperature
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if EOS token is generated
            if eos_id is not None and (idx_next == eos_id or idx_next.item() == eos_id):
                break

            # Live generation print
            if live:
                token_str = g_tokenizer.decode([idx_next.item()])
                print(token_str, end="", flush=True)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
