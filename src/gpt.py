import torch
import torch.nn as nn

from torch import Tensor

from .configurations import GptConfig
from .normalization import LayerNorm
from .transformer import TransformerBlock


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
        self.ptr_current_pos = 0  # Position tracking for cached attention

        # Final normalization and output head
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
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
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        # Combine token and position embeddings
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
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
        for blk in self.trf_blocks:
            blk: TransformerBlock
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


"""
For tests:

Run:
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)

Expect:
batch.shape == out.shape[0:2] == torch.Size([2, 4])
out.shape == torch.Size([2, 4, 50257])
out == tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
         [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
         [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
         [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

        [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
         [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
         [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
         [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
       grad_fn=<UnsafeViewBackward0>)

Run:
total_params = sum(p.numel() for p in model.parameters())

Expect:
total_params == 163,009,536

Run:
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())

Expect:
total_params_gpt2 == 124,412,160

Run:
# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4

# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)

Expect:
total_size_mb == 621.83


Run:
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
model.eval() # disable dropout

out = model.generate_text_simple(
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

Expect:
out == tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  
len(out[0]) == 10


Run:
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

Expect:
decoded_text == "Hello, I am Featureiman Byeswickattribute argue"


Implement another test based on the file chapters/ch04/01_main-chapter-code/tests.py
"""