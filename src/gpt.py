import torch
import torch.nn as nn

from transformer import TransformerBlock
from normalization import LayerNorm
from configurations import GPT_CONFIG_124M


class GptModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def generate_text_simple(self, idx, max_new_tokens, context_size):
        # idx is (B, T) array of indices in the current context
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
            # -1 is for keeping the dimensions correct for concatenation later
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx


    def generate_text_simple_softmax(self, idx, max_new_tokens, context_size):
        # idx is (batch, n_tokens) array of indices in the current context
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


import tiktoken
def main():

    torch.manual_seed(123)
    model = GptModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = model.generate_text_simple(
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M.context_length
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()






# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)
# print(batch)


### TEST
# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)

# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

# Should be:
# Input batch:
#  tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

# Output shape: torch.Size([2, 4, 50257])
# tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
#          [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
#          [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
#          [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

#         [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
#          [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
#          [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
#          [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
#        grad_fn=<UnsafeViewBackward0>)

### TEST
# total_params = sum(p.numel() for p in model.parameters())
# assert total_params == 163009536 


### TEST
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])

### TEST
# total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# Number of trainable parameters considering weight tying: 124,412,160


### Test
# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
# total_params = sum(p.numel() for p in model.parameters())
# total_size_bytes = total_params * 4
# Convert to megabytes
# total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")
# Total size of the model: 621.83 MB


"""
TESTS

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])

model.eval() # disable dropout
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
Output length: 10

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
Hello, I am Featureiman Byeswickattribute argue



"""