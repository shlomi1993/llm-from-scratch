"""
ch04_presentation.py

# Chapter 4: Implementing a GPT model from Scratch To Generate Text - Presentation Code

This is auto-generated code from the Jupyter notebook, modified to present sections 4.6 and 4.7 of Chapter 4.

Original file is located at
    https://colab.research.google.com/drive/1h0nmndcXRgmcUr6-KcasScKy_hFwMdUh

"""

import torch
import torch.nn as nn
import tiktoken

from importlib.metadata import version

print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))


########################################################################################################################
######################################### Relevant Code from Precious Sections #########################################
########################################################################################################################


# Multihead Attention class from chapter 3
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
        # this will result in errors in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


# Configuration for the GPT-2 124M parameter model from section 4.1
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


# Input batch example from section 4.1
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


# Normalization layer from section 4.2
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# Activation function from section 4.3
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# Feed-forward module from section 4.3
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# Transformer block from section 4.5
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


########################################################################################################################
######################################### Presentation of Sections 4.6 and 4.7 #########################################
########################################################################################################################

# Disable scientific notation for better readability during presentation
torch.set_printoptions(sci_mode=False)

# ============================================
# Section 4.6: The Complete GPT Model
# ============================================
# This class brings together all components: embeddings, transformer blocks, and output layer
# Architecture mirrors GPT-2 with 12 stacked transformer blocks for the 124M configuration

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token embedding: converts token IDs to 768-dimensional vectors
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional embedding: adds position information to token embeddings
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout applied to embeddings for regularization
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of transformer blocks (12 layers for GPT-2 124M)
        # Each block contains multi-head attention and feed-forward networks
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final layer normalization before output projection
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output head: projects back to vocabulary size (50,257 tokens)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # Extract batch size and sequence length from input
        batch_size, seq_len = in_idx.shape

        # Convert token IDs to embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Generate positional embeddings for the sequence
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine token and position information
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]

        # Apply dropout to combined embeddings
        x = self.drop_emb(x)

        # Process through all transformer blocks
        x = self.trf_blocks(x)

        # Apply final normalization
        x = self.final_norm(x)

        # Project to vocabulary size to get logits for next token prediction
        logits = self.out_head(x)
        return logits


# ============================================
# Model Instantiation
# ============================================
# Create a GPT model with random initial weights using the 124M parameter configuration
# Note: This model is untrained and will produce random outputs until trained

torch.manual_seed(123)  # Set seed for reproducibility
model = GPTModel(GPT_CONFIG_124M)


# ============================================
# Forward Pass Demo
# ============================================
# Pass a batch through the model to see the output shape
# Input: [batch_size=2, num_tokens=4] → Output: [batch_size=2, num_tokens=4, vocab_size=50257]

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


# ============================================
# Model Size Analysis: Parameter Count
# ============================================
# We'll verify the model's parameter count and understand the "124M vs 163M" discrepancy

# Count total parameters across all layers
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")


# ============================================
# Understanding Weight Tying
# ============================================
# Why does our model have 163M parameters instead of the advertised 124M?
# Answer: GPT-2 used "weight tying" - sharing weights between token embedding and output layers

# Both layers have the same shape: [vocab_size, emb_dim] = [50257, 768]
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)


# ============================================
# Calculating the "True" GPT-2 124M Count
# ============================================
# If we apply weight tying (reusing tok_emb weights for out_head), we subtract the output layer parameters
# This gives us the canonical 124M parameter count from the GPT-2 paper

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")


# ============================================
# Memory Footprint Estimation
# ============================================
# Estimate how much RAM/VRAM this model requires when loaded
# Useful for deployment and hardware planning

# Each parameter stored as float32 (4 bytes)
total_size_bytes = total_params * 4

# Convert bytes to megabytes for easier interpretation
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total size of the model: {total_size_mb:.2f} MB")


# ============================================
# Scaling Up: Other GPT-2 Configurations
# ============================================
# The architecture we built is flexible and can be scaled to larger models
# Simply adjust emb_dim, n_layers, and n_heads in the config dictionary
#
# GPT-2 Small (124M) - What we implemented:
#   - emb_dim: 768, n_layers: 12, n_heads: 12
#
# GPT-2 Medium (345M):
#   - emb_dim: 1024, n_layers: 24, n_heads: 16
#
# GPT-2 Large (762M):
#   - emb_dim: 1280, n_layers: 36, n_heads: 20
#
# GPT-2 XL (1.5B):
#   - emb_dim: 1600, n_layers: 48, n_heads: 25


# ============================================
# Section 4.7: Text Generation with Greedy Decoding
# ============================================
# Now that we have a complete GPT model, let's use it to generate text
# Generation is autoregressive: produce one token at a time, feed it back as input

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding - always pick the most likely next token.

    Args:
        model (nn.Module): The GPT model
        idx (torch.Tensor): Input token indices [batch_size, seq_len]
        max_new_tokens (int): How many tokens to generate
        context_size (int): Maximum context length the model supports

    Returns:
        torch.Tensor: Extended sequence with generated tokens appended
    """

    # Generate tokens one at a time in a loop
    for _ in range(max_new_tokens):

        # Step 1: Context Window Management
        # If our sequence exceeds the model's max context (e.g., 1024 tokens), crop it to keep only the most recent tokens
        idx_cond = idx[:, -context_size:]

        # Step 2: Get Model Predictions
        # Run the current context through the model, while disabling gradient computation since we're not training
        with torch.no_grad():
            logits = model(idx_cond)

        # Step 3: Focus on the Next Token
        # Extract predictions for only the last position in the sequence
        logits = logits[:, -1, :]  # Shape: [batch_size, seq_len, vocab_size] → [batch_size, vocab_size]

        # Step 4: Convert Logits to Probabilities
        # Apply softmax to get a probability distribution over the vocabulary
        probas = torch.softmax(logits, dim=-1)  # Shape: [batch_size, vocab_size]

        # Step 5: Greedy Selection
        # Pick the token with the highest probability (greedy decoding)
        # Alternative strategies: sampling, top-k, nucleus sampling (covered later)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # Shape: [batch_size, 1]

        # Step 6: Append to Sequence
        # Add the newly generated token to our running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Shape: [batch_size, seq_len+1]

    return idx


# ============================================
# Demo: Text Generation in Action
# ============================================
# Let's see the generation process with a concrete example

# Step 1: Prepare Input Text
# Start with a prompt that the model will continue
start_context = "Hello, I am"

# Step 2: Tokenize the Input
# Convert text to token IDs using the GPT-2 tokenizer, and add batch dimension: [seq_len] → [1, seq_len]
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# Step 3: Set Model to Evaluation Mode
# Disable dropout to get deterministic outputs during inference
model.eval()

# Step 4: Generate New Tokens
# Generate 6 additional tokens following the input context
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# Step 5: Decode Back to Text
# Convert token IDs back to human-readable text after removing the batch dimension before decoding
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# Important Note:
# The output will be gibberish because the model has random weights
# In Chapter 5, we'll train this model to produce coherent text
# For now, this demonstrates the generation mechanics
