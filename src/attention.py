import torch

from torch import nn, Tensor

from .configurations import GptConfig


class SelfAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    @staticmethod
    def from_config(config: GptConfig) -> 'SelfAttention':
        return SelfAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

"""
For tests

Global:
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

Run:
torch.manual_seed(123)
sa = SelfAttention(d_in, d_out)
print(sa(inputs))

Expect:
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)

Run:
torch.manual_seed(789)
sa = SelfAttention(d_in, d_out)
print(sa(inputs))

Expect:
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
"""

"""
Causal self-attention mechanism implementation with masking and dropout.
"""

class CausalAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def from_config(config: GptConfig) -> 'CausalAttention':
        return CausalAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose

        # New, _ ops are in-place
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec


"""
For tests

Global:
batch = torch.stack((inputs, inputs), dim=0)

Run:
torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

Expect:
context_vecs == tensor([[[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]],

        [[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)
context_vecs.shape == torch.Size([2, 6, 2])
"""

class MultiheadAttentionWrapper(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        attention_heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(n_heads)]
        self.heads = nn.ModuleList(attention_heads)

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionWrapper':
        return MultiheadAttentionWrapper(
            d_in=config.emb_dim,
            d_out=config.emb_dim // config.n_heads,
            context_length=config.context_length,
            dropout=config.drop_rate,
            n_heads=config.n_heads,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


"""
For tests:

Run:
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, n_heads=2)
context_vecs = mha(batch)

Expect:
context_vecs == tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
context_vecs.shape == torch.Size([2, 6, 4])

Run:
torch.manual_seed(123)

context_length = max_length
d_in = output_dim

num_heads=2
d_out = d_in // num_heads

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)

batch = input_embeddings
context_vecs = mha(batch)

Expect:
context_vecs.shape == torch.Size([8, 4, 256])
"""


"""
Efficient Multi-Head Attention implementation with unified QKV projections.
"""

class MultiheadAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionCached':
        return MultiheadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            n_heads=config.n_heads,
            qkv_bias=config.qkv_bias
        )

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `n_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, n_heads, head_dim)
        keys = keys.view(b, num_tokens, self.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)

        # Transpose: (b, num_tokens, n_heads, head_dim) -> (b, n_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply scaled softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


"""
For tests:

Run:
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, n_heads=2)
context_vecs = mha(batch)

Expect:
context_vecs = tensor([[[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]],

        [[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
context_vecs.shape == torch.Size([2, 6, 2])

Run:
torch.manual_seed(123)

context_length = max_length
d_in = output_dim
d_out = d_in

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

batch = input_embeddings
context_vecs = mha(batch)

Expect:
context_vecs.shape == torch.Size([8, 4, 256])

"""

"""
More tests:

Run:
torch.manual_seed(123)

d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

Expect:
context_vec == tensor([[[-9.1476e-02,  3.4164e-02],
         [-2.6796e-01, -1.3427e-03],
         [-4.8421e-01, -4.8909e-02],
         [-6.4808e-01, -1.0625e-01],
         [-8.8380e-01, -1.7140e-01],
         [-1.4744e+00, -3.4327e-01]],

        [[-9.1476e-02,  3.4164e-02],
         [-2.6796e-01, -1.3427e-03],
         [-4.8421e-01, -4.8909e-02],
         [-6.4808e-01, -1.0625e-01],
         [-8.8380e-01, -1.7140e-01],
         [-1.4744e+00, -3.4327e-01]]], grad_fn=<CatBackward0>)
context_vecs.shape == torch.Size([2, 6, 2])

Run:
context_length = 1024
d_in, d_out = 768, 768
num_heads = 12

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

Expect:
sum(p.numel() for p in mha.parameters() if p.requires_grad) == 2360064

"""


"""
Multi-Head Attention with KV cache support.
"""


class MultiheadAttentionCached(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, n_heads: int, qkv_bias: bool = False,
                 max_seq_len: int = None, window_size: int = None) -> None:

        super().__init__()
        assert d_out % n_heads == 0 == 0, "emb_dim must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        # Linear projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        # KV cache parameters
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    @staticmethod
    def from_config(config: GptConfig) -> 'MultiheadAttentionCached':
        return MultiheadAttentionCached(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            n_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
            window_size=config.kv_window_size or config.context_length
        )

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        batch_size, n_tokens, d_in = x.shape

        # Compute Q, K, V
        keys_new = self.W_key(x)  # Shape: (batch_size, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # Reshape for multi-head attention
        keys_new = keys_new.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        values_new = values_new.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        # Transpose for attention computation
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Handle KV caching
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != batch_size:
                self.cache_k = torch.zeros(batch_size, self.n_heads, self.window_size, self.head_dim, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # pointer to next free slot

            # Handle cache overflow
            if self.ptr_cur + n_tokens > self.window_size:
                overflow = self.ptr_cur + n_tokens - self.window_size
                # Shift cache left
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow

            # Update cache
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + n_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + n_tokens, :] = values_new
            self.ptr_cur += n_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        K = attn_scores.size(-1)
        if n_tokens == K:
            # No cache - use triangular mask
            causal_mask = torch.triu(torch.ones(n_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # With cache - offset diagonal
            offset = K - n_tokens
            row_idx = torch.arange(n_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx

        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        # Apply attention
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        context_vec = (attn_weights @ values).transpose(1, 2)  # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(batch_size, n_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
