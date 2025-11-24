import math
import torch
import torch.nn as nn
import torch.nn.attention.flex_attention

from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .configurations import GptConfig


def causal(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """
    Causal mask function for flex attention.

    This function defines the causal masking pattern for autoregressive attention,
    where each query position can only attend to key-value positions at or before
    its own position in the sequence.

    Args:
        b (int): Batch index (not used in causal masking)
        h (int): Head index (not used in causal masking)
        q_idx (int): Query position index
        kv_idx (int): Key-Value position index

    Returns:
        bool: True if the query can attend to the key-value position, False otherwise
    """
    return q_idx >= kv_idx


class SelfAttention(nn.Module):
    """
    Self-attention mechanism implementation for transformer models.

    This module implements the scaled dot-product self-attention mechanism, which allows each position in a sequence to
    attend to all positions in the same sequence. It computes attention weights based on the similarity between query
    and key vectors, then uses these weights to create a weighted combination of value vectors.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """
        Initialize the SelfAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the self-attention mechanism.

        Computes self-attention by:
        1. Projecting input to query, key, and value vectors
        2. Computing attention scores as query-key dot products
        3. Applying scaled softmax to get attention weights
        4. Computing weighted sum of values using attention weights

        Args:
            x (Tensor): Input tensor of shape (seq_len, d_in)

        Returns:
            Tensor: Context vectors of shape (seq_len, d_out)
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention mechanism implementation for transformer models.

    This module implements the scaled dot-product self-attention mechanism with causal masking, which prevents positions
    from attending to future positions in the sequence. This is essential for autoregressive language models where each
    position should only have access to previous tokens. The module also includes dropout regularization to prevent
    overfitting.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False) -> None:
        """
        Initialize the CausalAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the causal self-attention mechanism.

        Computes causal self-attention by:
        1. Projecting input to query, key, and value vectors
        2. Computing attention scores as query-key dot products
        3. Applying causal mask to prevent attention to future positions
        4. Applying scaled softmax to get attention weights
        5. Applying dropout regularization to attention weights
        6. Computing weighted sum of values using attention weights

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Context vectors of shape (batch_size, num_tokens, d_out)

        Note:
            For inputs where `num_tokens` exceeds `context_length`, this will result in errors in the mask creation.
            In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs do not exceed
            `context_length` before reaching this forward method.
        """
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose

        # New, _ ops are in-place
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    Multi-Head Attention wrapper for parallel attention computation.

    This module implements multi-head attention by running multiple CausalAttention heads in parallel and concatenating
    their outputs. Each head learns different representation subspaces, allowing the model to jointly attend to
    information from different representation subspaces at different positions.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttentionWrapper module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension per head
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
        """
        super().__init__()
        attention_heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        self.heads = nn.ModuleList(attention_heads)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Computes multi-head attention by:
        1. Running each attention head on the input in parallel
        2. Concatenating all head outputs along the feature dimension

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Concatenated attention outputs of shape (batch_size, num_tokens, d_out * num_heads)
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention implementation with unified QKV projections.

    This module implements multi-head attention using a more efficient approach where all heads are computed
    simultaneously through tensor reshaping and transposition operations. Unlike the wrapper approach, this
    implementation uses unified linear projections for all heads and then splits the computation, making it more memory
    and computationally efficient. The module includes causal masking for autoregressive language modeling and an output
    projection layer for additional representational capacity.

    The key difference from MultiHeadAttentionWrapper is that this implementation:
    1. Projects to the full d_out dimension and then splits into heads
    2. Computes all heads simultaneously using tensor operations
    3. Includes an output projection layer for learned combination of heads
    4. Is more memory efficient and faster for large models
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the efficient multi-head attention mechanism.

        Computes multi-head attention by:
        1. Projecting input to unified query, key, and value tensors
        2. Reshaping and transposing to separate heads: (batch, tokens, heads, head_dim)
        3. Computing scaled dot-product attention for all heads simultaneously
        4. Applying causal mask to prevent attention to future positions
        5. Applying dropout regularization to attention weights
        6. Combining attention outputs and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, this will result in errors
            in the mask creation. In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs do
            not exceed `context_length` before reaching this forward method.
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
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

        # Apply scaled softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


class MultiHeadAttentionCombinedQKV(nn.Module):
    """
    Efficient Multi-Head Attention implementation with combined QKV projections.

    This module implements multi-head attention using a highly optimized approach where query, key, and value
    projections are computed in a single linear transformation. This reduces the number of matrix multiplications and
    improves computational efficiency compared to separate QKV projections. The implementation uses advanced tensor
    reshaping and permutation operations to efficiently separate and process multiple attention heads simultaneously.

    Key features:
    1. Single linear layer computes all QKV projections at once (3 * d_out dimensions)
    2. Advanced tensor permutations for efficient head separation
    3. Causal masking for autoregressive language modeling
    4. Output projection layer for learned combination of heads
    5. Dropout regularization for attention weights
    6. Memory and computationally efficient for large-scale models
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttentionCombinedQKV module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the combined QKV multi-head attention mechanism.

        Computes multi-head attention using combined QKV projection by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Computing scaled dot-product attention for all heads simultaneously
        4. Applying causal mask to prevent attention to future positions
        5. Applying dropout regularization to attention weights
        6. Combining attention outputs and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            For inputs where `num_tokens` exceeds `context_length`, this will result in errors in the mask creation.
            This implementation uses advanced tensor operations for maximum efficiency, including permute and unbind
            operations for optimal memory layout and computation.
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec


class MHAEinsum(nn.Module):
    """
    Multi-Head Attention implementation using Einstein summation (einsum) operations.

    This module implements multi-head attention using einsum operations for tensor contractions, providing an
    alternative implementation that demonstrates the mathematical operations more explicitly. The einsum notation makes
    the tensor dimension manipulations clearer and can be more readable for understanding the underlying mathematics of attention mechanisms.

    Key features:
    1. Uses einsum operations for explicit tensor contractions
    2. Manual parameter initialization with Kaiming uniform for better training dynamics
    3. Supports optional biases for QKV projections
    4. Causal masking for autoregressive language modeling
    5. Output projection layer for learned combination of heads
    6. Dropout regularization for attention weights
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAEinsum module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            context_length (int): Maximum sequence length for the causal mask
            dropout (float): Dropout probability for attention weights
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): Whether to include bias in QKV projections. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform initialization.

        This method initializes the query, key, and value weight matrices using Kaiming uniform initialization, which is
        suitable for layers with ReLU-like activations. The bias terms (if used) are initialized uniformly within a
        bound derived from the fan-in of the weights.
        """
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the einsum-based multi-head attention mechanism.

        Computes multi-head attention using einsum operations by:
        1. Computing Q, K, V projections using einsum for explicit tensor contractions
        2. Adding optional biases to the projections
        3. Reshaping and transposing to separate heads
        4. Computing scaled dot-product attention using einsum operations
        5. Applying causal mask to prevent attention to future positions
        6. Applying dropout regularization to attention weights
        7. Aggregating context vectors using einsum operations
        8. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            The einsum operations use the following notation:
            - 'bnd,do->bno': batch matrix multiplication for linear projections
            - 'bhnd,bhmd->bhnm': scaled dot-product attention computation
            - 'bhnm,bhmd->bhnd': context vector aggregation
            where b=batch, n=sequence_length, d=feature_dim, h=heads, m=sequence_length, o=output_dim
        """
        b, n, _ = x.shape

        # Calculate Q, K, V using einsum, first perform linear transformations
        Q = torch.einsum("bnd,do->bno", x, self.W_query)
        K = torch.einsum("bnd,do->bno", x, self.W_key)
        V = torch.einsum("bnd,do->bno", x, self.W_value)

        # Add biases if they are used
        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        # Reshape for multi-head attention
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)

        # Apply mask
        mask = self.mask[:n, :n]
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate the attended context vectors
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # Combine heads and project the output
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's built-in scaled_dot_product_attention.

    This module implements multi-head attention by leveraging PyTorch's optimized
    nn.functional.scaled_dot_product_attention function, which provides hardware-accelerated attention computation with
    automatic optimization for different backends (FlashAttention, memory-efficient attention, etc.). This
    implementation offers the best performance and is the recommended approach for production use.

    Key features:
    1. Uses PyTorch's optimized scaled_dot_product_attention for maximum performance
    2. Automatic backend selection (FlashAttention, memory-efficient, etc.)
    3. Built-in causal masking support
    4. Efficient memory usage and computation
    5. Combined QKV projection for reduced memory bandwidth
    6. Output projection layer for learned combination of heads
    7. Training-aware dropout handling
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchScaledDotProduct module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length (for compatibility, not directly used)
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PyTorch scaled dot-product attention mechanism.

        Computes multi-head attention using PyTorch's optimized scaled_dot_product_attention by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Using PyTorch's scaled_dot_product_attention with causal masking
        4. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            This implementation leverages PyTorch's optimized attention kernels which automatically select the best
            backend (FlashAttention, memory-efficient attention, etc.) based on the input shapes and available hardware.
            The is_causal=True parameter enables automatic causal masking without requiring explicit mask creation.
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # Use Dropout only during training
        use_dropout = 0. if not self.training else self.dropout

        # Leverage PyTorch's built-in scaled_dot_product_attention with causal masking
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec


class MHAPyTorchSDPAWithoutFlash(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's scaled_dot_product_attention without FlashAttention.

    This module implements multi-head attention using PyTorch's scaled_dot_product_attention function while explicitly
    disabling FlashAttention optimizations. This can be useful for debugging, compatibility testing, or when specific
    attention computation behavior is required. Unlike the regular PyTorch scaled dot-product attention, this
    implementation uses explicit masking rather than the is_causal parameter.

    Key features:
    1. Uses PyTorch's scaled_dot_product_attention with explicit masking
    2. Disables FlashAttention for consistent behavior across different hardware
    3. Combined QKV projection for efficient memory usage
    4. Explicit causal masking with registered buffer
    5. Output projection layer for learned combination of heads
    6. Training-aware dropout handling
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchSDPAWithoutFlash module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # Use Dropout only during training
        use_dropout = 0. if not self.training else self.dropout

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # Leverage PyTorch's built-in scaled_dot_product_attention with explicit mask
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec


class MHAPyTorchClass(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's built-in MultiheadAttention module.

    This module implements multi-head attention by leveraging PyTorch's nn.MultiheadAttention class, which provides
    a complete implementation of multi-head attention with various optimization options. This approach offers high-level
    abstraction and is well-tested, making it suitable for production use when you want to use PyTorch's official
    implementation rather than custom implementations.

    Key features:
    1. Uses PyTorch's official nn.MultiheadAttention module
    2. Configurable attention weight output for analysis
    3. Built-in optimization and numerical stability features
    4. Explicit causal masking support
    5. Additional output projection layer for enhanced representational capacity
    6. Comprehensive bias options for QKV projections
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False, need_weights: bool = True) -> None:
        """
        Initialize the MHAPyTorchClass module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the causal mask
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projections. Defaults to False.
            need_weights (bool, optional): Whether to return attention weights (for analysis). Defaults to True.

        Note:
            The d_in parameter is included for API consistency but is not directly used since
            nn.MultiheadAttention expects the input dimension to match embed_dim.
        """
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PyTorch MultiheadAttention-based mechanism.

        Computes multi-head attention using PyTorch's nn.MultiheadAttention by:
        1. Preparing causal mask for the current sequence length
        2. Calling PyTorch's multihead_attn with query=key=value=x (self-attention)
        3. Applying additional output projection for enhanced representation

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_out)
                       Note: Input dimension must match embed_dim from initialization

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            This implementation uses PyTorch's nn.MultiheadAttention in self-attention mode where
            query, key, and value are all the same input tensor. The causal mask is dynamically
            adjusted based on the actual sequence length to ensure proper autoregressive behavior.
        """
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        output = self.proj(attn_output)

        return output


class MHAPyTorchFlexAttention(nn.Module):
    """
    Multi-Head Attention implementation using PyTorch's FlexAttention.

    This module implements multi-head attention using PyTorch's flexible attention mechanism (FlexAttention), which
    provides advanced masking capabilities and efficient computation for various attention patterns. FlexAttention
    allows for custom attention patterns through user-defined mask functions and supports block-sparse attention
    patterns for improved memory efficiency and computational performance.

    Key features:
    1. Uses PyTorch's FlexAttention for advanced masking capabilities
    2. Block-sparse attention patterns for memory efficiency
    3. Custom causal masking through mask functions
    4. Combined QKV projection for efficient memory usage
    5. Output projection layer for learned combination of heads
    6. Dropout regularization for attention weights
    7. Optimized for large-scale transformer models with flexible attention patterns
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Initialize the MHAPyTorchFlexAttention module.

        Args:
            d_in (int): Input embedding dimension
            d_out (int): Total output embedding dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads
            context_length (int): Maximum sequence length for the block mask creation
            dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            qkv_bias (bool, optional): Whether to include bias in QKV linear projection. Defaults to False.

        Raises:
            AssertionError: If d_out is not divisible by num_heads
            RuntimeError: If PyTorch version is below 2.5
        """
        super().__init__()

        # Check PyTorch version (FlexAttention requires PyTorch 2.5+)
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 5):
            raise RuntimeError("MHAPyTorchFlexAttention requires PyTorch 2.5+ with CUDA or MPS support")

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        # Create block mask on CPU (FlexAttention only supports CPU, CUDA, or HPU)
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length, device='cpu')


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FlexAttention-based multi-head attention mechanism.

        Computes multi-head attention using PyTorch's FlexAttention by:
        1. Projecting input to unified QKV tensor (3 * d_out dimensions)
        2. Reshaping and permuting to separate Q, K, V for each head
        3. Creating appropriate block mask for the current sequence length
        4. Using FlexAttention with the causal block mask for efficient computation
        5. Combining heads and applying output projection

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Multi-head attention output of shape (batch_size, num_tokens, d_out)

        Note:
            FlexAttention uses block-sparse patterns for efficient memory usage and supports
            custom masking functions. The causal mask is created using create_block_mask with
            the causal function to ensure autoregressive behavior.

            FlexAttention only supports CPU, CUDA, and HPU devices. For MPS devices,
            the computation is moved to CPU and then back to the original device.
        """
        batch_size, num_tokens, embed_dim = x.shape
        original_device = x.device

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        # FlexAttention only supports CPU, CUDA, and HPU devices
        # Move to CPU if on unsupported device (like MPS)
        compute_device = 'cpu' if original_device.type not in ['cpu', 'cuda', 'hpu'] else original_device

        if original_device.type not in ['cpu', 'cuda', 'hpu']:
            queries = queries.to('cpu')
            keys = keys.to('cpu')
            values = values.to('cpu')

        # Create block mask with correct dimensions for current sequence length
        attn_mask = create_block_mask(causal, B=None, H=None, Q_LEN=num_tokens, KV_LEN=num_tokens, device=compute_device)

        # For testing/debugging purposes, disable the compile warning
        # In production, you would want to use torch.compile(flex_attention) on CUDA
        original_debug_setting = getattr(torch.nn.attention.flex_attention, '_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG', False)
        torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

        try:
            # Leverage PyTorch's built-in FlexAttention with the block mask
            context_vec = flex_attention(queries, keys, values, block_mask=attn_mask)
        finally:
            # Restore original debug setting
            torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = original_debug_setting

        # Move back to original device if necessary
        if original_device.type not in ['cpu', 'cuda', 'hpu']:
            context_vec = context_vec.to(original_device)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Apply output projection
        context_vec = self.proj(context_vec)

        return context_vec


class MultiHeadAttentionCached(nn.Module):
    """
    Multi-Head Attention with KV cache support.

    Implements sliding window cache optimization and efficient memory management for autoregressive inference.
    """

    def __init__(self, config: GptConfig) -> None:
        """
        Initialize the cached multi-head attention module.

        Args:
            config (GptConfig): Model configuration containing attention parameters
        """
        super().__init__()
        assert config.emb_dim % config.n_heads == 0, "emb_dim must be divisible by n_heads"

        self.d_out = config.emb_dim
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        self.W_query = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        # KV cache parameters
        self.max_seq_len = config.context_length
        self.window_size = getattr(config, 'kv_window_size', config.context_length)

        # Register cache buffers
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """
        Forward pass with optional KV caching.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, emb_dim]
            use_cache (bool): Whether to use KV caching

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, emb_dim]
        """
        b, num_tokens, d_in = x.shape

        # Compute Q, K, V
        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # Reshape for multi-head attention
        keys_new = keys_new.view(b, num_tokens, self.n_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)

        # Transpose for attention computation
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Handle KV caching
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros(b, self.n_heads, self.window_size, self.head_dim, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0

            # Handle cache overflow
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                # Shift cache left
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow

            # Update cache
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = values_new
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        K = attn_scores.size(-1)
        if num_tokens == K:
            # No cache - use triangular mask
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # With cache - offset diagonal
            offset = K - num_tokens
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx

        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        # Apply attention
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

    def reset_cache(self) -> None:
        """
        Reset the KV cache.
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_cur = 0
