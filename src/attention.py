import math
import torch
import torch.nn as nn

from torch import Tensor


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
