
import math


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8": 1,
    "int8": 1,
}


def bytes_convert(n: int) -> str:
    """
    Convert bytes to gigabytes with formatted string output.

    Args:
        n (int): Number of bytes to convert.

    Returns:
        str: Formatted string representing the value in gigabytes (e.g., "1.23 GB").
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def kv_bytes_total(batch_size: int, context_length: int, emb_dim: int, n_heads: int, n_kv_heads: int, n_layers: int, bytes_per_elem: int) -> int:
    """
    Calculate total KV-cache memory in bytes for all layers.

    Args:
        batch_size (int): Number of sequences in a batch.
        context_length (int): Maximum sequence length.
        emb_dim (int): Embedding dimension.
        n_heads (int): Total number of attention heads.
        n_kv_heads (int): Number of key-value heads (n_heads for MHA, n_heads // n_kv_groups for GQA).
        n_layers (int): Number of transformer layers.
        bytes_per_elem (int): Bytes per element based on data type (e.g., 2 for fp16/bf16).

    Returns:
        int: Total memory in bytes required for KV-cache across all layers.
    """
    head_dim = math.ceil(emb_dim / n_heads)
    per_layer = batch_size * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_layers


def get_num_param_matrices(ffn_type: str) -> int:
    """
    Get number of parameter matrices for FFN type.

    Args:
        ffn_type (str): Either 'gelu' or 'swiglu'.

    Returns:
        int: Number of parameter matrices (2 for gelu, 3 for swiglu).

    Raises:
        ValueError: If ffn_type is not 'gelu' or 'swiglu'.
    """
    if ffn_type == "gelu":
        return 2
    if ffn_type == "swiglu":
        return 3
    raise ValueError("--ffn-type must be 'gelu' or 'swiglu'")
