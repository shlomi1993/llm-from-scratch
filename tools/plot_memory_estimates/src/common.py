"""
Common utilities and data structures for plotting memory estimates.
"""

import math


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8": 1,
    "int8": 1,
}


def kv_bytes_total(batch_size: int, context_length: int, emb_dim: int, n_heads: int, n_kv_heads: int, n_layers: int,
                   bytes_per_elem: int) -> int:
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


def bytes_to_gb(n_bytes: int) -> float:
    """
    Convert bytes to gigabytes.

    Args:
        n_bytes (int): Number of bytes to convert.

    Returns:
        float: Value in gigabytes.
    """
    return n_bytes / (1000.0 ** 3)


def parse_swa_ratio(ratio_str: str) -> tuple:
    """
    Parse SWA ratio string.

    Args:
        ratio_str (str): Ratio in format "a:b" (e.g., "5:1")

    Returns:
        Tuple of (a, b) integers

    Raises:
        ValueError: If ratio_str is not in valid format
    """
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa-ratio must be in the form 'a:b' with nonnegative integers and a+b>0")


def kv_bytes_total_mha(batch_size: int, context_length: int, emb_dim: int,
                       n_layers: int, bytes_per_elem: int) -> int:
    """
    Calculate total KV-cache bytes for MHA.

    For MHA, n_kv_heads = n_heads, which cancels out:
    total = B * L * E * 2 (K,V) * bytes * n_layers

    Args:
        batch_size (int): The number of sequences in a batch.
        context_length (int): Maximum sequence length
        emb_dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        bytes_per_elem (int): Bytes per element based on data type

    Returns:
        int: Total KV-cache bytes
    """
    return batch_size * context_length * emb_dim * 2 * bytes_per_elem * n_layers


def kv_bytes_total_gqa(batch_size: int, context_length: int, emb_dim: int,
                       n_layers: int, bytes_per_elem: int, n_kv_groups: int) -> int:
    """
    Calculate total KV-cache bytes for GQA.

    For GQA, n_kv_heads = n_heads / n_kv_groups => scale the MHA total by 1 / n_kv_groups

    Args:
        batch_size (int): The number of sequences in a batch.
        context_length (int): Maximum sequence length
        emb_dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        bytes_per_elem (int): Bytes per element based on data type
        n_kv_groups (int): Number of KV groups

    Returns:
        int: Total KV-cache bytes
    """
    base = kv_bytes_total_mha(batch_size, context_length, emb_dim, n_layers, bytes_per_elem)
    return base / n_kv_groups


def kv_bytes_total_mla(batch_size: int, context_length: int, n_layers: int,
                       latent_dim: int, bytes_per_elem: int) -> int:
    """
    Calculate total KV-cache bytes for MLA.

    Args:
        batch_size (int): The number of sequences in a batch.
        context_length (int): Maximum sequence length
        n_layers (int): Number of transformer layers
        latent_dim (int): Latent dimension
        bytes_per_elem (int): Bytes per element based on data type

    Returns:
        int: Total KV-cache bytes
    """
    return batch_size * context_length * n_layers * latent_dim * bytes_per_elem


def kv_bytes_total_mha_swa(batch_size: int, context_length: int, emb_dim: int, n_layers: int, bytes_per_elem: int,
                           window: int, swa_ratio: str) -> int:
    """
    Calculate total KV-cache bytes for MHA with SWA.

    Args:
        batch_size (int): The number of sequences in a batch.
        context_length (int): Maximum sequence length
        emb_dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        bytes_per_elem (int): Bytes per element based on data type
        window (int): Sliding window size
        swa_ratio (str): SWA ratio string (e.g., "5:1")

    Returns:
        int: Total KV-cache bytes
    """
    a, b = parse_swa_ratio(swa_ratio)
    total_blocks = a + b
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    n_full_layers = n_layers - n_swa_layers
    total_full = kv_bytes_total_mha(batch_size, context_length, emb_dim, n_full_layers, bytes_per_elem)
    total_swa = kv_bytes_total_mha(batch_size, window, emb_dim, n_swa_layers, bytes_per_elem)
    return total_full + total_swa


def kv_bytes_total_gqa_swa(batch_size: int, context_length: int, emb_dim: int, n_layers: int, bytes_per_elem: int,
                           n_kv_groups: int, window: int, swa_ratio: str) -> int:
    """
    Calculate total KV-cache bytes for GQA with SWA.

    Args:
        batch_size (int): The number of sequences in a batch.
        context_length (int): Maximum sequence length
        emb_dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        bytes_per_elem (int): Bytes per element based on data type
        n_kv_groups (int): Number of KV groups
        window (int): Sliding window size
        swa_ratio (str): SWA ratio string (e.g., "5:1")

    Returns:
        int: Total KV-cache bytes
    """
    a, b = parse_swa_ratio(swa_ratio)
    total_blocks = a + b
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    n_full_layers = n_layers - n_swa_layers
    total_full = kv_bytes_total_gqa(batch_size, context_length, emb_dim, n_full_layers, bytes_per_elem, n_kv_groups)
    total_swa = kv_bytes_total_gqa(batch_size, window, emb_dim, n_swa_layers, bytes_per_elem, n_kv_groups)
    return total_full + total_swa
