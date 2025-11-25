# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# KV-cache memory estimator for MHA vs GQA

import argparse
import math

from src.configurations import GptConfig


DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def bytes_convert(n: int) -> str:
    """Convert bytes to gigabytes with formatted string output.

    Args:
        n: Number of bytes to convert.

    Returns:
        Formatted string representing the value in gigabytes (e.g., "1.23 GB").
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def kv_bytes_total(
    batch_size: int,
    context_length: int,
    emb_dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    bytes_per_elem: int
) -> int:
    """
    Calculate total KV-cache memory in bytes for all layers.

    Args:
        batch_size: Number of sequences in a batch.
        context_length: Maximum sequence length.
        emb_dim: Embedding dimension.
        n_heads: Total number of attention heads.
        n_kv_heads: Number of key-value heads (n_heads for MHA, n_heads // n_kv_groups for GQA).
        n_layers: Number of transformer layers.
        bytes_per_elem: Bytes per element based on data type (e.g., 2 for fp16/bf16).

    Returns:
        Total memory in bytes required for KV-cache across all layers.
    """
    head_dim = math.ceil(emb_dim / n_heads)
    per_layer = batch_size * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_layers


def main() -> None:
    """
    Command-line interface for estimating KV-cache memory usage.

    Compares Multi-Head Attention (MHA) vs Grouped Query Attention (GQA) memory requirements based on model
    configuration parameters.
    """
    p = argparse.ArgumentParser(description="Estimate KV-cache memory for MHA vs GQA")
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    args = p.parse_args()

    cfg = GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        context_length=args.context_length,
        n_kv_groups=args.n_kv_groups,
    )

    if cfg.n_heads % cfg.n_kv_groups != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]
    head_dim = math.ceil(cfg.emb_dim / cfg.n_heads)
    n_kv_heads_mha = cfg.n_heads
    n_kv_heads_gqa = cfg.n_heads // cfg.n_kv_groups

    total_mha = kv_bytes_total(
        args.batch_size,
        cfg.context_length,
        cfg.emb_dim,
        cfg.n_heads,
        n_kv_heads_mha,
        cfg.n_layers,
        bytes_per_elem,
    )

    total_gqa = kv_bytes_total(
        args.batch_size,
        cfg.context_length,
        cfg.emb_dim,
        cfg.n_heads,
        n_kv_heads_gqa,
        cfg.n_layers,
        bytes_per_elem,
    )

    ratio = total_mha / total_gqa
    savings = 1 - (total_gqa / total_mha)

    print("==== Config ====")
    for k, v in vars(cfg).items():
        print(f"{k:17}: {v}")
    print(f"batch_size       : {args.batch_size}")
    print(f"dtype            : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim         : {head_dim}")
    print(f"GQA n_kv_heads   : {n_kv_heads_gqa}")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache  : {bytes_convert(total_mha)}")
    print(f"GQA total KV cache  : {bytes_convert(total_gqa)}")
    print(f"Ratio (MHA / GQA)   : {ratio:,.2f}x")
    print(f"Savings (GQA vs MHA): {savings*100:,.2f}%")


if __name__ == "__main__":
    main()
