"""
Memory estimator package for LLM attention mechanisms and MoE architectures.

This package provides estimation tools for KV-cache memory usage across different
attention mechanisms (MHA, GQA, MLA, SWA) and parameter counts for MoE FFN layers.
"""

from .src import (
    # Common utilities
    DTYPE_BYTES,
    bytes_convert,
    kv_bytes_total,
    # Estimator functions
    estimate_mha,
    estimate_gqa,
    estimate_mla,
    estimate_swa,
    estimate_moe,
    # Mode enum
    Mode,
    # Result dataclasses
    MhaGqaResult,
    MlaResult,
    SwaResult,
    MoeResult,
)

__all__ = [
    # Common utilities
    "DTYPE_BYTES",
    "bytes_convert",
    "kv_bytes_total",
    # Estimator functions
    "estimate_mha",
    "estimate_gqa",
    "estimate_mla",
    "estimate_swa",
    "estimate_moe",
    # Mode enum
    "Mode",
    # Result dataclasses
    "MhaGqaResult",
    "MlaResult",
    "SwaResult",
    "MoeResult",
]
