"""
Memory estimate plotter package for LLM attention mechanisms and MoE architectures.

This package provides plotting tools for visualizing KV-cache memory usage across
different attention mechanisms (MHA, GQA, MLA, SWA) and parameter counts for MoE FFN layers.
"""

from .src.plotters import (
    plot_gqa,
    plot_mla,
    plot_swa,
    plot_moe,
)

from .src.parser import (
    parse_args,
    validate_args,
)

from .src.common import (
    bytes_to_gb,
    parse_swa_ratio,
    kv_bytes_total_mha,
    kv_bytes_total_gqa,
    kv_bytes_total_mla,
    kv_bytes_total_mha_swa,
    kv_bytes_total_gqa_swa,
)

__all__ = [
    # Plotting functions
    "plot_gqa",
    "plot_mla",
    "plot_swa",
    "plot_moe",
    # Argument parsing
    "parse_args",
    "validate_args",
    # Utility functions
    "bytes_to_gb",
    "parse_swa_ratio",
    "kv_bytes_total_mha",
    "kv_bytes_total_gqa",
    "kv_bytes_total_mla",
    "kv_bytes_total_mha_swa",
    "kv_bytes_total_gqa_swa",
]
