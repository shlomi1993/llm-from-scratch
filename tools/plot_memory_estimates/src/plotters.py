"""
Plotting functions for different memory estimation modes.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

from argparse import Namespace
from pathlib import Path

# Add parent directory to path to import from memory_estimator  // TODO Ugly hack, find better solution, low priority
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from memory_estimator.src.estimators import estimate_moe

from .common import (
    DTYPE_BYTES,
    bytes_to_gb,
    kv_bytes_total,
    kv_bytes_total_mha,
    kv_bytes_total_gqa,
    kv_bytes_total_mla,
    kv_bytes_total_mha_swa,
    kv_bytes_total_gqa_swa,
)


def plot_gqa(args: Namespace) -> None:
    """
    Plot KV-cache vs context length for MHA and GQA with multiple n_kv_groups.

    Based on chapters/ch04/04_gqa/plot_memory_estimates_gqa.py

    Args:
        args (Namespace): Parsed command-line arguments
    """
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # Calculate MHA for all context lengths
    mha_gb = []
    for L in context_lengths:
        total_mha = kv_bytes_total(
            args.batch_size, L, args.emb_dim, args.n_heads,
            args.n_heads,  # MHA: n_kv_heads = n_heads
            args.n_layers, bytes_per_elem
        )
        mha_gb.append(bytes_to_gb(total_mha))

    plt.figure()
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    # GQA curves for selected n_kv_groups
    groups_list = [4, 8, 12, 24]
    for g in groups_list:
        if args.n_heads % g != 0:
            continue
        n_kv_heads = args.n_heads // g
        gqa_gb = []
        for L in context_lengths:
            total_gqa = kv_bytes_total(args.batch_size, L, args.emb_dim, args.n_heads, n_kv_heads, args.n_layers, bytes_per_elem)
            gqa_gb.append(bytes_to_gb(total_gqa))

        comp = (args.n_heads / n_kv_heads) if n_kv_heads > 0 else float("inf")
        plt.plot(context_lengths, gqa_gb, marker="o", label=f"GQA (n_kv_groups={g}, {comp:,.1f}× compression)")

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (multi-group)\n"
        f"(n_heads={args.n_heads}, emb_dim={args.emb_dim}, n_layers={args.n_layers}, "
        f"batch={args.batch_size}, dtype={args.dtype})",
        fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    output_path = args.output or "kv_bytes_vs_context_length_gqa.pdf"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_mla(args: Namespace) -> None:
    """
    Plot KV-cache vs context length for MHA and MLA with multiple latent dimensions.

    Based on chapters/ch04/05_mla/plot_memory_estimates_mla.py

    Args:
        args (Namespace): Parsed command-line arguments
    """
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    context_lengths = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # Calculate MHA for all context lengths
    mha_gb = []
    for L in context_lengths:
        total_mha = kv_bytes_total(args.batch_size, L, args.emb_dim, args.n_heads, args.n_heads, args.n_layers, bytes_per_elem)
        mha_gb.append(bytes_to_gb(total_mha))

    plt.figure()
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    # MLA curves for selected latent dimensions
    latent_dims = args.latent_dims or [1024, 512, 256, 64]
    L_ref = context_lengths[-1]
    total_mha_ref = kv_bytes_total(args.batch_size, L_ref, args.emb_dim, args.n_heads, args.n_heads, args.n_layers, bytes_per_elem)

    for latent_dim in latent_dims:
        mla_gb = []
        for L in context_lengths:
            total_mla = kv_bytes_total_mla(args.batch_size, L, args.n_layers, latent_dim, bytes_per_elem)
            mla_gb.append(bytes_to_gb(total_mla))

        total_mla_ref = kv_bytes_total_mla(args.batch_size, L_ref, args.n_layers, latent_dim, bytes_per_elem)
        comp = total_mha_ref / total_mla_ref if total_mla_ref != 0 else float("inf")

        plt.plot(context_lengths, mla_gb, marker="o", label=f"MLA (latent_dim={latent_dim}, {comp:,.1f}× compression)")

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs MLA\n"
        f"(n_heads={args.n_heads}, emb_dim={args.emb_dim}, n_layers={args.n_layers}, "
        f"batch={args.batch_size}, dtype={args.dtype})",
        fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    output_path = args.output or "kv_bytes_vs_context_length_mla.pdf"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_swa(args: Namespace) -> None:
    """
    Plot KV-cache vs context length for MHA and GQA with SWA overlays.

    Based on chapters/ch04/06_swa/plot_memory_estimates_swa.py

    Args:
        args (Namespace): Parsed command-line arguments
    """
    bytes_per_elem = DTYPE_BYTES[args.dtype]
    n_kv_groups = args.n_kv_groups or 4
    valid_gqa = (args.n_heads % n_kv_groups == 0)

    context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    series = {
        "MHA (KV total)": [],
        f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})": [],
    }
    if valid_gqa:
        series[f"GQA kv_groups={n_kv_groups} (full)"] = []
        series[f"SWA on GQA kv_groups={n_kv_groups} (ratio {args.swa_ratio}, W={args.sliding_window_size})"] = []

    for L in context_lengths:
        total_mha = kv_bytes_total_mha(args.batch_size, L, args.emb_dim, args.n_layers, bytes_per_elem)
        total_mha_swa = kv_bytes_total_mha_swa(args.batch_size, L, args.emb_dim, args.n_layers, bytes_per_elem, window=args.sliding_window_size, swa_ratio=args.swa_ratio)
        series["MHA (KV total)"].append(bytes_to_gb(total_mha))
        series[f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})"].append(bytes_to_gb(total_mha_swa))

        if valid_gqa:
            total_gqa = kv_bytes_total_gqa(args.batch_size, L, args.emb_dim, args.n_layers, bytes_per_elem, n_kv_groups)
            total_gqa_swa = kv_bytes_total_gqa_swa(args.batch_size, L, args.emb_dim, args.n_layers, bytes_per_elem, n_kv_groups, args.sliding_window_size, args.swa_ratio)
            series[f"GQA kv_groups={n_kv_groups} (full)"].append(bytes_to_gb(total_gqa))
            series[f"SWA on GQA kv_groups={n_kv_groups} (ratio {args.swa_ratio}, W={args.sliding_window_size})"].append(bytes_to_gb(total_gqa_swa))

    plt.figure(figsize=(10, 5))
    x = np.array(context_lengths, dtype=float)

    colors = {
        "MHA": "#1f77b4",
        "GQA": "#ff7f0e",
    }

    for label, yvals in series.items():
        y = np.array(yvals, dtype=float)
        if np.all(np.isnan(y)):
            continue

        linestyle = "--" if "SWA" in label else "-"
        if "MHA" in label:
            color = colors["MHA"]
        elif "GQA" in label:
            color = colors["GQA"]
        else:
            color = None

        plt.plot(x, y, marker="o", label=label, linestyle=linestyle, color=color)

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (SWA overlays)\n"
        f"(n_heads={args.n_heads}, emb_dim={args.emb_dim}, n_layers={args.n_layers}, "
        f"batch={args.batch_size}, dtype={args.dtype}; "
        f"SWA ratio={args.swa_ratio}, W={args.sliding_window_size})",
        fontsize=8,
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    output_path = args.output or "kv_bytes_vs_context_length_swa.pdf"
    plt.savefig(output_path)
    plt.close()

    if not valid_gqa:
        print(f"Skipped GQA kv_groups={n_kv_groups} because n_heads={args.n_heads} is not divisible by {n_kv_groups}.")
    print(f"Saved plot to: {output_path}")


def plot_moe(args: Namespace) -> None:
    """
    Plot active vs total parameters per token for MoE vs dense FFN.

    Based on chapters/ch04/07_moe/plot_memory_estimates_moe.py

    Args:
        args (Namespace): Parsed command-line arguments
    """
    # Expert counts to plot
    experts = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
    max_experts = args.max_experts or 512
    experts = [e for e in experts if e <= max_experts]

    moe_active = []
    moe_total = []

    for e in experts:
        result = estimate_moe(
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            ffn_type=args.ffn_type,
            num_experts=e,
            top_k=args.top_k,
            dtype=args.dtype,
            match_dense=args.match_dense,
        )
        moe_active.append(result.moe_active_params_per_token)
        moe_total.append(result.moe_total)

    # Get dense params for baseline
    result_dense = estimate_moe(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        ffn_type=args.ffn_type,
        num_experts=1,
        top_k=1,
        dtype=args.dtype,
        match_dense=False,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(experts, moe_active, marker="o", label="MoE active per token")
    plt.plot(experts, moe_total, marker="s", linestyle="--", label="MoE total parameters")
    plt.axhline(result_dense.dense_params, linestyle=":", color="gray", label="FFN dense (active = total)")
    plt.xlabel(f"Number of experts (top_k = {args.top_k})")
    plt.ylabel("Parameters")
    if not args.no_log:
        plt.yscale("log")
    plt.title(f"Active vs Total Parameters per Token\n(emb_dim={args.emb_dim}, hidden_dim={args.hidden_dim}, ffn={args.ffn_type}, top_k={args.top_k})")
    plt.legend()
    plt.tight_layout()

    output_path = args.output or "moe_params_vs_experts.pdf"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {output_path}")
