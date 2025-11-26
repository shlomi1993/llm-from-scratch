# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Unified KV-cache memory estimator for MHA, GQA, MLA, SWA, and MoE FFN.

This tool provides comprehensive memory estimation across multiple attention mechanisms and FFN architectures to help
with model architecture decisions.
"""

import argparse
import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from src.configurations import GptConfig


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8": 1,
    "int8": 1,
}


@dataclass
class AbstractResults(ABC):

    @abstractmethod
    def print(self) -> None:
        """Print estimation results."""
        pass


@dataclass
class MhaGqaResult(AbstractResults):
    """
    Results from MHA vs GQA estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    total_mha: int
    total_gqa: int
    ratio: float
    savings: float

    def print(self, config: GptConfig, args: argparse.Namespace) -> None:
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA total KV cache  : {bytes_convert(self.total_mha)}")
        print(f"GQA total KV cache  : {bytes_convert(self.total_gqa)}")
        print(f"Ratio (MHA / GQA)   : {self.ratio:,.2f}x")
        print(f"Savings (GQA vs MHA): {self.savings * 100:,.2f}%")


@dataclass
class MlaResult(AbstractResults):
    """
    Results from MHA vs GQA vs MLA estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    total_mha: int
    total_gqa: int
    ratio: float
    savings: float
    latent_dim: int
    total_mla: int
    ratio_mha_mla: float
    savings_mla: float

    def print(self, config: GptConfig, args: argparse.Namespace) -> None:
        """Print MHA vs GQA vs MLA estimation results."""
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA total KV cache  : {bytes_convert(self.total_mha)}")
        print(f"GQA total KV cache  : {bytes_convert(self.total_gqa)}")
        print(f"MLA total KV cache  : {bytes_convert(self.total_mla)}")
        print(f"Ratio (MHA / GQA)   : {self.ratio:,.2f}x")
        print(f"Savings (GQA vs MHA): {self.savings * 100:,.2f}%")
        print(f"Ratio (MHA / MLA)   : {self.ratio_mha_mla:,.2f}x")
        print(f"Savings (MLA vs MHA): {self.savings_mla * 100:,.2f}%")




@dataclass
class SwaResult(AbstractResults):
    """
    Results from Sliding Window Attention estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    eff_W: int
    n_swa_layers: int
    n_full_layers: int
    total_mha_allfull: int
    total_gqa_allfull: int
    total_mixed_mha: int
    total_mixed_gqa: int

    def print(self, config: GptConfig, args: argparse.Namespace) -> None:
        """Print SWA estimation results."""
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'sliding_window_size':23}: {args.sliding_window_size}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print(f"{'Effective SWA window W':23}: {self.eff_W}")
        print(f"{'Layer ratio (SWA:Full)':23}: {args.swa_ratio} -> {self.n_swa_layers} SWA, {self.n_full_layers} Full")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA KV total           : {bytes_convert(self.total_mha_allfull)}")
        print(f"GQA KV total           : {bytes_convert(self.total_gqa_allfull)}")
        print(f"MHA + SWA (ratio {args.swa_ratio})  : {bytes_convert(self.total_mixed_mha)}")
        print(f"GQA + SWA (ratio {args.swa_ratio})  : {bytes_convert(self.total_mixed_gqa)}")


@dataclass
class MoeResult(AbstractResults):
    """
    Results from MoE FFN estimation.
    """
    dense_params: int
    router: int
    moe_hidden_dim: int
    per_expert_params: int
    moe_total: int
    moe_active_params_per_token: int
    bytes_per_elem: int

    def print(self, args: argparse.Namespace) -> None:
        """Print MoE estimation results."""
        print("==== Config ====")
        print(f"{'emb_dim':23}: {args.emb_dim}")
        print(f"{'hidden_dim':23}: {args.hidden_dim}")
        print(f"{'ffn_type':23}: {args.ffn_type}")
        print(f"{'num_experts':23}: {args.num_experts}")
        print(f"{'top_k':23}: {args.top_k}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'match_dense':23}: {args.match_dense}")
        print()
        print("==== Model weights (parameters) ====")
        print(f"{'Dense FFN params':23}: {self.dense_params:,} ({bytes_convert(self.dense_params * self.bytes_per_elem)})")
        print(f"{'Per-expert params':23}: {self.per_expert_params:,} ({bytes_convert(self.per_expert_params * self.bytes_per_elem)})")
        print(f"{'Router params':23}: {self.router:,} ({bytes_convert(self.router * self.bytes_per_elem)})")
        print(f"{'MoE TOTAL params':23}: {self.moe_total:,} ({bytes_convert(self.moe_total * self.bytes_per_elem)})")
        print(f"{'MoE ACTIVE/Token':23}: {self.moe_active_params_per_token:,} ({bytes_convert(self.moe_active_params_per_token * self.bytes_per_elem)})")
        print(f"{'moe_hidden_dim':23}: {self.moe_hidden_dim}")


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


def mla_bytes_total(batch_size: int, context_length: int, n_layers: int, latent_dim: int, bytes_per_elem: int) -> int:
    """
    Calculate total memory for KV-cache using Multi-Head Latent Attention (MLA).

    MLA compresses the KV-cache by storing compressed latent representations instead of full per-head KV states,
    significantly reducing memory usage.

    Args:
        batch_size (int): Number of sequences in a batch.
        context_length (int): Maximum sequence length.
        n_layers (int): Number of transformer layers.
        latent_dim (int): Dimension of the compressed latent space.
        bytes_per_elem (int): Bytes per element based on data type (e.g., 2 for fp16/bf16).
    Returns:
        int: Total memory in bytes required for MLA cache across all layers.
    """
    return batch_size * context_length * n_layers * latent_dim * bytes_per_elem


def kv_bytes_per_layer(batch_size: int, context_length: int, head_dim: int, n_kv_heads: int, bytes_per_elem: int) -> int:
    """
    Calculate KV-cache memory for a single layer.

    Args:
        batch_size (int): Number of sequences in a batch.
        context_length (int): Maximum sequence length.
        head_dim (int): Dimension per attention head.
        n_kv_heads (int): Number of key-value heads.
        bytes_per_elem (int): Bytes per element based on data type.

    Returns:
        int: Memory in bytes for KV-cache of one layer.
    """
    return batch_size * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem


def parse_swa_ratio(ratio_str: str) -> Tuple[int, int]:
    """
    Parse SWA ratio string in format 'a:b'.

    Args:
        ratio_str (str): Ratio string like '1:0', '5:1', or '1:5'.

    Returns:
        Tuple[int, int]: Tuple of (a, b) representing SWA:Full layer ratio.

    Raises:
        ValueError: If ratio string is invalid.
    """
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa-ratio must be in the form 'a:b' with nonnegative integers and a + b > 0")


def distribute_swa_layers(n_layers: int, a: int, b: int) -> Tuple[int, int]:
    """
    Distribute layers into SWA and full attention layers based on ratio.

    Args:
        n_layers (int): Total number of layers.
        a (int): Number of SWA layers per block.
        b (int): Number of full layers per block.

    Returns:
        Tuple[int, int]: Tuple of (n_swa_layers, n_full_layers).
    """
    block = a + b
    blocks = n_layers // block
    rem = n_layers % block
    swa = blocks * a + min(a, rem)
    full = blocks * b + max(0, rem - a)
    return swa, full


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


def ffn_params(emb_dim: int, hidden_dim: int, ffn_type: str) -> int:
    """
    Calculate number of parameters in FFN layer.

    Args:
        emb_dim (int): Embedding dimension.
        hidden_dim (int): FFN hidden dimension.
        ffn_type (str): Type of FFN ('gelu' or 'swiglu').

    Returns:
        int: Number of parameters in FFN.
    """
    return get_num_param_matrices(ffn_type) * emb_dim * hidden_dim


def router_params(emb_dim: int, num_experts: int) -> int:
    """
    Calculate number of parameters in MoE router.

    Args:
        emb_dim (int): Embedding dimension.
        num_experts (int): Number of experts in MoE.

    Returns:
        int: Number of router parameters.
    """
    return emb_dim * num_experts


def estimate_mha_gqa(config: GptConfig, batch_size: int, dtype: str, n_kv_groups: int) -> MhaGqaResult:
    """
    Estimate MHA vs GQA memory usage.

    Args:
        config (GptConfig): Model configuration.
        batch_size (int): Batch size for inference.
        dtype (str): Data type (fp32, bf16, fp16, fp8, int8).
        n_kv_groups (int): Number of KV groups for GQA.

    Returns:
        MhaGqaResult: Dataclass containing memory estimates and metrics.
    """
    if config.n_heads % n_kv_groups != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[dtype]
    head_dim = math.ceil(config.emb_dim / config.n_heads)
    n_kv_heads_mha = config.n_heads
    n_kv_heads_gqa = config.n_heads // n_kv_groups

    total_mha = kv_bytes_total(batch_size, config.context_length, config.emb_dim, config.n_heads, n_kv_heads_mha,
                               config.n_layers, bytes_per_elem)

    total_gqa = kv_bytes_total(batch_size, config.context_length, config.emb_dim, config.n_heads, n_kv_heads_gqa,
                               config.n_layers, bytes_per_elem)

    ratio = total_mha / total_gqa if total_gqa != 0 else float("inf")
    savings = 1 - (total_gqa / total_mha) if total_mha != 0 else 0.0

    return MhaGqaResult(
        bytes_per_elem=bytes_per_elem,
        head_dim=head_dim,
        n_kv_heads_gqa=n_kv_heads_gqa,
        total_mha=total_mha,
        total_gqa=total_gqa,
        ratio=ratio,
        savings=savings,
    )


def estimate_mla(config: GptConfig, batch_size: int, dtype: str, latent_dim: int, n_kv_groups: int) -> MlaResult:
    """
    Estimate MHA vs GQA vs MLA memory usage.

    Args:
        config (GptConfig): Model configuration.
        batch_size (int): Batch size for inference.
        dtype (str): Data type (fp32, bf16, fp16, fp8, int8).
        latent_dim (int): Latent dimension for MLA.
        n_kv_groups (int): Number of KV groups for GQA.
    Returns:
        MlaResult: Dataclass containing memory estimates and metrics.
    """
    result = estimate_mha_gqa(config, batch_size, dtype, n_kv_groups)

    bytes_per_elem = DTYPE_BYTES[dtype]
    total_mla = mla_bytes_total(batch_size, config.context_length, config.n_layers, latent_dim, bytes_per_elem)

    ratio_mha_mla = result.total_mha / total_mla if total_mla != 0 else float("inf")
    savings_mla = 1 - (total_mla / result.total_mha) if result.total_mha != 0 else 0.0

    return MlaResult(
        bytes_per_elem=result.bytes_per_elem,
        head_dim=result.head_dim,
        n_kv_heads_gqa=result.n_kv_heads_gqa,
        total_mha=result.total_mha,
        total_gqa=result.total_gqa,
        ratio=result.ratio,
        savings=result.savings,
        latent_dim=latent_dim,
        total_mla=total_mla,
        ratio_mha_mla=ratio_mha_mla,
        savings_mla=savings_mla,
    )


def estimate_swa(config: GptConfig, batch_size: int, dtype: str, n_kv_groups: int, sliding_window_size: int,
                 swa_ratio: str) -> SwaResult:
    """
    Estimate memory usage with Sliding Window Attention (SWA).

    Args:
        config (GptConfig): Model configuration.
        batch_size (int): Batch size for inference.
        dtype (str): Data type (fp32, bf16, fp16, fp8, int8).
        n_kv_groups (int): Number of KV groups for GQA.
        sliding_window_size (int): Window size for SWA layers.
        swa_ratio (str): Ratio string like '1:0' or '5:1' (SWA:Full).

    Returns:
        SwaResult: Dataclass containing memory estimates and metrics.
    """
    if config.n_heads % n_kv_groups != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[dtype]
    head_dim = math.ceil(config.emb_dim / config.n_heads)
    n_kv_heads_mha = config.n_heads
    n_kv_heads_gqa = config.n_heads // n_kv_groups

    a_swa, b_full = parse_swa_ratio(swa_ratio)
    n_swa_layers, n_full_layers = distribute_swa_layers(config.n_layers, a_swa, b_full)

    eff_W = min(config.context_length, sliding_window_size)
    L = config.context_length

    # Per-layer costs
    per_mha_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_gqa, bytes_per_elem)
    per_mha_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_gqa, bytes_per_elem)

    # Totals
    total_mha_allfull = per_mha_full * config.n_layers
    total_gqa_allfull = per_gqa_full * config.n_layers
    total_mixed_mha = n_swa_layers * per_mha_swa + n_full_layers * per_mha_full
    total_mixed_gqa = n_swa_layers * per_gqa_swa + n_full_layers * per_gqa_full

    return SwaResult(
        bytes_per_elem=bytes_per_elem,
        head_dim=head_dim,
        n_kv_heads_gqa=n_kv_heads_gqa,
        eff_W=eff_W,
        n_swa_layers=n_swa_layers,
        n_full_layers=n_full_layers,
        total_mha_allfull=total_mha_allfull,
        total_gqa_allfull=total_gqa_allfull,
        total_mixed_mha=total_mixed_mha,
        total_mixed_gqa=total_mixed_gqa,
    )


def estimate_moe(emb_dim: int, hidden_dim: int, ffn_type: str, num_experts: int, top_k: int, dtype: str, match_dense: bool = False) -> MoeResult:
    """
    Estimate FFN vs MoE parameter memory.

    Args:
        emb_dim (int): Model embedding dimension.
        hidden_dim (int): FFN intermediate/hidden dimension.
        ffn_type (str): Type of FFN ('gelu' or 'swiglu').
        num_experts (int): Number of experts in MoE.
        top_k (int): Number of experts activated per token.
        dtype (str): Data type (fp32, bf16, fp16, fp8, int8).
        match_dense (bool): If True, auto-set per-expert hidden to match dense params.

    Returns:
        MoeResult: Dataclass containing parameter counts and memory estimates.
    """
    bytes_per_elem = DTYPE_BYTES[dtype]

    P_dense = ffn_params(emb_dim, hidden_dim, ffn_type)
    R = router_params(emb_dim, num_experts)

    if match_dense:
        num_param_matrices = get_num_param_matrices(ffn_type)
        num = P_dense - R
        den = num_experts * num_param_matrices * emb_dim
        if num <= 0:
            raise ValueError("Dense layer too small for requested num_experts.")
        moe_hidden_dim = int(round(num / float(den)))
    else:
        moe_hidden_dim = hidden_dim

    per_expert_params = ffn_params(emb_dim, moe_hidden_dim, ffn_type)
    moe_total = num_experts * per_expert_params + R
    moe_active_params_per_token = R + top_k * per_expert_params

    return MoeResult(
        dense_params=P_dense,
        router=R,
        moe_hidden_dim=moe_hidden_dim,
        per_expert_params=per_expert_params,
        moe_total=moe_total,
        moe_active_params_per_token=moe_active_params_per_token,
        bytes_per_elem=bytes_per_elem,
    )





def main() -> None:
    """
    Command-line interface for unified memory estimation.

    Supports MHA vs GQA, MLA, SWA, and MoE FFN estimation modes.
    """
    parser = argparse.ArgumentParser(description="Unified memory estimator for MHA/GQA/MLA/SWA and MoE FFN")

    # Mode selection
    parser.add_argument("--mode", choices=["gqa", "mla", "swa", "moe"], required=True,
                        help="Estimation mode: gqa (MHA vs GQA), mla (MHA vs GQA vs MLA), swa (Sliding Window Attention), moe (Mixture of Experts FFN)")

    # Common arguments for attention-based modes
    parser.add_argument("--context-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--emb-dim", type=int, help="Embedding dimension (required for gqa/mla/swa modes)")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads (required for gqa/mla/swa modes)")
    parser.add_argument("--n-layers", type=int, help="Number of transformer layers (required for gqa/mla/swa modes)")
    parser.add_argument("--n-kv-groups", type=int, help="Number of KV groups for GQA (required for gqa/mla/swa modes)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="float16", help="Data type")

    # MLA-specific
    parser.add_argument("--latent-dim", type=int, help="Latent dimension for MLA (required for mla mode)")
    # SWA-specific
    parser.add_argument("--sliding-window-size", type=int, help="SWA window size (required for swa mode)")
    parser.add_argument("--swa-ratio", type=str, default="1:0", help="SWA:Full layer ratio (e.g., '5:1', '1:5', default '1:0' = all SWA)")

    # MoE-specific
    parser.add_argument("--hidden-dim", type=int, help="FFN hidden dimension (required for moe mode)")
    parser.add_argument("--ffn-type", choices=["gelu", "swiglu"], default="swiglu", help="FFN type (for moe mode)")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts (for moe mode)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of experts activated per token (for moe mode)")
    parser.add_argument("--match-dense", action="store_true", help="Auto-set per-expert hidden to match dense FFN params (for moe mode)")

    args = parser.parse_args()

    # Validate mode-specific required arguments
    if args.mode in ["gqa", "mla", "swa"]:
        if not all([args.emb_dim, args.n_heads, args.n_layers, args.n_kv_groups]):
            parser.error(f"{args.mode} mode requires --emb_dim, --n_heads, --n_layers, --n_kv_groups")

    if args.mode == "mla" and not args.latent_dim:
        parser.error("mla mode requires --latent_dim")

    if args.mode == "swa" and not args.sliding_window_size:
        parser.error("swa mode requires --sliding_window_size")

    if args.mode == "moe":
        if not all([args.emb_dim, args.hidden_dim]):
            parser.error("moe mode requires --emb_dim and --hidden_dim")

    # Execute appropriate estimation
    if args.mode == "gqa":
        config = GptConfig(
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_length=args.context_length,
        )
        result = estimate_mha_gqa(config, args.batch_size, args.dtype, args.n_kv_groups)
        result.print(args.config, args)

    elif args.mode == "mla":
        config = GptConfig(
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_length=args.context_length,
        )
        result = estimate_mla(config, args.batch_size, args.dtype, args.latent_dim, args.n_kv_groups)
        result.print(args.config, args)

    elif args.mode == "swa":
        config = GptConfig(
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_length=args.context_length,
        )
        result = estimate_swa(config, args.batch_size, args.dtype, args.n_kv_groups, args.sliding_window_size, args.swa_ratio)
        result.print(args.config, args)

    elif args.mode == "moe":
        result = estimate_moe(args.emb_dim, args.hidden_dim, args.ffn_type, args.num_experts, args.top_k, args.dtype, args.match_dense)
        result.print(args.config, args)

if __name__ == "__main__":
    main()
