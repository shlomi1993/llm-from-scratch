import math

from typing import Tuple

from .common import DTYPE_BYTES, kv_bytes_total, get_num_param_matrices
from .results import MhaGqaResult, MlaResult, SwaResult, MoeResult
from src.configurations import GptConfig


def estimate_mha(config: GptConfig, batch_size: int, dtype: str) -> MhaGqaResult:
    """
    Estimate MHA memory usage.

    Args:
        config (GptConfig): Model configuration.
        batch_size (int): Batch size for inference.
        dtype (str): Data type (fp32, bf16, fp16, fp8, int8).

    Returns:
        MhaGqaResult: Dataclass containing memory estimates (total_gqa will equal total_mha).
    """
    bytes_per_elem = DTYPE_BYTES[dtype]
    head_dim = math.ceil(config.emb_dim / config.n_heads)
    n_kv_heads_mha = config.n_heads

    total_mha = kv_bytes_total(batch_size, config.context_length, config.emb_dim, config.n_heads, n_kv_heads_mha,
                               config.n_layers, bytes_per_elem)

    return MhaGqaResult(
        bytes_per_elem=bytes_per_elem,
        head_dim=head_dim,
        n_kv_heads_gqa=n_kv_heads_mha,  # For MHA, n_kv_heads equals n_heads
        total_mha=total_mha,
        total_gqa=total_mha,  # For MHA-only, GQA total equals MHA total
        ratio=1.0,
        savings=0.0,
    )


def estimate_gqa(config: GptConfig, batch_size: int, dtype: str, n_kv_groups: int) -> MhaGqaResult:
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
    result = estimate_gqa(config, batch_size, dtype, n_kv_groups)

    bytes_per_elem = DTYPE_BYTES[dtype]
    total_mla = batch_size * config.context_length * config.n_layers * latent_dim * bytes_per_elem

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

    def parse_swa_ratio(ratio_str: str) -> Tuple[int, int]:
        try:
            a_str, b_str = ratio_str.split(":")
            a, b = int(a_str), int(b_str)
            assert a >= 0 and b >= 0 and (a + b) > 0
            return a, b
        except Exception:
            raise ValueError("--swa-ratio must be in the form 'a:b' with nonnegative integers and a + b > 0")

    def distribute_swa_layers(n_layers: int, a: int, b: int) -> Tuple[int, int]:
        block = a + b
        blocks = n_layers // block
        rem = n_layers % block
        swa = blocks * a + min(a, rem)
        full = blocks * b + max(0, rem - a)
        return swa, full

    a_swa, b_full = parse_swa_ratio(swa_ratio)
    n_swa_layers, n_full_layers = distribute_swa_layers(config.n_layers, a_swa, b_full)

    eff_W = min(config.context_length, sliding_window_size)
    L = config.context_length

    def kv_bytes_per_layer(batch_size: int, context_length: int, head_dim: int, n_kv_heads: int, bytes_per_elem: int) -> int:
        return batch_size * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem

    # Per-layer costs
    per_mha_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_gqa, bytes_per_elem)
    per_mha_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_gqa, bytes_per_elem)

    # Totals
    total_mha_all_full = per_mha_full * config.n_layers
    total_gqa_all_full = per_gqa_full * config.n_layers
    total_mixed_mha = n_swa_layers * per_mha_swa + n_full_layers * per_mha_full
    total_mixed_gqa = n_swa_layers * per_gqa_swa + n_full_layers * per_gqa_full

    return SwaResult(
        bytes_per_elem=bytes_per_elem,
        head_dim=head_dim,
        n_kv_heads_gqa=n_kv_heads_gqa,
        eff_W=eff_W,
        n_swa_layers=n_swa_layers,
        n_full_layers=n_full_layers,
        total_mha_all_full=total_mha_all_full,
        total_gqa_all_full=total_gqa_all_full,
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

    def count_ffn_params(emb_dim: int, hidden_dim: int, ffn_type: str) -> int:
        return get_num_param_matrices(ffn_type) * emb_dim * hidden_dim

    p_dense = count_ffn_params(emb_dim, hidden_dim, ffn_type)
    r = emb_dim * num_experts  # Router parameters

    if match_dense:
        num_param_matrices = get_num_param_matrices(ffn_type)
        num = p_dense - r
        den = num_experts * num_param_matrices * emb_dim
        if num <= 0:
            raise ValueError("Dense layer too small for requested num_experts.")
        moe_hidden_dim = int(round(num / float(den)))
    else:
        moe_hidden_dim = hidden_dim

    per_expert_params = count_ffn_params(emb_dim, moe_hidden_dim, ffn_type)
    moe_total = num_experts * per_expert_params + r
    moe_active_params_per_token = r + top_k * per_expert_params

    return MoeResult(
        dense_params=p_dense,
        router=r,
        moe_hidden_dim=moe_hidden_dim,
        per_expert_params=per_expert_params,
        moe_total=moe_total,
        moe_active_params_per_token=moe_active_params_per_token,
        bytes_per_elem=bytes_per_elem,
    )
