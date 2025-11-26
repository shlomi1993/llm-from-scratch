from argparse import Namespace, ArgumentParser
from typing import List

from .common import DTYPE_BYTES
from .mode import Mode, MODE_FLAG_REQUIREMENTS


def find_missing_flags(mode: Mode, args: Namespace) -> List[str]:
    """
    Find missing required flags for a given mode.

    Args:
        mode: The mode to check.
        args: Parsed command-line arguments.

    Returns:
        List[str]: List of missing flag names.
    """
    requirements = MODE_FLAG_REQUIREMENTS[mode]
    missing_flags = [flag for flag in requirements if getattr(args, flag.lstrip("--").replace("-", "_")) is None]
    return missing_flags


def validate_args(args: Namespace) -> List[str]:
    """
    Validate command-line arguments for required flags based on selected modes.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List[str]: List of error messages for missing required flags.
    """
    error_list = []
    for mode in args.mode:
        missing_flags = find_missing_flags(Mode(mode), args)
        if missing_flags:
            error_list.append(f"  Mode '{mode}' requires " + ", ".join(missing_flags))
    return error_list


def parse_args() -> Namespace:
    """
    Parse command-line arguments for the unified memory estimator.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser(description="Unified memory estimator for MHA/GQA/MLA/SWA and MoE FFN")

    # Mode selection
    parser.add_argument("--mode", nargs="+", choices=["mha", "gqa", "mla", "swa", "moe"], required=True,
                        help="Estimation mode(s): " + ", ".join([
                            "mha (Multi-Head Attention)",
                            "gqa (Grouped Query Attention)",
                            "mla (Multi-Head Latent Attention)",
                            "swa (Sliding Window Attention)",
                            "moe (Mixture of Experts FFN). Multiple modes can be specified."]))

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

    error_list = validate_args(args)
    if error_list:
        parser.error("\n" + "\n".join(error_list))


    return args
