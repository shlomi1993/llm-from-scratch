"""Command-line argument parser for plot_memory_estimates tool."""

from argparse import ArgumentParser, Namespace

from .common import DTYPE_BYTES


def validate_args(args: Namespace) -> None:
    """
    Validate required arguments for each mode.

    Args:
        args (Namespace): Parsed command-line arguments

    Raises:
        ValueError: If required arguments are missing for the selected mode
    """
    if args.mode in ["gqa", "mla", "swa"]:
        required = ["emb_dim", "n_heads", "n_layers"]
        missing = [arg for arg in required if getattr(args, arg) is None]
        if missing:
            missing_args = ', '.join(f'--{arg.replace("_", "-")}' for arg in missing)
            raise ValueError(f"Mode '{args.mode}' requires: {missing_args}")

    if args.mode == "swa":
        if args.sliding_window_size is None:
            raise ValueError("Mode 'swa' requires: --sliding-window-size")

    if args.mode == "moe":
        required = ["emb_dim", "hidden_dim"]
        missing = [arg for arg in required if getattr(args, arg) is None]
        if missing:
            missing_args = ', '.join(f'--{arg.replace("_", "-")}' for arg in missing)
            raise ValueError(f"Mode 'moe' requires: {missing_args}")


def parse_args() -> Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = ArgumentParser(
        description="Unified memory estimation plotting tool for MHA, GQA, MLA, SWA, and MoE"
    )

    # Plot mode selection
    parser.add_argument( "--mode", choices=["gqa", "mla", "swa", "moe"], required=True,
                        help="Plot mode: gqa (MHA vs GQA), mla (MHA vs MLA), swa (SWA overlays), moe (MoE parameters)")

    # Common parameters
    parser.add_argument("--output", type=str, help="Output file path (default: auto-generated)")
    parser.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bfloat16", help="Data type")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # KV-cache parameters (for gqa, mla, swa)
    parser.add_argument("--emb-dim", type=int, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, help="Number of transformer layers")

    # MLA-specific parameters
    parser.add_argument("--latent-dims", type=int, nargs="+", help="Latent dimensions to plot (default: [1024, 512, 256, 64])")

    # SWA-specific parameters
    parser.add_argument("--sliding-window-size", type=int, help="SWA window size W")
    parser.add_argument("--swa-ratio", type=str, default="5:1", help="SWA:Full ratio (e.g., '5:1')")
    parser.add_argument("--n-kv-groups", type=int, help="Number of KV groups for GQA")

    # MoE-specific parameters
    parser.add_argument("--hidden-dim", type=int, help="FFN hidden dimension")
    parser.add_argument("--ffn-type", choices=["gelu", "swiglu"], default="swiglu", help="FFN activation type")
    parser.add_argument("--top-k", type=int, default=2, help="Active experts per token")
    parser.add_argument("--max-experts", type=int, help="Maximum number of experts to plot")
    parser.add_argument("--match-dense", action="store_true", default=True, help="Match MoE total parameters to dense FFN")
    parser.add_argument("--no-match-dense", dest="match_dense", action="store_false", help="Disable matching MoE parameters to dense FFN")
    parser.add_argument("--no-log", action="store_true", help="Disable log-scale y-axis for MoE")

    return parser.parse_args()
