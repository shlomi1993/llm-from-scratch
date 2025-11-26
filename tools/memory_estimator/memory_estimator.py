"""
Unified KV-cache memory estimator for MHA, GQA, MLA, SWA, and MoE FFN.

This tool provides comprehensive memory estimation across multiple attention mechanisms and FFN architectures to help
with model architecture decisions.
"""

from src.common import bytes_convert
from src.estimators import estimate_mha, estimate_gqa, estimate_mla, estimate_swa, estimate_moe
from src.parser import parse_args
from src.mode import Mode
from src.configurations import GptConfig


def main() -> None:
    """
    Command-line interface for unified memory estimation.

    Supports MHA vs GQA, MLA, SWA, and MoE FFN estimation modes.
    """
    args = parse_args()

    for m in args.mode:
        mode = Mode(m)

        if mode == Mode.MHA:
            config = GptConfig(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, context_length=args.context_length)
            result = estimate_mha(config, args.batch_size, args.dtype)
            # Print only MHA portion
            print("==== Config ====")
            for k, v in vars(config).items():
                if v is not None:
                    print(f"{k:23}: {v}")
            print(f"{'batch_size':23}: {args.batch_size}")
            print(f"{'dtype':23}: {args.dtype} ({result.bytes_per_elem} Bytes/elem)")
            print(f"{'head_dim':23}: {result.head_dim}")
            print()
            print("==== KV-cache totals across all layers ====")
            print(f"MHA total KV cache  : {bytes_convert(result.total_mha)}")

        elif mode == Mode.GQA:
            config = GptConfig(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, context_length=args.context_length)
            result = estimate_gqa(config, args.batch_size, args.dtype, args.n_kv_groups)
            result.print(config, args)

        elif mode == Mode.MLA:
            config = GptConfig(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, context_length=args.context_length)
            result = estimate_mla(config, args.batch_size, args.dtype, args.latent_dim, args.n_kv_groups)
            result.print(config, args)

        elif mode == Mode.SWA:
            config = GptConfig(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, context_length=args.context_length)
            result = estimate_swa(config, args.batch_size, args.dtype, args.n_kv_groups, args.sliding_window_size, args.swa_ratio)
            result.print(config, args)

        elif mode == Mode.MOE:
            result = estimate_moe(args.emb_dim, args.hidden_dim, args.ffn_type, args.num_experts, args.top_k, args.dtype, args.match_dense)
            result.print(args)

if __name__ == "__main__":
    main()
