import argparse

from dataclasses import dataclass


@dataclass(frozen=True)
class GptConfig:
    emb_dim: int                        # Embedding dimension
    n_layers: int                       # Number of layers
    n_heads: int                        # Number of attention heads
    vocab_size: int = 50257             # Vocabulary size
    context_length: int = 1024          # Context length
    drop_rate: float = 0.1              # Dropout rate
    qkv_bias: bool = False              # Query-Key-Value bias
    kv_window_size: int = None          # KV cache window size


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--emb-dim", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--n-heads", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size.")
    parser.add_argument("--context-length", type=int, default=1024, help="Context length.")
    parser.add_argument("--drop-rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--qkv-bias", action="store_true", help="Use bias in QKV projections.")
    parser.add_argument("--kv-window-size", type=int, default=None, help="KV cache window size.")


# GPT2-Small (124M parameters)
GPT_CONFIG_124M = GptConfig(
    emb_dim=768,
    n_layers=12,
    n_heads=12
)


# GPT2-Medium (355M parameters)
GPT_CONFIG_355M = GptConfig(
    emb_dim=1024,
    n_layers=24,
    n_heads=16
)


# GPT2-Large (774M parameters)
GPT_CONFIG_774M = GptConfig(
    emb_dim=1280,
    n_layers=36,
    n_heads=20
)


# GPT2-XL (1558M parameters)
GPT_CONFIG_1558M = GptConfig(
    emb_dim=1600,
    n_layers=48,
    n_heads=25
)
