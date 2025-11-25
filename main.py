"""
Generic GPT text generation tool supporting various attention mechanisms and optimizations.
"""

import argparse
import time
import tiktoken
import torch

from src.configurations import GptConfig
from src.gpt import GptModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic GPT text generation tool with various attention mechanisms"
    )

    # Model architecture parameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--qkv_bias", action="store_true", help="Use bias in QKV projections")

    # Grouped Query Attention parameters
    parser.add_argument("--n_kv_groups", type=int, default=None,
                        help="Number of key/value groups for GQA (enables GQA)")

    # KV-cache parameters
    parser.add_argument("--kv_window_size", type=int, default=None,
                        help="KV cache window size for optimized cache")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default="Hello, I am",
                        help="Starting prompt for text generation")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    # Optimization flags
    parser.add_argument("--use_cache", action="store_true",
                        help="Use KV-cache optimization for generation")
    parser.add_argument("--use_softmax", action="store_true",
                        help="Use softmax generation instead of simple argmax")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for model (bfloat16 recommended for CUDA)")

    # Performance measurement
    parser.add_argument("--measure_time", action="store_true",
                        help="Measure and report generation time")
    parser.add_argument("--measure_memory", action="store_true",
                        help="Measure and report GPU memory usage (CUDA only)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Build configuration
    config_dict = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }

    # Add optional parameters if specified
    if args.n_kv_groups is not None:
        config_dict["n_kv_groups"] = args.n_kv_groups
        # Validate n_kv_groups
        if args.n_heads % args.n_kv_groups != 0:
            raise ValueError("n_kv_groups must divide n_heads exactly")

    if args.kv_window_size is not None:
        config_dict["kv_window_size"] = args.kv_window_size

    config = GptConfig(**config_dict)

    # Initialize model
    model = GptModel(config)
    model.to(device, dtype=dtype)
    model.eval()  # Disable dropout

    # Prepare input
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(args.prompt)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    # Print input information
    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print(f"\nInput text: {args.prompt}")
    print(f"Encoded input text: {encoded}")
    print(f"encoded_tensor.shape: {encoded_tensor.shape}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    if args.n_kv_groups is not None:
        print(f"Using GQA with {args.n_kv_groups} key/value groups")
    if args.use_cache:
        print("Using KV-cache optimization")

    # Start timing if requested
    if args.measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

    # Generate text using appropriate method
    with torch.no_grad():
        if args.use_cache:
            # Use cached generation
            token_ids = model.generate_text_simple_cached(
                idx=encoded_tensor,
                max_new_tokens=args.max_new_tokens,
                context_size=config.context_length,
                use_cache=True
            )
        elif args.use_softmax:
            # Use softmax-based generation
            token_ids = model.generate_text_softmax(
                idx=encoded_tensor,
                max_new_tokens=args.max_new_tokens,
                context_size=config.context_length
            )
        else:
            # Use simple generation
            token_ids = model.generate_text_simple(
                idx=encoded_tensor,
                max_new_tokens=args.max_new_tokens,
                context_size=config.context_length
            )

    # End timing if requested
    if args.measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time

    # Decode output
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    # Print output information
    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print(f"\nOutput: {token_ids}")
    print(f"Output length: {len(token_ids[0])}")
    print(f"Output text: {decoded_text}")

    # Print performance metrics if requested
    if args.measure_time:
        print(f"\nTime: {total_time:.2f} sec")
        tokens_per_sec = int(len(token_ids[0]) / total_time)
        print(f"{tokens_per_sec} tokens/sec")

    if args.measure_memory and torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
