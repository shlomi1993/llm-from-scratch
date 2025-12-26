import argparse
import time
import tiktoken
import torch

from src.configurations import GptConfig
from src.gpt import GptModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic GPT text generation tool with various attention mechanisms")

    # Model architecture parameters
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--emb-dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--drop-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--qkv-bias", action="store_true", help="Use bias in QKV projections")

    # KV-cache parameters
    parser.add_argument("--kv-window-size", type=int, default=None, help="KV cache window size for optimized cache")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default="Hello, I am", help="Starting prompt for text generation")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    # Optimization flags
    parser.add_argument("--use-cache", action="store_true", help="Use KV-cache optimization for generation")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model (bfloat16 recommended for CUDA)")

    # Performance measurement
    parser.add_argument("--measure-time", action="store_true", help="Measure and report generation time")
    parser.add_argument("--measure-memory", action="store_true", help="Measure and report GPU memory usage (CUDA only)")

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select torch data type
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # Build configuration
    config = GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.qkv_bias,
        kv_window_size=args.kv_window_size,
    )

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
    print(f"KV-Cache: {'Enabled' if args.use_cache else 'Disabled'}")

    # Start timing if requested
    if args.measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

    # Generate text using appropriate method
    with torch.no_grad():
        token_ids = model.generate2(encoded_tensor, args.max_new_tokens, config.context_length, args.use_cache)

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
