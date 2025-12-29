import argparse
import tiktoken

from src.config import GptConfig
from src.gpt_utils import run_model_training_flow, run_model_generation_flow, run_model_interactive_flow, download_gpt2


class InputError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT Demonstration Tool")

    # Parent parser for common arguments
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-d", "--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device to run the model on (cpu, cuda, mps, or auto). Default: auto.")
    parent.add_argument("-t", "--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model parameters (float32, float16, bfloat16). Default: float32.")
    parent.add_argument("-s", "--seed", type=int, default=123, help="Random seed for reproducibility. Default: 123.")

    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Train subcommand - architecture and training args
    train_parser = subparsers.add_parser("train", parents=[parent], help="Train a new model from scratch.")
    train_parser.add_argument("-b", "--batch-size", type=int, default=2, help="Training batch size. Default: 2.")
    train_parser.add_argument("-c", "--context-length", type=int, default=1024, help="Maximum context length for the model. Default: 1024.")
    train_parser.add_argument("-f", "--data-file", type=str, default="the-verdict.txt", help="Path to the training text file. Default: the-verdict.txt.")
    train_parser.add_argument("-e", "--emb-dim", type=int, default=768, help="Embedding dimension size. Default: 768.")
    train_parser.add_argument("-k", "--kv-window-size", type=int, default=None, help="KV cache window size for optimized cache. Default: None.")
    train_parser.add_argument("-l", "--learning-rate", type=float, default=5e-4, help="Learning rate for optimizer. Default: 5e-4.")
    train_parser.add_argument("-T", "--n-epochs", type=int, default=10, help="Number of training epochs. Default: 10.")
    train_parser.add_argument("-H", "--n-heads", type=int, default=12, help="Number of attention heads. Default: 12.")
    train_parser.add_argument("-L", "--n-layers", type=int, default=12, help="Number of transformer layers. Default: 12.")
    train_parser.add_argument("-o", "--drop-rate", type=float, default=0.1, help="Dropout rate for regularization. Default: 0.1.")
    train_parser.add_argument("-p", "--plot-path", type=str, default="loss.pdf", help="Path to save the loss plot. Default: loss.pdf.")
    train_parser.add_argument("-q", "--qkv-bias", action="store_true", help="Enable bias in QKV projections.")
    train_parser.add_argument("-S", "--save-path", type=str, default="gpt2.pth", help="Path to save the trained model. Default: gpt2.pth.")
    train_parser.add_argument("-v", "--vocab-size", type=int, default=50257, help="Vocabulary size for the tokenizer. Default: 50257.")
    train_parser.add_argument("-w", "--weight-decay", type=float, default=0.1, help="Weight decay for optimizer. Default: 0.1.")

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", parents=[parent], help="Generate text using a local (trained or downloaded) model.")
    gen_parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive prompt mode.")
    gen_parser.add_argument("-m", "--max-new-tokens", type=int, default=25, help="Number of new tokens to generate. Default: 25.")
    gen_parser.add_argument("-M", "--model-size", type=str, default="124M", choices=["124M", "355M", "774M", "1558M"], help="Model size to use (for pretrained models). Default: 124M.")
    gen_parser.add_argument("-D", "--models-dir", type=str, default="gpt2", help="Directory containing model weights. Default: gpt2.")
    gen_parser.add_argument("-y", "--measure-memory", action="store_true", help="Measure memory usage (GPU only, non-interactive mode).")
    gen_parser.add_argument("-T", "--measure-time", action="store_true", help="Measure generation time (non-interactive mode).")
    gen_parser.add_argument("-p", "--prompt", type=str, default="Hello, I am", help="Input prompt for generation. Default: 'Hello, I am'.")
    gen_parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature for generation. Default: 1.0.")
    gen_parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling for generation. Default: 50.")
    gen_parser.add_argument("-u", "--use-cache", action="store_true", help="Enable KV-cache optimization during generation.")

    # Download subcommand
    download_parser = subparsers.add_parser("download", parents=[parent], help="Download a pretrained GPT-2 model.")
    download_parser.add_argument("-M", "--model-size", type=str, default="124M", choices=["124M", "355M", "774M", "1558M"], help="Size of the GPT-2 model to download. Default: 124M.")
    download_parser.add_argument("-D", "--models-dir", type=str, default="gpt2", help="Directory to save downloaded model weights. Default: gpt2.")

    return parser.parse_args()


def args_to_gpt_config(args: argparse.Namespace) -> GptConfig:
    return GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.qkv_bias,
        kv_window_size=args.kv_window_size,
    )


def main() -> None:

    args = parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")

    if args.command == "train":
        run_model_training_flow(
            config=args_to_gpt_config(args),
            training_set_path=args.data_file,
            tokenizer=tokenizer,
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            saved_model_path=args.save_path,
            saved_plot_path=args.plot_path
        )
    elif args.command == "generate":
        if getattr(args, "interactive", False):
            if getattr(args, "measure_time", False) or getattr(args, "measure_memory", False):
                raise InputError("--measure-time and --measure-memory are only allowed in non-interactive mode.")
            run_model_interactive_flow(
                config=args_to_gpt_config(args),
                models_dir=args.models_dir,
                model_size=args.model_size,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
                seed=args.seed
            )
        else:
            run_model_generation_flow(
                config=args_to_gpt_config(args),
                prompt=args.prompt,
                models_dir=args.models_dir,
                model_size=args.model_size,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
                seed=args.seed,
                measure_time=args.measure_time,
                measure_memory=args.measure_memory
            )
    elif args.command == "download":
        download_gpt2(model_size=args.model_size, models_dir=args.models_dir)
    else:
        raise InputError(f"Unknown command {args.command}")

    print("Done.")


if __name__ == "__main__":
    main()
