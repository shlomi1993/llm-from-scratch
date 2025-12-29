import argparse
import tiktoken

from src.config import GptConfig
from src.utils import get_device
from src.gpt_utils import run_model_training_flow, run_model_generation_flow, run_model_interactive_flow, download_gpt2


class InputError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT Demonstration Tool")

    # Parent parser for common arguments
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--seed", type=int, default=123, help="Random seed")
    parent.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device to run on")
    parent.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model")

    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Train subcommand - architecture and training args
    train_parser = subparsers.add_parser("train", parents=[parent], help="Train a new model from scratch")
    train_parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    train_parser.add_argument("--context-length", type=int, default=1024, help="Maximum context length")
    train_parser.add_argument("--emb-dim", type=int, default=768, help="Embedding dimension")
    train_parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    train_parser.add_argument("--n-layers", type=int, default=12, help="Number of transformer layers")
    train_parser.add_argument("--drop-rate", type=float, default=0.1, help="Dropout rate")
    train_parser.add_argument("--qkv-bias", action="store_true", help="Use bias in QKV projections")
    train_parser.add_argument("--kv-window-size", type=int, default=None, help="KV cache window size for optimized cache")
    train_parser.add_argument("--data-file", type=str, default="the-verdict.txt", help="Path to training text file")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    train_parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for optimizer")
    train_parser.add_argument("--save-path", type=str, default="gpt2.pth", help="Where to save the model")
    train_parser.add_argument("--plot-path", type=str, default="loss.pdf", help="Where to save loss plot")

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", parents=[parent], help="Generate text using a local (trained or downloaded) model")
    gen_parser.add_argument("--prompt", type=str, default="Hello, I am", help="Input prompt")
    gen_parser.add_argument("--max-new-tokens", type=int, default=25, help="Number of new tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--models-dir", type=str, default="gpt2", help="Directory for model weights")
    gen_parser.add_argument("--model-size", type=str, default="124M", choices=["124M", "355M", "774M", "1558M"], help="Model size (for pretrained)")
    gen_parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt mode")
    gen_parser.add_argument("--use-cache", action="store_true", help="Use KV-cache optimization")
    gen_parser.add_argument("--measure-time", action="store_true", help="Measure generation time (non-interactive only)")
    gen_parser.add_argument("--measure-memory", action="store_true", help="Measure memory usage (GPU only, non-interactive only)")

    # Download subcommand
    download_parser = subparsers.add_parser("download", parents=[parent], help="Download a pretrained GPT-2 model")
    download_parser.add_argument("--model-size", type=str, default="124M", choices=["124M", "355M", "774M", "1558M"], help="Size of the GPT-2 model to download")
    download_parser.add_argument("--models-dir", type=str, default="gpt2", help="Directory for model weights")

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
            lr=args.lr,
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
        download_gpt2(model_size=args.model_size)
    else:
        raise InputError(f"Unknown command {args.command}")

    print("Done.")


if __name__ == "__main__":
    main()




# TODO Improve the tests: test_evaluate_model_runs, test_generate_and_print_sample_runs, test_train_model_runs
# TODO Add tests for run_model_training_flow, run_model_generation_flow, run_model_interactive_flow (or simply to main.py)

# TODO Add the tests below:

# import pytest
# from gpt_train import main
# import requests

# @pytest.fixture
# def gpt_config():
#     return {
#         "vocab_size": 50257,
#         "context_length": 12,  # small for testing efficiency
#         "emb_dim": 32,         # small for testing efficiency
#         "n_heads": 4,          # small for testing efficiency
#         "n_layers": 2,         # small for testing efficiency
#         "drop_rate": 0.1,
#         "qkv_bias": False
#     }


# @pytest.fixture
# def other_settings():
#     return {
#         "learning_rate": 5e-4,
#         "num_epochs": 1,    # small for testing efficiency
#         "batch_size": 2,
#         "weight_decay": 0.1
#     }


# def test_main(gpt_config, other_settings):
#     train_losses, val_losses, tokens_seen, model = main(gpt_config, other_settings)

#     assert len(train_losses) == 39, "Unexpected number of training losses"
#     assert len(val_losses) == 39, "Unexpected number of validation losses"
#     assert len(tokens_seen) == 39, "Unexpected number of tokens seen"


# def check_file_size(url, expected_size):
#     try:
#         response = requests.head(url, allow_redirects=True, timeout=30)
#         if response.status_code != 200:
#             return False, f"{url} not accessible"

#         size = response.headers.get("Content-Length")
#         if size is None:
#             return False, "Content-Length header is missing"

#         size = int(size)
#         if size != expected_size:
#             return False, f"{url} file has expected size {expected_size}, but got {size}"

#         return True, f"{url} file size is correct"

#     except requests.exceptions.RequestException as e:
#         return False, f"Failed to access {url}: {e}"


# def test_model_files():
#     def check_model_files(base_url):

#         model_size = "124M"
#         files = {
#             "checkpoint": 77,
#             "encoder.json": 1042301,
#             "hparams.json": 90,
#             "model.ckpt.data-00000-of-00001": 497759232,
#             "model.ckpt.index": 5215,
#             "model.ckpt.meta": 471155,
#             "vocab.bpe": 456318
#         }

#         for file_name, expected_size in files.items():
#             url = f"{base_url}/{model_size}/{file_name}"
#             valid, message = check_file_size(url, expected_size)
#             assert valid, message

#         model_size = "355M"
#         files = {
#             "checkpoint": 77,
#             "encoder.json": 1042301,
#             "hparams.json": 91,
#             "model.ckpt.data-00000-of-00001": 1419292672,
#             "model.ckpt.index": 10399,
#             "model.ckpt.meta": 926519,
#             "vocab.bpe": 456318
#         }

#         for file_name, expected_size in files.items():
#             url = f"{base_url}/{model_size}/{file_name}"
#             valid, message = check_file_size(url, expected_size)
#             assert valid, message

#     check_model_files(base_url="https://openaipublic.blob.core.windows.net/gpt-2/models")
#     check_model_files(base_url="https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2")
