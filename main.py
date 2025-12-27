import argparse
import os
import time
import tiktoken
import torch
import requests

from src.config import GptConfig
from src.gpt import GptModel
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="All-in-one GPT Tool: Train, Generate, or Load Pretrained.")

    # Parent parser for common arguments
    parent = argparse.ArgumentParser(add_help=False)

    # General parameters
    parent.add_argument("--seed", type=int, default=123, help="Random seed")
    parent.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device to run on")

    # Architecture parameters
    parent.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parent.add_argument("--context-length", type=int, default=1024, help="Maximum context length")
    parent.add_argument("--emb-dim", type=int, default=768, help="Embedding dimension")
    parent.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parent.add_argument("--n-layers", type=int, default=12, help="Number of transformer layers")
    parent.add_argument("--drop-rate", type=float, default=0.1, help="Dropout rate")
    parent.add_argument("--qkv-bias", action="store_true", help="Use bias in QKV projections")

    # Optimization flags
    parent.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model")
    parent.add_argument("--kv-window-size", type=int, default=None, help="KV cache window size for optimized cache")

    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", parents=[parent], help="Train a new model from scratch")
    train_parser.add_argument("--data-file", type=str, default="the-verdict.txt", help="Path to training text file")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    train_parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for optimizer")
    train_parser.add_argument("--save-path", type=str, default="gpt2.pth", help="Where to save the model")
    train_parser.add_argument("--plot-path", type=str, default="loss.pdf", help="Where to save loss plot")

    # Gen subcommand
    gen_parser = subparsers.add_parser("generate", parents=[parent], help="Generate text using a local model")
    gen_parser.add_argument("--prompt", type=str, default="Hello, I am", help="Input prompt")
    gen_parser.add_argument("--max-new-tokens", type=int, default=25, help="Number of new tokens to generate")
    gen_parser.add_argument("--use-cache", action="store_true", help="Use KV-cache optimization")
    gen_parser.add_argument("--measure-time", action="store_true", help="Measure generation time")
    gen_parser.add_argument("--measure-memory", action="store_true", help="Measure memory usage (GPU only)")

    # Pretrained subcommand
    pretrained_parser = subparsers.add_parser("download", parents=[parent], help="Generate text with a pretrained OpenAI GPT-2 model")
    
    
    pretrained_parser.add_argument("--max-new-tokens", type=int, default=25, help="Number of new tokens to generate")
    pretrained_parser.add_argument("--use-cache", action="store_true", help="Use KV-cache optimization")
    pretrained_parser.add_argument("--measure-time", action="store_true", help="Measure generation time")
    pretrained_parser.add_argument("--measure-memory", action="store_true", help="Measure memory usage (GPU only)")

    return parser.parse_args()


def download_data_if_needed(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
    else:
        print(f"Found {file_path}, skipping download.")


def train(args: argparse.Namespace) -> None:
    pass


def generate(args: argparse.Namespace) -> None:
    pass





def main() -> None:

    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "gen":
        generate(args)
    elif args.mode == "pretrained":
        load(args)

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
        token_ids = model.generate_cached(encoded_tensor, args.max_new_tokens, config.context_length, args.use_cache)

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


# TODO Improve the following tests
# test_evaluate_model_runs
# test_generate_and_print_sample_runs
# test_train_model_runs
# And the whole test_main.py module


### TODO GPT TRAIN - TO INTEGRATE IN THE MAIN ABOVE ###
def train(config: GptConfig, learning_rate: float, n_epochs: int, batch_size: int, weight_decay: float):

    device = get_device()

#     # Download data if necessary
#     file_path = "the-verdict.txt"
#     if not os.path.exists(file_path):
#         url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#         text_data = response.text
#         with open(file_path, "w", encoding="utf-8") as file:
#             file.write(text_data)
#     else:
#         with open(file_path, "r", encoding="utf-8") as file:
#             text_data = file.read()

#     # Initialize model
#     model = GptModel(config)
#     model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Set up dataloaders
#     train_loader, val_loader = train_test_split(text_data, config.context_length, batch_size, train_ratio=0.90)

#     # Train model
#     tokenizer = tiktoken.get_encoding("gpt2")
#     train_losses, val_losses, tokens_seen = train_model_simple(
#         model, train_loader, val_loader, optimizer, device,
#         n_epochs=n_epochs, eval_freq=5, eval_iter=1,
#         start_context="Every effort moves you", tokenizer=tokenizer
#     )

#     return train_losses, val_losses, tokens_seen, model


# if __name__ == "__main__":

#     GPT_CONFIG_124M_256CONT = {
#         "vocab_size": 50257,
#         "context_length": 256,
#         "emb_dim": 768,
#         "n_heads": 12,
#         "n_layers": 12,
#         "drop_rate": 0.1,
#         "qkv_bias": False
#     }

#     # Training
#     n_epochs = 10
#     train_losses, val_losses, tokens_seen, model = main(
#         GPT_CONFIG_124M_256CONT,
#         learning_rate=5e-4,
#         n_epochs=n_epochs,
#         batch_size=2,
#         weight_decay=0.1
#     )

#     # Plot results
#     epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
#     plt.savefig("loss.pdf")

#     # Save and load model
#     torch.save(model.state_dict(), "model.pth")
#     model = GptModel(GPT_CONFIG_124M_256CONT)
#     model.load_state_dict(torch.load("model.pth", weights_only=True))


### TODO GPT GENERATE - TO INTEGRATE IN THE MAIN ABOVE ###

# def main():
#     parser = argparse.ArgumentParser(description="Generate text with a pretrained GPT-2 model.")
#     parser.add_argument("--model-size", default="124M", choices=["124M", "355M", "774M", "1558M"], help="Size of the GPT-2 model to use.")
#     parser.add_argument("--prompt", required=True, help="Prompt text used to seed the generation.")
#     parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device for running inference, e.g., cpu, cuda, mps, or auto.")
#     parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
#     args = parser.parse_args()

#     device = torch.device(args.device) if args.device != "auto" else get_device()
#     torch.manual_seed(args.seed)
#     print("PyTorch:", torch.__version__)
#     print("Device:", device)
#     print("Model size:", args.model_size)
#     print("Seed:", args.seed)
#     print("Prompt:", args.prompt)

#     config = {
#         "124M": GPT_CONFIG_124M,
#         "355M": GPT_CONFIG_355M,
#         "774M": GPT_CONFIG_774M,
#         "1558M": GPT_CONFIG_1558M
#     }[args.model_size]

#     _, params = download_and_load_gpt2(model_size=args.model_size, models_dir="gpt2")

#     gpt = load_weights_into_gpt(config, params)
#     gpt.to(device)
#     gpt.eval()

#     tokenizer = tiktoken.get_encoding("gpt2")

#     token_ids = gpt.generate(
#         idx=text_to_token_ids(args.prompt, tokenizer).to(device),
#         max_new_tokens=25,
#         context_size=config.context_length,
#         top_k=50,
#         temperature=1.0
#     )

#     print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# if __name__ == "__main__":
#     main()


### TODO TEST FOR MAIN


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

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
