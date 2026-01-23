#!/usr/bin/env python3
"""
Script to compare different models on coding tasks.
Tests assistant model and pretrained model using the same coding test data.
"""
import argparse
import json
import torch

from src.scripts.finetune.instruction import test_assistant
from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.ollama import format_input
from src.dataset import AlpacaCodeDataset


def main():
    parser = argparse.ArgumentParser(
        description="Compare assistant and pretrained models on coding tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="models/coder/coder_results.json",
        help="Path to the coding test data JSON"
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default="models/assistant/assistant.pth",
        help="Path to the assistant model"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="models/355M/model.pth",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set up device
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    g_logger.info(f"Using device '{device.type}' and random seed {args.seed}")

    # Load test data
    g_logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Keep only the necessary fields and create fresh copies
    test_data_clean = [
        {
            "instruction": entry["instruction"],
            "input": entry.get("input", ""),
            "output": entry["output"]
        }
        for entry in test_data
    ]

    g_logger.info(f"Loaded {len(test_data_clean)} test examples")

    # Test 1: Assistant model on coding tasks
    g_logger.info("="*60)
    g_logger.info("Testing ASSISTANT model on coding tasks...")
    g_logger.info("="*60)

    assistant_model, _ = load_model(args.assistant_model, device)
    assistant_test_data = [entry.copy() for entry in test_data_clean]

    test_assistant(
        model=assistant_model,
        test_data=assistant_test_data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        test_output_path="assistant_on_code_results.json",
        format_func=format_input,  # Assistant model uses its own format
        evaluate=False,
        seed=args.seed
    )

    g_logger.info("Assistant model testing complete!")
    del assistant_model  # Free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: Pretrained model on coding tasks
    g_logger.info("="*60)
    g_logger.info("Testing PRETRAINED model on coding tasks...")
    g_logger.info("="*60)

    pretrained_model, _ = load_model(args.pretrained_model, device)
    pretrained_test_data = [entry.copy() for entry in test_data_clean]

    test_assistant(
        model=pretrained_model,
        test_data=pretrained_test_data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        test_output_path="pretrained_on_code_results.json",
        format_func=AlpacaCodeDataset.format_input,
        evaluate=False,
        seed=args.seed
    )

    g_logger.info("Pretrained model testing complete!")
    del pretrained_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Results saved:")
    print(f"  - Assistant model: assistant_on_code_results.json")
    print(f"  - Pretrained model: pretrained_on_code_results.json")
    print(f"  - Coder model: {args.test_data}")
    print("\nNext steps:")
    print("  1. Evaluate assistant model:")
    print("     python evaluate_coder.py --results-file assistant_on_code_results.json --output-file assistant_scores.json")
    print("\n  2. Evaluate pretrained model:")
    print("     python evaluate_coder.py --results-file pretrained_on_code_results.json --output-file pretrained_scores.json")
    print("\n  3. Evaluate coder model:")
    print(f"     python evaluate_coder.py --results-file {args.test_data} --output-file coder_scores.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
