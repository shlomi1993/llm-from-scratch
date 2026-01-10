import os
import re
import sys

from pathlib import Path

from tests.common import run_subprocess, print_title, compare_losses


def extract_losses(output: str) -> dict:
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'steps': []
    }

    # Pattern for chapter script: "Ep 1 (Step 000000): Train loss 9.123, Val loss 8.456"
    pattern1 = r'Ep \d+ \(Step (\d+)\): Train loss ([\d.]+), Val loss ([\d.]+)'

    # Pattern for CLI: "  Step 000000: Train loss 9.123, Val loss 8.456"
    pattern2 = r'\s+Step (\d+): Train loss ([\d.]+), Val loss ([\d.]+)'

    # Try both patterns
    for pattern in [pattern1, pattern2]:
        matches = re.findall(pattern, output)
        if matches:
            for step, train_loss, val_loss in matches:
                metrics['steps'].append(int(step))
                metrics['train_losses'].append(float(train_loss))
                metrics['val_losses'].append(float(val_loss))
            break  # Stop after finding matches with one pattern

    return metrics


def test_pretrain_cli_vs_script(tmp_path: Path, chapters_path: Path):

    # Paths
    chapter_path = chapters_path / "ch05/01_main-chapter-code/gpt_train.py"
    cli_model_path = tmp_path / "test_model_cli.pth"
    training_file = "datasets/the-verdict.txt"
    assert os.path.exists(training_file), f"Training file {training_file} not found"

    # Run the chapter script with live output
    print_title("Running chapter script for reference")
    chapter_cmd = [sys.executable, "-u", str(chapter_path)]
    chapter_output = run_subprocess(chapter_cmd, cwd=tmp_path)
    chapter_losses = extract_losses(chapter_output)

    # Run the CLI pretrain command with live output
    print_title("Running CLI command to test")
    cli_cmd = [  # Parameters must match chapter script for deterministic results
        "gpt2", "pretrain",
        "--training-set-path", training_file,
        "--n-epochs", "10",
        "--batch-size", "2",
        "--lr", "5e-4",
        "--weight-decay", "0.1",
        "--seed", "123",
        "--device", "cpu",
        "--max-length", "256",
        "--eval-freq", "5",
        "--eval-iter", "1",
        "--saved-model-path", str(cli_model_path),
        "--context-length", "256",
        "--emb-dim", "768",
        "--n-layers", "12",
        "--n-heads", "12",
        "--vocab-size", "50257",
        "--drop-rate", "0.1"
        # qkv_bias defaults to False, so we don't need to pass --use-qkv-bias
    ]
    cli_output = run_subprocess(cli_cmd)
    cli_losses = extract_losses(cli_output)

    # Validation
    compare_losses(actual_losses=cli_losses, expected_losses=chapter_losses, tolerance=1e-5)
