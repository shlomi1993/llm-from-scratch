import os
import sys

from pathlib import Path

from tests.common import run_subprocess, print_title, extract_losses, compare_losses


def test_pretrain_cli_vs_script(tmp_path: Path, chapters_path: Path):
    chapter_path = chapters_path / "ch05/01_main-chapter-code/gpt_train.py"
    cli_model_path = tmp_path / "test_model_cli.pth"
    training_file = "dataset/the-verdict.txt"
    assert os.path.exists(training_file), f"Training file {training_file} not found"

    print_title("Running CLI command for test")
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
    cli_losses = extract_losses(cli_output, ref=False)

    print_title("Running chapter script for reference")
    chapter_cmd = [sys.executable, "-u", str(chapter_path)]
    chapter_output = run_subprocess(chapter_cmd, cwd=tmp_path)
    chapter_losses = extract_losses(chapter_output, ref=True)

    print_title("Validation")
    compare_losses(actual_losses=cli_losses, expected_losses=chapter_losses, tolerance=1e-5)
