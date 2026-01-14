import re

from pathlib import Path

from tests.common import COLOR_GREEN, COLOR_RESET, run_subprocess, extract_losses


def validate_loss_sanity(metrics: dict) -> None:
    """
    Verifies that the loss behavior is logical without a reference script.
    Checks for: Data existence, No NaNs, and non-exploding values.
    """
    print("Validating loss logic...", end=" ")

    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']

    # Check if we actually trained
    assert len(train_losses) > 0, "No training steps recorded in output."
    assert len(val_losses) > 0, "No validation steps recorded in output."

    # Check for NaNs (instability)
    assert all(l == l for l in train_losses), "NaN values detected in training loss!"
    assert all(l == l for l in val_losses), "NaN values detected in validation loss!"

    # Check for exploding gradients (sanity check for 124M/355M models) Loss > 15 means the model is broken or input is garbage
    final_loss = train_losses[-1]
    assert final_loss < 15.0, f"Final loss is suspiciously high: {final_loss}"

    print(f"{COLOR_GREEN}✓ Loss metrics detected and valid (Final Train: {final_loss:.4f}){COLOR_RESET}")


def validate_evaluation_score(output: str, min_score: float = 40.0) -> None:
    """
    Parses the Ollama evaluation score. Matches log: "Average score 49.90% across 10 samples" or "Average score: 85.50"
    """
    print("Validating evaluation score...", end=" ")

    match = re.search(r'Average score:?\s*([\d.]+)', output, re.IGNORECASE)
    if not match:
        raise ValueError("Could not find 'Average score' in output. Did evaluation run?")

    score = float(match.group(1))
    assert score > min_score, f"Coder score is too low: {score:.2f}/100 (Threshold: {min_score})"

    print(f"{COLOR_GREEN}✓ Coder score: {score:.2f}/100{COLOR_RESET}")


def test_finetune_coding_cli(tmp_path: Path):
    cli_cmd = [
        "gpt2", "finetune", "coding",
        "--pretrained-model-path", "models/355M/model.pth",
        "--dataset-path", "dataset/python_code_instructions",
        "--batch-size", "2",
        "--seed", "123",
        "--device", "cpu",
        "--lr", "5e-5",
        "--n-epochs", "1",
        "--max-samples", "100",  # Limit dataset size for test speed
        "--train-frac", "0.8",
        "--eval-freq", "2",
        "--eval-iter", "1",
        "--model-save-path", str(tmp_path / "test_coder_model.pth"),
        "--test-output-path", str(tmp_path / "coder_results.json"),
        "--evaluate"
    ]
    output = run_subprocess(cli_cmd)
    metrics = extract_losses(output, ref=False)
    validate_loss_sanity(metrics)
    validate_evaluation_score(output, min_score=40.0)
