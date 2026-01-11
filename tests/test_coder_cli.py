import re

from pathlib import Path

from tests.common import COLOR_GREEN, COLOR_RESET, run_subprocess, print_title


def extract_training_metrics(output: str) -> dict:
    """
    Parses the stdout to find 'Step X: Train loss Y, Val loss Z' patterns.
    """
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'steps': []
    }
    # Matches: "Step 10: Train loss 2.4532, Val loss 2.1123"
    loss_pattern = r'Step (\d+)\)?: Train loss ([\d.]+), Val loss ([\d.]+)'
    loss_matches = re.findall(loss_pattern, output)

    for step, train_loss, val_loss in loss_matches:
        metrics['steps'].append(int(step))
        metrics['train_losses'].append(float(train_loss))
        metrics['val_losses'].append(float(val_loss))

    return metrics


def validate_loss_sanity(metrics: dict) -> None:
    """
    Verifies that the loss behavior is logical without a reference script.
    Checks for: Data existence, No NaNs, and non-exploding values.
    """
    print_title("Validating loss logic")

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
    Parses the Ollama evaluation score. Matches log: "Average Score: 85.50"
    """
    print_title("Validating evaluation score")

    match = re.search(r'Average Score:\s*([\d.]+)', output)
    if not match:
        raise ValueError("Could not find 'Average Score:' in output. Did evaluation run?")

    score = float(match.group(1))
    assert score > min_score, f"Coder score is too low: {score:.2f}/100 (Threshold: {min_score})"

    print(f"{COLOR_GREEN}✓ Coder score: {score:.2f}/100{COLOR_RESET}")


def test_finetune_coding_cli(tmp_path: Path):
    """
    Tests the 'gpt2 finetune coding' command flow.
    """
    # Paths
    cli_model_path = tmp_path / "test_coder_model.pth"
    cli_test_output = tmp_path / "coder_results.json"
    pretrained_model_path = "models/355M/model.pth"
    dataset_path = "data_sets/python_code_instructions"

    # CLI Command
    print_title("Running 'gpt2 finetune coding' CLI test")

    cli_cmd = [
        "gpt2", "finetune", "coding",
        "--pretrained-model-path", pretrained_model_path,
        "--dataset-path", dataset_path,
        "--batch-size", "2",
        "--seed", "123",
        "--device", "cpu",          # Force CPU for CI/Test consistency
        "--lr", "5e-5",
        "--n-epochs", "1",          # 1 Epoch is enough for a flow test
        "--max-samples", "20",      # TODO CRITICAL: Limit dataset size for speed
        "--eval-freq", "2",         # Frequent eval to generate logs
        "--eval-iter", "1",
        "--model-save-path", str(cli_model_path),
        "--test-output-path", str(cli_test_output),
        "--evaluate"                # Triggers Ollama
    ]

    # Execute
    output = run_subprocess(cli_cmd)
    metrics = extract_training_metrics(output)

    # Validation
    validate_loss_sanity(metrics)
    validate_evaluation_score(output, min_score=40.0)
