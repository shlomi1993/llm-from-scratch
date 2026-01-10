import re
import sys

from pathlib import Path

from tests.common import COLOR_GREEN, COLOR_RESET, run_subprocess, print_title, compare_losses


def extract_training_metrics(output: str) -> dict:
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'steps': []
    }
    loss_pattern = r'Step (\d+)\)?: Train loss ([\d.]+), Val loss ([\d.]+)'
    loss_matches = re.findall(loss_pattern, output)
    for step, train_loss, val_loss in loss_matches:
        metrics['steps'].append(int(step))
        metrics['train_losses'].append(float(train_loss))
        metrics['val_losses'].append(float(val_loss))
    return metrics


def validate_evaluation_score(output: str, min_score: float = 90.0) -> None:
    print_title("Validating evaluation score")
    match = re.search(r'Average score:\s*([\d.]+)', output)
    score = float(match.group(1)) if match else 0.0
    assert score > min_score, f"Assistant score is too low: {score:.2f}/100"
    print(f"{COLOR_GREEN}✓ Assistant score: {score:.2f}/100{COLOR_RESET}")


def test_finetune_instruction_cli_vs_script(tmp_path: Path, chapters_path: Path):

    # Paths
    cli_model_path = tmp_path / "test_instruction_model.pth"
    cli_test_output = tmp_path / "instruction-test-responses.json"
    chapter_path = chapters_path / "ch07/01_main-chapter-code/gpt_instruction_finetuning.py"
    pretrained_model_path = "models/355M/model.pth"
    instruction_data_path = "datasets/instruction_data/instruction-data.json"

    # Run the CLI finetune instruction command with live output
    print_title("Running CLI command to test")
    cli_cmd = [
        "gpt2", "finetune", "instruction",
        "--pretrained-model-path", pretrained_model_path,
        "--tuning-set-path", instruction_data_path,
        "--train-frac", "0.85",
        "--test-frac", "0.1",
        "--batch-size", "8",
        "--seed", "123",
        "--device", "cpu",
        "--lr", "5e-5",
        "--n-epochs", "2",
        "--weight-decay", "0.1",
        "--eval-freq", "5",
        "--eval-iter", "5",
        "--loss-plot-save-path", str(tmp_path / "instruction_loss_plot.png"),
        "--model-save-path", cli_model_path,
        "--test-output-path", cli_test_output,
        "--evaluate"
    ]
    cli_output = run_subprocess(cli_cmd)
    cli_metrics = extract_training_metrics(cli_output)

    # Run the chapter script with live output
    print_title("Running chapter script for reference")
    chapter_cmd = [sys.executable, "-u", str(chapter_path)]
    chapter_output = run_subprocess(chapter_cmd, cwd=tmp_path)
    chapter_metrics = extract_training_metrics(chapter_output)

    # Validation
    compare_losses(actual_losses=cli_metrics, expected_losses=chapter_metrics, tolerance=1e-2)
    validate_evaluation_score(cli_output, min_score=40.0)
