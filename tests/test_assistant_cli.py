import re
import sys

from pathlib import Path

from tests.common import COLOR_GREEN, COLOR_RESET, run_subprocess, print_title, extract_losses, compare_losses


def validate_evaluation_score(output: str, min_score: float = 90.0) -> None:
    print("Validating evaluation score...", end=" ")
    match = re.search(r'Average score:?\s*([\d.]+)', output, re.IGNORECASE)
    score = float(match.group(1)) if match else 0.0
    assert score > min_score, f"Assistant score is too low: {score:.2f}/100"
    print(f"{COLOR_GREEN}✓ Assistant score: {score:.2f}/100{COLOR_RESET}")


def test_finetune_instruction_cli_vs_script(tmp_path: Path, chapters_path: Path):
    cli_model_path = tmp_path / "test_instruction_model.pth"
    cli_test_output = tmp_path / "instruction-test-responses.json"
    chapter_path = chapters_path / "ch07/01_main-chapter-code/gpt_instruction_finetuning.py"
    pretrained_model_path = "models/355M/model.pth"
    instruction_data_path = "data_sets/instruction_data/instruction-data.json"

    print_title("Running CLI command for test")
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
    cli_metrics = extract_losses(cli_output, ref=False)

    print_title("Running chapter script for reference")
    chapter_cmd = [sys.executable, "-u", str(chapter_path)]
    chapter_output = run_subprocess(chapter_cmd, cwd=tmp_path)
    chapter_metrics = extract_losses(chapter_output, ref=True)

    print_title("Validation")
    compare_losses(actual_losses=cli_metrics, expected_losses=chapter_metrics, tolerance=1e-2)
    validate_evaluation_score(cli_output, min_score=40.0)
