import os
import psutil
import re
import requests
import sys

from tests.common import run_subprocess


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


def test_finetune_instruction_cli_vs_script():

    # Paths
    cli_model_path = "tests/test_instruction_model.pth"
    cli_test_output = "tests/instruction-test-responses.json"
    script_path = "../chapters/ch07/01_main-chapter-code/gpt_instruction_finetuning.py"
    pretrained_model_path = "models/124M/model.pth"
    instruction_data_path = "datasets/instruction_data/instruction-data.json"

    # Make sure instruction data exists
    if not os.path.exists(instruction_data_path):
        print(f"Instruction data not found at {instruction_data_path}")
        print("Downloading instruction data...")
        os.makedirs(os.path.dirname(instruction_data_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(instruction_data_path, 'w') as f:
            f.write(response.text)

    try:
        # Run the chapter script with live output
        print("\n" + "=" * 80 + "\nRunning chapter script\n" + "=" * 80)
        chapter_cmd = [sys.executable, "-u", script_path]
        script_output = run_subprocess(chapter_cmd, cwd="tests")
        script_metrics = extract_training_metrics(script_output)

        # Run the CLI finetune instruction command with live output
        print("\n" + "=" * 80 + "\nRunning CLI command\n" + "=" * 80)
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
            "--model-save-path", cli_model_path,
            "--test-output-path", cli_test_output,
            "--evaluate"
        ]
        cli_output = run_subprocess(cli_cmd)
        cli_metrics = extract_training_metrics(cli_output)

        # Compare losses
        print("\n" + "=" * 80 + "\nComparing training metrics\n" + "=" * 80)
        loss_checkpoints = zip(
            script_metrics['train_losses'],
            cli_metrics['train_losses'],
            script_metrics['val_losses'],
            cli_metrics['val_losses']
        )
        tolerance = 1e-2
        for i, (scr_train_loss, cli_train_loss, scr_val_loss, cli_val_loss) in enumerate(loss_checkpoints):
            train_diff = abs(scr_train_loss - cli_train_loss)
            val_diff = abs(scr_val_loss - cli_val_loss)
            if train_diff > tolerance or val_diff > tolerance:
                print(f"Loss checkpoint {i + 1}:")
                print(f"  Script - Train: {scr_train_loss:.6f}, Val: {scr_val_loss:.6f}")
                print(f"  CLI    - Train: {cli_train_loss:.6f}, Val: {cli_val_loss:.6f}")
                print(f"  Diff   - Train: {train_diff:.2e}, Val: {val_diff:.2e}")
                assert False, f"Training losses differ at checkpoint {i + 1}"
        print("✓ All training metrics match!")

        # Validate evaluation score
        match = re.search(r'Average score:\s*([\d.]+)', script_output)
        score = float(match.group(1)) if match else 0.0
        assert score > 90, f"Assistant score is too low: {score:.2f}/100"

    finally:
        cleanup_files = [
            cli_model_path,
            cli_test_output,
            "tests/instruction-data.json",
            "tests/instruction-data-with-response.json",
            "tests/loss-plot-standalone.pdf"
        ]
        for f in cleanup_files:
            if os.path.exists(f):
                os.remove(f)
