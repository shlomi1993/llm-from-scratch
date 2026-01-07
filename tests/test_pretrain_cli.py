import os
import re
import subprocess
import sys


def extract_training_metrics(output: str) -> dict:
    """
    Extract training metrics from output text.
    """
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


def test_pretrain_cli_vs_script():

    # Paths
    training_file = "datasets/the-verdict.txt"
    cli_model_path = "tests/test_model_cli.pth"
    script_path = "../chapters/ch05/01_main-chapter-code/gpt_train.py"  # Relative to tests/ directory

    # Make sure training file exists
    assert os.path.exists(training_file), f"Training file {training_file} not found"

    try:
        # Run the chapter script with live output
        print("\n" + "=" * 80 + "\nRunning chapter script\n" + "=" * 80)
        chapter_cmd = [sys.executable, "-u", script_path]

        # Capture output while streaming it live
        script_output = []
        process = subprocess.Popen(chapter_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, cwd="tests", bufsize=1)
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()  # Force immediate display
            script_output.append(line)
        process.wait()
        assert process.returncode == 0, f"Chapter script failed with return code {process.returncode}"

        script_metrics = extract_training_metrics(''.join(script_output))

        # Run the CLI pretrain command with live output
        print("\n" + "=" * 80 + "\nRunning CLI command\n" + "=" * 80)
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
            "--saved-model-path", cli_model_path,
            "--context-length", "256",
            "--emb-dim", "768",
            "--n-layers", "12",
            "--n-heads", "12",
            "--vocab-size", "50257",
            "--drop-rate", "0.1"
            # qkv_bias defaults to False, so we don't need to pass --use-qkv-bias
        ]

        # Capture output while streaming it live
        cli_output = []
        process = subprocess.Popen(cli_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()  # Force immediate display
            cli_output.append(line)
        process.wait()
        assert process.returncode == 0, f"CLI command failed with return code {process.returncode}"

        cli_metrics = extract_training_metrics(''.join(cli_output))

        # Compare metrics
        print("\n" + "=" * 80 + "\nComparing training metrics\n" + "=" * 80)
        assert len(script_metrics['train_losses']) == len(cli_metrics['train_losses']), \
            f"Different number of training checkpoints: Script={len(script_metrics['train_losses'])}, CLI={len(cli_metrics['train_losses'])}"

        # Compare losses (allowing small floating point differences)
        tolerance = 1e-5
        for i in range(len(script_metrics['train_losses'])):
            train_diff = abs(script_metrics['train_losses'][i] - cli_metrics['train_losses'][i])
            val_diff = abs(script_metrics['val_losses'][i] - cli_metrics['val_losses'][i])
            if train_diff > tolerance or val_diff > tolerance:
                print(f"\nCheckpoint {i+1}:")
                print(f"  Script - Train: {script_metrics['train_losses'][i]:.6f}, Val: {script_metrics['val_losses'][i]:.6f}")
                print(f"  CLI    - Train: {cli_metrics['train_losses'][i]:.6f}, Val: {cli_metrics['val_losses'][i]:.6f}")
                print(f"  Diff   - Train: {train_diff:.2e}, Val: {val_diff:.2e}")
                assert False, f"Training metrics differ at checkpoint {i+1}"
    finally:
        for f in [cli_model_path, "tests/the-verdict.txt", "tests/model.pth", "tests/loss.pdf"]:
            if os.path.exists(f):
                os.remove(f)
