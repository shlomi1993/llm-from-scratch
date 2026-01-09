import os
import psutil
import re
import requests
import subprocess
import sys
import time

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


def extract_average_score(output: str) -> float:
    pattern = r'Average score:\s*([\d.]+)'
    match = re.search(pattern, output)
    return float(match.group(1)) if match else 0.0


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
        chapter_cmd = [sys.executable, "-u", script_path, "--test_mode"]
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
            "--test-output-path", cli_test_output
        ]
        cli_output = run_subprocess(cli_cmd)
        cli_metrics = extract_training_metrics(cli_output)

        # Compare metrics
        print("\n" + "=" * 80 + "\nComparing training metrics\n" + "=" * 80)
        assert len(script_metrics['train_losses']) == len(cli_metrics['train_losses']), \
            f"Different number of training checkpoints: Script={len(script_metrics['train_losses'])}, CLI={len(cli_metrics['train_losses'])}"

        # Compare losses (allowing small floating point differences)
        tolerance = 1e-2
        for i in range(len(script_metrics['train_losses'])):
            train_diff = abs(script_metrics['train_losses'][i] - cli_metrics['train_losses'][i])
            val_diff = abs(script_metrics['val_losses'][i] - cli_metrics['val_losses'][i])
            if train_diff > tolerance or val_diff > tolerance:
                print(f"\nLoss checkpoint {i+1}:")
                print(f"  Script - Train: {script_metrics['train_losses'][i]:.6f}, Val: {script_metrics['val_losses'][i]:.6f}")
                print(f"  CLI    - Train: {cli_metrics['train_losses'][i]:.6f}, Val: {cli_metrics['val_losses'][i]:.6f}")
                print(f"  Diff   - Train: {train_diff:.2e}, Val: {val_diff:.2e}")
                assert False, f"Training losses differ at checkpoint {i + 1}"

        print("✓ All training metrics match!")

        # Evaluate responses with ollama if available
        print("\n" + "=" * 80 + "\nEvaluating model responses with Ollama\n" + "=" * 80)
        ollama_running = any("ollama" in proc.info["name"] for proc in psutil.process_iter(["name"]))
        if ollama_running:
            print("Ollama is already running, restarting it...")
            run_subprocess("killall ollama")
            time.sleep(2)  # Wait for Ollama to fully shut down
            subprocess.Popen("ollama serve", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Wait for Ollama to start up
            print("Ollama restarted successfully")

        print("Ollama is running, evaluating responses...")

        # Run chapter's ollama_evaluate.py script
        print("\nRunning chapter ollama evaluation script...")
        chapter_eval_script = "../chapters/ch07/01_main-chapter-code/ollama_evaluate.py"
        chapter_eval_cmd = [sys.executable, "-u", chapter_eval_script, "--file_path", cli_test_output]
        chapter_eval_output = run_subprocess(chapter_eval_cmd, cwd="tests")
        chapter_score = extract_average_score(chapter_eval_output)
        print(f"Chapter evaluation average score: {chapter_score:.2f}/100")

        # Run CLI gpt2 evaluate command
        print("\nRunning CLI evaluation command...")
        cli_eval_cmd = ["gpt2", "evaluate", "--file-path", cli_test_output]
        cli_eval_output = run_subprocess(cli_eval_cmd)
        cli_score = extract_average_score(''.join(cli_eval_output))
        print(f"CLI evaluation average score: {cli_score:.2f}/100")

        # Compare scores
        print("\n" + "=" * 80 + "\nComparing evaluation scores\n" + "=" * 80)
        score_diff = abs(chapter_score - cli_score)
        score_tolerance = 0.1  # Allow small floating point differences

        if score_diff > score_tolerance:
            print(f"  Chapter score: {chapter_score:.2f}/100")
            print(f"  CLI score:     {cli_score:.2f}/100")
            print(f"  Difference:    {score_diff:.2f}")
            assert False, "Evaluation scores differ between chapter script and CLI"

        print("✓ Ollama evaluation scores match!")

    finally:
        try:
            run_subprocess("killall ollama")
        except Exception:
            pass

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
