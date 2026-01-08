import os
import re
import sys
import shutil

from tests.common import run_subprocess


def extract_training_metrics(output: str) -> dict:
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'steps': []
    }

    # Pattern for loss - handles both formats:
    #   Script: "Ep 1 (Step 000050): Train loss 0.693, Val loss 0.693"
    #   CLI:    "[timestamp] :: classification :: INFO ::   Step 000050: Train loss 0.693, Val loss 0.693"
    loss_pattern = r'Step (\d+)\)?: Train loss ([\d.]+), Val loss ([\d.]+)'
    loss_matches = re.findall(loss_pattern, output)
    for step, train_loss, val_loss in loss_matches:
        metrics['steps'].append(int(step))
        metrics['train_losses'].append(float(train_loss))
        metrics['val_losses'].append(float(val_loss))

    # Pattern for accuracy: "Training accuracy: 96.43% | Validation accuracy: 95.71%"
    acc_pattern = r'Training accuracy: ([\d.]+)%.*?Validation accuracy: ([\d.]+)%'
    acc_matches = re.findall(acc_pattern, output)
    for train_acc, val_acc in acc_matches:
        metrics['train_accs'].append(float(train_acc))
        metrics['val_accs'].append(float(val_acc))

    return metrics


def test_finetune_classifier_cli_vs_script():

    # Paths
    cli_model_path = "tests/test_classifier_model.pth"
    script_path = "../chapters/ch06/01_main-chapter-code/gpt_class_finetune.py"
    pretrained_model_path = "models/124M/model.pth"

    try:
        # Run the chapter script with live output
        print("\n" + "=" * 80 + "\nRunning chapter script\n" + "=" * 80)
        chapter_cmd = [sys.executable, "-u", script_path]
        script_output = run_subprocess(chapter_cmd, cwd="tests")
        script_metrics = extract_training_metrics(''.join(script_output))

        # Run the CLI finetune classification command with live output
        print("\n" + "=" * 80 + "\nRunning CLI command\n" + "=" * 80)
        cli_cmd = [
            "gpt2", "finetune", "classification",
            "--pretrained-model-path", pretrained_model_path,
            "--tuning-set-path", "datasets/sms_spam_collection/SMSSpamCollection.tsv",
            "--column-names", "Label", "Text",
            "--train-frac", "0.7",
            "--validation-frac", "0.1",
            "--batch-size", "8",
            "--seed", "123",
            "--device", "cpu",
            "--lr", "5e-5",
            "--n-epochs", "5",
            "--weight-decay", "0.1",
            "--eval-freq", "50",
            "--eval-iter", "5",
            "--model-save-path", cli_model_path
        ]

        # Capture output while streaming it live
        cli_output = run_subprocess(cli_cmd)
        cli_metrics = extract_training_metrics(''.join(cli_output))

        # Compare metrics
        print("\n" + "=" * 80 + "\nComparing training metrics\n" + "=" * 80)
        assert len(script_metrics['train_losses']) == len(cli_metrics['train_losses']), \
            f"Different number of training checkpoints: Script={len(script_metrics['train_losses'])}, CLI={len(cli_metrics['train_losses'])}"
        assert len(script_metrics['train_accs']) == len(cli_metrics['train_accs']), \
            f"Different number of accuracy measurements: Script={len(script_metrics['train_accs'])}, CLI={len(cli_metrics['train_accs'])}"

        # Compare losses (allowing small floating point differences)
        tolerance = 1e-2  # Larger tolerance for classification tasks
        for i in range(len(script_metrics['train_losses'])):
            train_diff = abs(script_metrics['train_losses'][i] - cli_metrics['train_losses'][i])
            val_diff = abs(script_metrics['val_losses'][i] - cli_metrics['val_losses'][i])
            if train_diff > tolerance or val_diff > tolerance:
                print(f"\nLoss checkpoint {i+1}:")
                print(f"  Script - Train: {script_metrics['train_losses'][i]:.6f}, Val: {script_metrics['val_losses'][i]:.6f}")
                print(f"  CLI    - Train: {cli_metrics['train_losses'][i]:.6f}, Val: {cli_metrics['val_losses'][i]:.6f}")
                print(f"  Diff   - Train: {train_diff:.2e}, Val: {val_diff:.2e}")
                assert False, f"Training losses differ at checkpoint {i + 1}"

        # Compare accuracies
        acc_tolerance = 1.0  # 1% tolerance for accuracy
        for i in range(len(script_metrics['train_accs'])):
            train_acc_diff = abs(script_metrics['train_accs'][i] - cli_metrics['train_accs'][i])
            val_acc_diff = abs(script_metrics['val_accs'][i] - cli_metrics['val_accs'][i])
            if train_acc_diff > acc_tolerance or val_acc_diff > acc_tolerance:
                print(f"\nAccuracy checkpoint {i + 1}:")
                print(f"  Script - Train: {script_metrics['train_accs'][i]:.2f}%, Val: {script_metrics['val_accs'][i]:.2f}%")
                print(f"  CLI    - Train: {cli_metrics['train_accs'][i]:.2f}%, Val: {cli_metrics['val_accs'][i]:.2f}%")
                print(f"  Diff   - Train: {train_acc_diff:.2f}%, Val: {val_acc_diff:.2f}%")
                assert False, f"Training accuracies differ at checkpoint {i + 1}"

        print("✓ All metrics match!")

    finally:
        cleanup_files = [
            cli_model_path,
            "tests/train.csv",
            "tests/validation.csv",
            "tests/test.csv",
            "tests/loss-plot.pdf",
            "tests/accuracy-plot.pdf",
            "tests/sms_spam_collection.zip"
        ]
        for f in cleanup_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("tests/sms_spam_collection"):
            shutil.rmtree("tests/sms_spam_collection")
