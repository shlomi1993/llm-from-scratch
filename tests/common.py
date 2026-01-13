import re
from shutil import get_terminal_size
from subprocess import Popen, PIPE, STDOUT
from sys import stdout


COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RESET = '\033[0m'

# Pattern for chapter script loss: "Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392"
CHAPTER_LOSS_PATTERN = r'Ep \d+ \(Step (\d+)\): Train loss ([\d.]+), Val loss ([\d.]+)'

# Pattern for CLI app loss (format_training_progress): "Epoch 1/5 | Step 50/650 | Train Loss: 0.693 | Val Loss: 0.693"
CLI_LOSS_PATTERN = r'Step (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)'

# Pattern for chapter script accuracy: "Training accuracy: 70.00% | Validation accuracy: 72.50%"
CHAPTER_ACC_PATTERN = r'Training accuracy: ([\d.]+)%.*?Validation accuracy: ([\d.]+)%'

# Pattern for CLI app accuracy: "Train Acc: 70.00% | Val Acc: 72.50%"
CLI_ACC_PATTERN = r'Train Acc: ([\d.]+)% \| Val Acc: ([\d.]+)%'


def run_subprocess(cmd: list[str] | str, cwd: str = None) -> list[str]:
    shell = isinstance(cmd, str)
    msg_cmd = cmd if shell else ' '.join(str(c) for c in cmd)
    print(f"Running:\n{msg_cmd}\n")
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=cwd, bufsize=1, shell=shell)
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        stdout.flush()  # Force immediate display
        output_lines.append(line)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {msg_cmd}")
    return "\n".join(output_lines)


def print_title(title: str, char: str = "-") -> None:
    width = get_terminal_size().columns
    sep = char * width
    print(f"\n\n{COLOR_BLUE}{sep}\n{title}\n{sep}\n{COLOR_RESET}")


def extract_losses(output: str, ref: bool = False) -> dict:
    metrics = {'train_losses': [], 'val_losses': [], 'steps': []}
    loss_pattern = CHAPTER_LOSS_PATTERN if ref else CLI_LOSS_PATTERN
    loss_matches = re.findall(loss_pattern, output)
    for step, train_loss, val_loss in loss_matches:
        metrics['steps'].append(int(step))
        metrics['train_losses'].append(float(train_loss))
        metrics['val_losses'].append(float(val_loss))
    return metrics


def compare_losses(actual_losses: dict, expected_losses: dict, tolerance: float = 1e-5) -> None:
    print("Comparing loss metrics...", end=" ")
    loss_checkpoints = zip(
        expected_losses['train_losses'],
        actual_losses['train_losses'],
        expected_losses['val_losses'],
        actual_losses['val_losses']
    )
    for i, (s_train, c_train, s_val, c_val) in enumerate(loss_checkpoints):
        train_diff = abs(s_train - c_train)
        val_diff = abs(s_val - c_val)
        if train_diff > tolerance or val_diff > tolerance:
            print(f"\nCheckpoint {i + 1}:")
            print(f"  Script - Train: {s_train:.6f}, Val: {s_val:.6f}")
            print(f"  CLI    - Train: {c_train:.6f}, Val: {c_val:.6f}")
            print(f"  Diff   - Train: {train_diff:.2e}, Val: {val_diff:.2e}")
            assert False, f"Training losses differ at checkpoint {i + 1}"
    print(f"{COLOR_GREEN}✓ All loss metrics match!{COLOR_RESET}")


def compare_accuracies(actual_metrics: dict, expected_metrics: dict, tolerance: float = 1.0) -> None:
    print("Comparing accuracy metrics...", end=" ")
    accuracy_checkpoints = zip(
        expected_metrics['train_accs'],
        actual_metrics['train_accs'],
        expected_metrics['val_accs'],
        actual_metrics['val_accs']
    )
    for i, (s_train, c_train, s_val, c_val) in enumerate(accuracy_checkpoints):
        train_acc_diff = abs(s_train - c_train)
        val_acc_diff = abs(s_val - c_val)
        if train_acc_diff > tolerance or val_acc_diff > tolerance:
            print(f"\nAccuracy checkpoint {i + 1}:")
            print(f"  Script - Train: {s_train:.2f}%, Val: {s_val:.2f}%")
            print(f"  CLI    - Train: {c_train:.2f}%, Val: {c_val:.2f}%")
            print(f"  Diff   - Train: {train_acc_diff:.2f}%, Val: {val_acc_diff:.2f}%")
            assert False, f"Training accuracies differ at checkpoint {i + 1}"
    print(f"{COLOR_GREEN}✓ All accuracy metrics match!{COLOR_RESET}")

