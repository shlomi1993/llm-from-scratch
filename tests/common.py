from subprocess import Popen, PIPE, STDOUT
from sys import stdout


def run_subprocess(cmd: list[str] | str, cwd: str = None) -> list[str]:
    output_lines = []
    shell = isinstance(cmd, str)
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=cwd, bufsize=1, shell=shell)
    for line in process.stdout:
        print(line, end='')
        stdout.flush()  # Force immediate display
        output_lines.append(line)
    process.wait()
    assert process.returncode == 0, f"Subprocess failed: {' '.join(cmd) if not shell else cmd}"
    return "\n".join(output_lines)


def print_title(title: str, char: str = "=", sep_len: int = 80) -> None:
    sep = char * sep_len
    print(f"\n{sep}\n{title}\n{sep}\n")


def compare_losses(actual_losses: dict, expected_losses: dict, tolerance: float = 1e-5) -> None:
    print_title("Comparing loss metrics")
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
    print("✓ All loss metrics match!")


def compare_accuracies(actual_metrics: dict, expected_metrics: dict, tolerance: float = 1.0) -> None:
    print_title("Comparing accuracy metrics")
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
    print("✓ All accuracy metrics match!")


def compare_all_metrics(actual_metrics: dict, expected_metrics: dict) -> None:
    compare_losses(expected_metrics, actual_metrics, tolerance=1e-2)
    compare_accuracies(expected_metrics, actual_metrics, tolerance=1.0)
    print("✓✓ All metrics match!")
