"""
Tests for the memory estimator CLI tool.

This test suite tests the CLI interface using subprocess calls.
For programmatic testing of the underlying functions, see the original test_memory_estimator.py
which imports from tools.memory_estimator.src.
"""

import subprocess
import sys

from pathlib import Path
from typing import List


# Path to the memory estimator tool
TOOL_PATH = Path(__file__).parent.parent / "memory_estimator" / "memory_estimator.py"


def run_memory_estimator(args: List[str]) -> subprocess.CompletedProcess:
    """
    Run the memory estimator CLI tool and return results.

    Args:
        args (List[str]): Command-line arguments (without the script name)

    Returns:
        subprocess.CompletedProcess: Completed process with return code, stdout, and stderr
    """
    return subprocess.run(
        [sys.executable, str(TOOL_PATH)] + args,
        capture_output=True,
        text=True,
        cwd=TOOL_PATH.parent  # Run from the tool's directory
    )


class TestCliGqa:
    """
    Test GQA mode CLI based on chapters/ch04/04_gqa/README.md example.

    Command from README:
    python memory_estimator.py --mode gqa --emb_dim 4096 --n_heads 32 --n_layers 32 \\
        --context_length 32768 --n_kv_groups 4 --batch_size 1 --dtype bf16

    Expected output:
    MHA total KV cache  : 17.18 GB
    GQA total KV cache  : 4.29 GB
    Ratio (MHA / GQA)   : 4.00x
    Savings (GQA vs MHA): 75.00%
    """

    def test_gqa_readme_example(self):
        """Test GQA mode CLI with README example."""
        result = run_memory_estimator([
            "--mode", "gqa",
            "--emb-dim", "4096",
            "--n-heads", "32",
            "--n-layers", "32",
            "--context-length", "32768",
            "--n-kv-groups", "4",
            "--batch-size", "1",
            "--dtype", "bfloat16"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        assert "17.18 GB" in result.stdout, "Output should contain MHA total (17.18 GB)"
        assert "4.29 GB" in result.stdout, "Output should contain GQA total (4.29 GB)"
        assert "4.00x" in result.stdout, "Output should contain ratio (4.00x)"
        assert "75.00%" in result.stdout, "Output should contain savings (75.00%)"

    def test_gqa_invalid_n_kv_groups(self):
        """Test that invalid n_kv_groups produces an error."""
        result = run_memory_estimator([
            "--mode", "gqa",
            "--emb-dim", "4096",
            "--n-heads", "32",
            "--n-layers", "32",
            "--context-length", "32768",
            "--n-kv-groups", "5",  # 5 doesn't divide 32
            "--batch-size", "1",
            "--dtype", "bfloat16"
        ])

        assert result.returncode != 0, "CLI should fail with invalid n_kv_groups"
        assert "n_kv_groups must divide n_heads" in result.stderr or "n_kv_groups must divide n_heads" in result.stdout


class TestCliMla:
    """
    Test MLA mode CLI based on chapters/ch04/05_mla/README.md example.

    Command from README:
    python memory_estimator.py --mode mla --context_length 8192 --emb_dim 2048 \\
        --n_heads 24 --n_layers 48 --n_kv_groups 4 --batch_size 1 --dtype bf16 --latent_dim 1024

    Expected output:
    MHA total KV cache  : 3.25 GB
    GQA total KV cache  : 0.81 GB
    MLA total KV cache  : 0.81 GB
    """

    def test_mla_readme_example(self):
        """Test MLA mode CLI with README example."""
        result = run_memory_estimator([
            "--mode", "mla",
            "--context-length", "8192",
            "--emb-dim", "2048",
            "--n-heads", "24",
            "--n-layers", "48",
            "--n-kv-groups", "4",
            "--batch-size", "1",
            "--dtype", "bfloat16",
            "--latent-dim", "1024"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        assert "3.25 GB" in result.stdout, "Output should contain MHA total (3.25 GB)"
        assert "0.81 GB" in result.stdout, "Output should contain GQA and MLA totals (0.81 GB)"
        assert "75.19%" in result.stdout, "Output should contain MLA savings (75.19%)"


class TestCliSwa:
    """
    Test SWA mode CLI based on chapters/ch04/06_swa/README.md example.

    Command from README:
    python memory_estimator.py --mode swa --emb_dim 4096 --n_heads 32 --n_layers 32 \\
        --context_length 32768 --n_kv_groups 4 --batch_size 1 --dtype bf16 \\
        --sliding_window_size 1024 --swa_ratio "5:1"

    Expected output:
    MHA KV total           : 17.18 GB
    GQA KV total           : 4.29 GB
    MHA + SWA (Ratio: 5:1) : 3.14 GB
    MHA + GQA (Ratio: 5:1) : 0.78 GB
    """

    def test_swa_readme_example(self):
        """Test SWA mode CLI with README example."""
        result = run_memory_estimator([
            "--mode", "swa",
            "--emb-dim", "4096",
            "--n-heads", "32",
            "--n-layers", "32",
            "--context-length", "32768",
            "--n-kv-groups", "4",
            "--sliding-window-size", "1024",
            "--swa-ratio", "5:1",
            "--batch-size", "1",
            "--dtype", "bfloat16"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        assert "17.18 GB" in result.stdout, "Output should contain MHA total (17.18 GB)"
        assert "4.29 GB" in result.stdout, "Output should contain GQA total (4.29 GB)"
        assert "3.14 GB" in result.stdout, "Output should contain MHA+SWA (3.14 GB)"
        assert "0.78 GB" in result.stdout, "Output should contain GQA+SWA (0.78 GB)"


class TestCliMoe:
    """
    Test MoE mode CLI based on chapters/ch04/07_moe/README.md example.

    Command from README:
    python memory_estimator.py --mode moe --emb_dim 7168 --hidden_dim 14336 \\
        --ffn_type swiglu --num_experts 8 --top_k 2 --match_dense

    Expected output:
    Dense FFN params       : 308,281,344 (0.62 GB)
    MoE TOTAL params       : 308,338,688 (0.62 GB)
    MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
    """

    def test_moe_readme_example(self):
        """Test MoE mode CLI with README example."""
        result = run_memory_estimator([
            "--mode", "moe",
            "--emb-dim", "7168",
            "--hidden-dim", "14336",
            "--ffn-type", "swiglu",
            "--num-experts", "8",
            "--top-k", "2",
            "--dtype", "bfloat16",
            "--match-dense"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        assert "308,281,344" in result.stdout, "Output should contain dense params (308,281,344)"
        assert "308,338,688" in result.stdout, "Output should contain MoE total (308,338,688)"
        assert "77,127,680" in result.stdout, "Output should contain active params/token (77,127,680)"


class TestCliMultiMode:
    """Test multi-mode execution."""

    def test_multi_mode_mha_gqa(self):
        """Test running multiple modes in one call."""
        result = run_memory_estimator([
            "--mode", "mha", "gqa",
            "--emb-dim", "2048",
            "--n-heads", "16",
            "--n-layers", "12",
            "--n-kv-groups", "4",
            "--context-length", "8192",
            "--batch-size", "1",
            "--dtype", "float16"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        assert "MHA total KV cache" in result.stdout, "Output should contain MHA results"
        assert "GQA total KV cache" in result.stdout, "Output should contain GQA results"

    def test_multi_mode_mha_gqa_swa(self):
        """Test running MHA, GQA, and SWA modes together."""
        result = run_memory_estimator([
            "--mode", "mha", "gqa", "swa",
            "--emb-dim", "4096",
            "--n-heads", "32",
            "--n-layers", "32",
            "--context-length", "32768",
            "--n-kv-groups", "4",
            "--batch-size", "1",
            "--dtype", "bfloat16",
            "--sliding-window-size", "1024",
            "--swa-ratio", "5:1"
        ])

        assert result.returncode == 0, f"CLI should exit successfully. stderr: {result.stderr}"
        # All three modes should produce output
        assert result.stdout.count("==== Config ====") == 3, "Should have output for all 3 modes"


class TestCliHelp:
    """Test help and version output."""

    def test_help_flag(self):
        """Test --help flag."""
        result = run_memory_estimator(["--help"])

        assert result.returncode == 0, "Help should exit successfully"
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower(), "Should show usage information"
        assert "--mode" in result.stdout or "--mode" in result.stderr, "Should describe --mode argument"

    def test_no_args(self):
        """Test running with no arguments."""
        result = run_memory_estimator([])

        # Should fail or show help
        assert result.returncode != 0, "Should fail when no arguments provided"
        assert "required" in result.stderr.lower() or "usage:" in result.stderr.lower(), "Should indicate missing arguments"
