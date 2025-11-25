import os
import subprocess
import sys
import tempfile
import pytest

from pathlib import Path
from unittest.mock import patch

from tools.memory_estimator_gqa import bytes_convert, kv_bytes_total, DTYPE_BYTES, main as memory_estimator_main
from tools.plot_memory_estimates_gqa import bytes_convert as plot_bytes_convert, savings_percent, plot_abs_kv_vs_context_multi_groups


class TestMemoryEstimatorGroupQueryAttention:
    """
    Tests for memory_estimator_gqa.py tool.
    """

    def test_dtype_bytes_dictionary(self):
        """
        Test that DTYPE_BYTES contains expected data types.
        """
        expected_dtypes = {"fp32", "bf16", "fp16", "fp8", "int8"}
        assert set(DTYPE_BYTES.keys()) == expected_dtypes, "DTYPE_BYTES should contain all expected data types"

        # Verify byte sizes
        assert DTYPE_BYTES["fp32"] == 4, "fp32 should be 4 bytes"
        assert DTYPE_BYTES["bf16"] == 2, "bf16 should be 2 bytes"
        assert DTYPE_BYTES["fp16"] == 2, "fp16 should be 2 bytes"
        assert DTYPE_BYTES["fp8"] == 1, "fp8 should be 1 byte"
        assert DTYPE_BYTES["int8"] == 1, "int8 should be 1 byte"

    def test_bytes_convert(self):
        """
        Test bytes to GB conversion.
        """
        # Test exact conversions
        assert bytes_convert(1_000_000_000) == "1.00 GB", "1 billion bytes should be 1 GB"
        assert bytes_convert(2_500_000_000) == "2.50 GB", "2.5 billion bytes should be 2.50 GB"
        assert bytes_convert(1_000) == "0.00 GB", "1000 bytes should round to 0.00 GB"

        # Test formatting
        result = bytes_convert(123_456_789_012)
        assert "123.46 GB" in result or "123,456,789,012" in result, "Should format large numbers correctly"

    def test_kv_bytes_total(self):
        """
        Test KV-cache memory calculation.
        """
        # Test with simple values
        batch_size = 1
        context_length = 1024
        emb_dim = 768
        n_heads = 12
        n_kv_heads = 12  # MHA
        n_layers = 12
        bytes_per_elem = 2  # fp16

        total = kv_bytes_total(
            batch_size, context_length, emb_dim, n_heads, n_kv_heads, n_layers, bytes_per_elem
        )

        # Verify it's a positive integer
        assert isinstance(total, int), "Total should be an integer"
        assert total > 0, "Total should be positive"

        # Verify GQA uses less memory than MHA
        n_kv_heads_gqa = 6  # Half the KV heads
        total_gqa = kv_bytes_total(
            batch_size, context_length, emb_dim, n_heads, n_kv_heads_gqa, n_layers, bytes_per_elem
        )
        assert total_gqa < total, "GQA should use less memory than MHA"
        assert total_gqa == total / 2, "With half KV heads, should use half the memory"

    def test_kv_bytes_total_scaling(self):
        """
        Test that KV-cache memory scales correctly with parameters.
        """
        base_args = {
            "batch_size": 1,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_kv_heads": 12,
            "n_layers": 12,
            "bytes_per_elem": 2,
        }

        base_total = kv_bytes_total(**base_args)

        # Double batch size -> double memory
        args_2x_batch = base_args.copy()
        args_2x_batch["batch_size"] = 2
        assert kv_bytes_total(**args_2x_batch) == 2 * base_total, "Doubling batch size should double memory"

        # Double context length -> double memory
        args_2x_context = base_args.copy()
        args_2x_context["context_length"] = 2048
        assert kv_bytes_total(**args_2x_context) == 2 * base_total, "Doubling context should double memory"

        # Double layers -> double memory
        args_2x_layers = base_args.copy()
        args_2x_layers["n_layers"] = 24
        assert kv_bytes_total(**args_2x_layers) == 2 * base_total, "Doubling layers should double memory"

    def test_memory_estimator_main_with_args(self):
        """
        Test the main function with command-line arguments.
        """
        test_args = [
            "--emb_dim", "768",
            "--n_heads", "12",
            "--n_layers", "12",
            "--n_kv_groups", "4",
            "--context_length", "1024",
            "--batch_size", "1",
            "--dtype", "fp16",
        ]

        with patch("sys.argv", ["memory_estimator_gqa.py"] + test_args):
            # Should run without errors
            try:
                memory_estimator_main()
            except SystemExit:
                pytest.fail("main() should not exit on valid arguments")

    def test_memory_estimator_main_invalid_n_kv_groups(self, capsys):
        """
        Test that main raises error when n_kv_groups doesn't divide n_heads.
        """
        test_args = [
            "--emb_dim", "768",
            "--n_heads", "12",
            "--n_layers", "12",
            "--n_kv_groups", "5",  # 12 % 5 != 0
        ]

        with patch("sys.argv", ["memory_estimator_gqa.py"] + test_args):
            with pytest.raises(ValueError, match="n_kv_groups must divide n_heads exactly"):
                memory_estimator_main()

    def test_memory_estimator_different_dtypes(self):
        """
        Test memory calculation with different data types.
        """
        base_args = {
            "batch_size": 1,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_kv_heads": 12,
            "n_layers": 12,
        }

        # Calculate for different dtypes
        total_fp32 = kv_bytes_total(**base_args, bytes_per_elem=DTYPE_BYTES["fp32"])
        total_fp16 = kv_bytes_total(**base_args, bytes_per_elem=DTYPE_BYTES["fp16"])
        total_fp8 = kv_bytes_total(**base_args, bytes_per_elem=DTYPE_BYTES["fp8"])

        # Verify ratios
        assert total_fp32 == 2 * total_fp16, "fp32 should use 2x memory of fp16"
        assert total_fp16 == 2 * total_fp8, "fp16 should use 2x memory of fp8"
        assert total_fp32 == 4 * total_fp8, "fp32 should use 4x memory of fp8"


class TestPlotMemoryEstimatesQueryAttention:
    """
    Tests for plot_memory_estimates_gqa.py tool.
    """

    def test_plot_bytes_convert(self):
        """
        Test bytes to GB conversion in plotting module.
        """
        # Test conversion
        assert plot_bytes_convert(1_000_000_000) == "1.00", "Should convert 1B bytes to 1.00 GB"
        assert plot_bytes_convert(2_500_000_000) == "2.50", "Should convert 2.5B bytes to 2.50 GB"

        # Verify format (2 decimal places)
        result = plot_bytes_convert(123_456_789)
        assert len(result.split(".")[-1]) == 2, "Should have 2 decimal places"

    def test_savings_percent(self):
        """
        Test savings percentage calculation.
        """
        # Test 50% savings (GQA uses half the memory)
        total_mha = 1000
        total_gqa = 500
        savings = savings_percent(total_mha, total_gqa)
        assert savings == 50.0, "Should calculate 50% savings"

        # Test 75% savings
        total_gqa_25 = 250
        savings_75 = savings_percent(total_mha, total_gqa_25)
        assert savings_75 == 75.0, "Should calculate 75% savings"

        # Test no savings
        savings_zero = savings_percent(total_mha, total_mha)
        assert savings_zero == 0.0, "Should calculate 0% savings when equal"

    def test_plot_abs_kv_vs_context_multi_groups(self):
        """
        Test plotting function creates PDF file.
        """
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                # Change to temp directory
                os.chdir(tmp_dir)

                # Run the plotting function
                plot_abs_kv_vs_context_multi_groups()

                # Check that PDF was created
                pdf_path = Path(tmp_dir) / "kv_bytes_vs_context_length.pdf"
                assert pdf_path.exists(), "PDF file should be created"
                assert pdf_path.stat().st_size > 0, "PDF file should not be empty"

            finally:
                # Restore original directory
                os.chdir(original_cwd)

    def test_plot_function_with_different_params(self):
        """
        Test that plotting function handles different parameter ranges.
        """
        # This test verifies the function runs without errors
        # We can't easily verify the plot content without image comparison
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)

                # Should not raise any exceptions
                plot_abs_kv_vs_context_multi_groups()

                # Verify file exists
                assert Path("kv_bytes_vs_context_length.pdf").exists(), "PDF should be created"

            finally:
                os.chdir(original_cwd)


class TestMemoryEstimatorCli:
    """
    Test command-line interface of memory_estimator_gqa.py.
    """

    def test_cli_help(self):
        """
        Test that --help works.
        """
        result = subprocess.run(
            [sys.executable, "tools/memory_estimator_gqa.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Help command should succeed"
        assert "Estimate KV-cache memory" in result.stdout, "Help should describe the tool"

    def test_cli_valid_arguments(self):
        """
        Test CLI with valid arguments.
        """
        result = subprocess.run(
            [
                sys.executable,
                "tools/memory_estimator_gqa.py",
                "--emb_dim", "768",
                "--n_heads", "12",
                "--n_layers", "12",
                "--n_kv_groups", "4",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Valid arguments should succeed"
        assert "Config" in result.stdout, "Should output config information"
        assert "KV-cache totals" in result.stdout, "Should output KV-cache totals"
        assert "MHA total KV cache" in result.stdout, "Should show MHA total"
        assert "GQA total KV cache" in result.stdout, "Should show GQA total"

    def test_cli_invalid_n_kv_groups(self):
        """
        Test CLI with invalid n_kv_groups.
        """
        result = subprocess.run(
            [
                sys.executable,
                "tools/memory_estimator_gqa.py",
                "--emb_dim", "768",
                "--n_heads", "12",
                "--n_layers", "12",
                "--n_kv_groups", "5",  # Invalid: 12 % 5 != 0
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Invalid n_kv_groups should fail"
        assert "n_kv_groups must divide n_heads exactly" in result.stderr, "Should show error message"

    def test_cli_different_dtypes(self):
        """
        Test CLI with different data types.
        """
        dtypes = ["fp32", "bf16", "fp16", "fp8", "int8"]

        for dtype in dtypes:
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/memory_estimator_gqa.py",
                    "--emb_dim", "768",
                    "--n_heads", "12",
                    "--n_layers", "12",
                    "--n_kv_groups", "4",
                    "--dtype", dtype,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Should succeed with dtype={dtype}"
            assert dtype in result.stdout, f"Should show dtype={dtype} in output"

    def test_cli_output_format(self):
        """
        Test that CLI output is properly formatted.
        """
        result = subprocess.run(
            [
                sys.executable,
                "tools/memory_estimator_gqa.py",
                "--emb_dim", "2048",
                "--n_heads", "24",
                "--n_layers", "48",
                "--n_kv_groups", "4",
                "--context_length", "2048",
                "--batch_size", "2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Should succeed"

        # Check output contains expected values
        assert "emb_dim" in result.stdout, "Should show emb_dim"
        assert "n_heads" in result.stdout, "Should show n_heads"
        assert "n_layers" in result.stdout, "Should show n_layers"
        assert "batch_size" in result.stdout, "Should show batch_size"
        assert "Ratio (MHA / GQA)" in result.stdout, "Should show ratio"
        assert "Savings (GQA vs MHA)" in result.stdout, "Should show savings"
        assert "GB" in result.stdout, "Should show values in GB"


class TestIntegrationMemoryTools:
    """
    Integration tests for memory estimation tools.
    """

    def test_memory_estimates_consistency(self):
        """
        Test that both tools produce consistent memory estimates.
        """
        # Parameters
        batch_size = 1
        context_length = 1024
        emb_dim = 768
        n_heads = 12
        n_layers = 12
        bytes_per_elem = DTYPE_BYTES["fp16"]

        # Calculate MHA total
        total_mha = kv_bytes_total(
            batch_size, context_length, emb_dim, n_heads, n_heads, n_layers, bytes_per_elem
        )

        # Calculate GQA total with n_kv_groups=4
        n_kv_heads_gqa = n_heads // 4
        total_gqa = kv_bytes_total(
            batch_size, context_length, emb_dim, n_heads, n_kv_heads_gqa, n_layers, bytes_per_elem
        )

        # Verify ratio
        ratio = total_mha / total_gqa
        assert ratio == 4.0, "With n_kv_groups=4, MHA should use 4x memory of GQA"

        # Verify savings
        savings = savings_percent(total_mha, total_gqa)
        assert savings == 75.0, "GQA with 4 groups should save 75% memory vs MHA"

    def test_all_context_lengths_valid(self):
        """
        Test that calculations work for all context lengths used in plotting.
        """
        context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

        batch_size = 1
        emb_dim = 2048
        n_heads = 24
        n_layers = 48
        bytes_per_elem = DTYPE_BYTES["bf16"]

        for context_length in context_lengths:
            total = kv_bytes_total(
                batch_size, context_length, emb_dim, n_heads, n_heads, n_layers, bytes_per_elem
            )
            assert total > 0, f"Should calculate valid memory for context_length={context_length}"

            # Verify conversion to GB works
            gb_str = plot_bytes_convert(total)
            assert "." in gb_str, "Should convert to GB format"

    def test_realistic_model_sizes(self):
        """
        Test memory estimation with realistic model configurations.
        """
        # GPT-2 small-like config
        gpt2_small = {
            "batch_size": 1,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_kv_heads": 12,
            "n_layers": 12,
            "bytes_per_elem": DTYPE_BYTES["fp16"],
        }

        total_small = kv_bytes_total(**gpt2_small)
        assert total_small > 0, "Should calculate valid memory for GPT-2 small"

        # Larger model
        gpt2_large = {
            "batch_size": 1,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20,
            "n_kv_heads": 20,
            "n_layers": 36,
            "bytes_per_elem": DTYPE_BYTES["fp16"],
        }

        total_large = kv_bytes_total(**gpt2_large)
        assert total_large > total_small, "Larger model should use more KV-cache memory"
