import os
import subprocess
import sys
import tempfile
import pytest

from pathlib import Path

# Import from plotting tool
from tools.plot_memory_estimates_gqa import (
    bytes_convert as plot_bytes_convert,
    savings_percent,
    plot_abs_kv_vs_context_multi_groups,
    plot_abs_kv_vs_context_with_mla,
    mla_bytes_convert
)


class TestPlotMemoryEstimatesQueryAttention:
    """
    Tests for plot_memory_estimates.py tool with ...
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
        Test GQA plotting function creates PDF file.
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
                pdf_path = Path(tmp_dir) / "kv_bytes_vs_context_length_gqa.pdf"
                assert pdf_path.exists(), "GQA PDF file should be created"
                assert pdf_path.stat().st_size > 0, "PDF file should not be empty"

            finally:
                # Restore original directory
                os.chdir(original_cwd)

    def test_plot_function_with_different_params(self):
        """
        Test that GQA plotting function handles different parameter ranges.
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
                assert Path("kv_bytes_vs_context_length_gqa.pdf").exists(), "GQA PDF should be created"

            finally:
                os.chdir(original_cwd)

    def test_plot_abs_kv_vs_context_with_mla(self):
        """
        Test MLA plotting function creates PDF file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)

                # Import the new plotting function
                from tools.plot_memory_estimates_gqa import plot_abs_kv_vs_context_with_mla

                # Run the MLA plotting function
                plot_abs_kv_vs_context_with_mla()

                # Check that PDF was created
                pdf_path = Path(tmp_dir) / "kv_bytes_vs_context_length_mla.pdf"
                assert pdf_path.exists(), "MLA PDF file should be created"
                assert pdf_path.stat().st_size > 0, "PDF file should not be empty"

            finally:
                os.chdir(original_cwd)

    def test_plot_mla_with_custom_latent_dims(self):
        """
        Test MLA plotting with custom latent dimensions.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)

                from tools.plot_memory_estimates_gqa import plot_abs_kv_vs_context_with_mla

                # Test with custom latent dimensions
                custom_dims = [2048, 1024, 512]
                plot_abs_kv_vs_context_with_mla(latent_dims=custom_dims)

                # Verify file exists
                pdf_path = Path(tmp_dir) / "kv_bytes_vs_context_length_mla.pdf"
                assert pdf_path.exists(), "MLA PDF with custom dims should be created"

            finally:
                os.chdir(original_cwd)

    def test_mla_bytes_convert(self):
        """
        Test MLA bytes conversion helper function.
        """
        from tools.plot_memory_estimates_gqa import mla_bytes_convert

        # Test with known values
        result = mla_bytes_convert(
            batch_size=1,
            context_length=1024,
            n_layers=12,
            latent_dim=128,
            dtype="float32"
        )
        
        # Should return a float representing GB
        assert isinstance(result, float), "Should return float"
        assert result > 0, "Should return positive value"
        
        # Test with different dtype (should be smaller with bf16)
        result_bf16 = mla_bytes_convert(
            batch_size=1,
            context_length=1024,
            n_layers=12,
            latent_dim=128,
            dtype="bfloat16"
        )
        assert result_bf16 < result, "bf16 should use less memory than fp32"
