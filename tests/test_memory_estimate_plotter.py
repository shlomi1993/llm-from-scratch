"""
Tests for the memory_estimate_plotter tool.
"""

import os
import tempfile
import pytest

from argparse import Namespace
from pathlib import Path

from tools.memory_estimate_plotter import (
    plot_gqa,
    plot_mla,
    plot_swa,
    plot_moe,
    bytes_to_gb,
    parse_swa_ratio,
    kv_bytes_total_mha,
    kv_bytes_total_gqa,
    kv_bytes_total_mla,
    kv_bytes_total_mha_swa,
    kv_bytes_total_gqa_swa,
    validate_args,
)


class TestUtilityFunctions:
    """
    Test utility functions for memory estimate plotter.
    """

    def test_bytes_to_gb(self):
        """
        Test bytes to gigabytes conversion.
        """
        assert bytes_to_gb(1_000_000_000) == 1.0, "Should convert 1B bytes to 1.0 GB"
        assert bytes_to_gb(2_500_000_000) == 2.5, "Should convert 2.5B bytes to 2.5 GB"
        assert bytes_to_gb(500_000_000) == 0.5, "Should convert 500M bytes to 0.5 GB"
        assert bytes_to_gb(0) == 0.0, "Should handle zero bytes"

    def test_parse_swa_ratio_valid(self):
        """
        Test parsing valid SWA ratio strings.
        """
        assert parse_swa_ratio("5:1") == (5, 1), "Should parse '5:1' correctly"
        assert parse_swa_ratio("1:5") == (1, 5), "Should parse '1:5' correctly"
        assert parse_swa_ratio("1:0") == (1, 0), "Should parse '1:0' correctly"
        assert parse_swa_ratio("0:1") == (0, 1), "Should parse '0:1' correctly"
        assert parse_swa_ratio("10:20") == (10, 20), "Should parse larger numbers"

    def test_parse_swa_ratio_invalid(self):
        """
        Test parsing invalid SWA ratio strings.
        """
        with pytest.raises(ValueError, match="must be in the form 'a:b'"):
            parse_swa_ratio("5-1")

        with pytest.raises(ValueError, match="must be in the form 'a:b'"):
            parse_swa_ratio("abc")

        with pytest.raises(ValueError, match="must be in the form 'a:b'"):
            parse_swa_ratio("0:0")

        with pytest.raises(ValueError, match="must be in the form 'a:b'"):
            parse_swa_ratio("-1:5")


class TestKVCacheCalculations:
    """
    Test KV-cache memory calculation functions.
    """

    def test_kv_bytes_total_mha(self):
        """
        Test MHA KV-cache calculation.
        """
        # Simple test case
        total = kv_bytes_total_mha(
            batch_size=1,
            context_length=1024,
            emb_dim=768,
            n_layers=12,
            bytes_per_elem=2  # bf16
        )
        expected = 1 * 1024 * 768 * 2 * 2 * 12  # B * L * E * 2(K,V) * bytes * layers
        assert total == expected, f"MHA calculation should match expected value"

    def test_kv_bytes_total_gqa(self):
        """
        Test GQA KV-cache calculation.
        """
        # GQA should use less memory than MHA
        mha_total = kv_bytes_total_mha(1, 1024, 768, 12, 2)
        gqa_total = kv_bytes_total_gqa(1, 1024, 768, 12, 2, n_kv_groups=4)

        assert gqa_total < mha_total, "GQA should use less memory than MHA"
        assert gqa_total == mha_total / 4, "GQA with 4 groups should use 1/4 memory"

    def test_kv_bytes_total_mla(self):
        """
        Test MLA KV-cache calculation.
        """
        total = kv_bytes_total_mla(
            batch_size=1,
            context_length=1024,
            n_layers=12,
            latent_dim=128,
            bytes_per_elem=2
        )
        expected = 1 * 1024 * 12 * 128 * 2
        assert total == expected, "MLA calculation should match expected value"

    def test_kv_bytes_total_mha_swa(self):
        """
        Test MHA with SWA calculation.
        """
        # SWA should use less memory than full MHA
        full_mha = kv_bytes_total_mha(1, 8192, 768, 12, 2)
        swa_mha = kv_bytes_total_mha_swa(1, 8192, 768, 12, 2, window=2048, swa_ratio="5:1")

        assert swa_mha < full_mha, "SWA should use less memory than full attention"

    def test_kv_bytes_total_gqa_swa(self):
        """
        Test GQA with SWA calculation.
        """
        # GQA+SWA should use less memory than both GQA alone and MHA+SWA
        full_gqa = kv_bytes_total_gqa(1, 8192, 768, 12, 2, n_kv_groups=4)
        swa_gqa = kv_bytes_total_gqa_swa(1, 8192, 768, 12, 2, 4, window=2048, swa_ratio="5:1")

        assert swa_gqa < full_gqa, "GQA+SWA should use less memory than full GQA"


class TestArgumentParsing:
    """
    Test argument parsing and validation.
    """

    def test_validate_args_gqa_mode_valid(self):
        """
        Test validation passes for valid GQA mode arguments.
        """
        args = Namespace(
            mode="gqa",
            emb_dim=2048,
            n_heads=24,
            n_layers=48,
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        # Should not raise
        validate_args(args)

    def test_validate_args_gqa_mode_missing_required(self):
        """
        Test validation fails for GQA mode with missing required args.
        """
        args = Namespace(
            mode="gqa",
            emb_dim=None,
            n_heads=24,
            n_layers=48,
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        with pytest.raises(ValueError, match="Mode 'gqa' requires"):
            validate_args(args)

    def test_validate_args_mla_mode_valid(self):
        """
        Test validation passes for valid MLA mode arguments.
        """
        args = Namespace(
            mode="mla",
            emb_dim=2048,
            n_heads=24,
            n_layers=48,
            latent_dims=[1024, 512],
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        validate_args(args)

    def test_validate_args_swa_mode_valid(self):
        """
        Test validation passes for valid SWA mode arguments.
        """
        args = Namespace(
            mode="swa",
            emb_dim=2048,
            n_heads=24,
            n_layers=48,
            sliding_window_size=2048,
            swa_ratio="5:1",
            n_kv_groups=4,
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        validate_args(args)

    def test_validate_args_swa_mode_missing_window(self):
        """
        Test validation fails for SWA mode without sliding window size.
        """
        args = Namespace(
            mode="swa",
            emb_dim=2048,
            n_heads=24,
            n_layers=48,
            sliding_window_size=None,
            swa_ratio="5:1",
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        with pytest.raises(ValueError, match="Mode 'swa' requires.*--sliding-window-size"):
            validate_args(args)

    def test_validate_args_moe_mode_valid(self):
        """
        Test validation passes for valid MoE mode arguments.
        """
        args = Namespace(
            mode="moe",
            emb_dim=2048,
            hidden_dim=8192,
            ffn_type="swiglu",
            num_experts=8,
            top_k=2,
            match_dense=True,
            no_log=False,
            max_experts=512,
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        validate_args(args)

    def test_validate_args_moe_mode_missing_required(self):
        """
        Test validation fails for MoE mode with missing required args.
        """
        args = Namespace(
            mode="moe",
            emb_dim=2048,
            hidden_dim=None,
            batch_size=1,
            dtype="bfloat16",
            output="test.pdf"
        )
        with pytest.raises(ValueError, match="Mode 'moe' requires"):
            validate_args(args)


class TestPlottingFunctions:
    """
    Test plotting functions create PDF files correctly.
    """

    def test_plot_gqa_creates_file(self):
        """
        Test that plot_gqa creates a PDF file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_gqa.pdf"
            args = Namespace(
                mode="gqa",
                emb_dim=768,
                n_heads=12,
                n_layers=12,
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_gqa(args)

            assert output_path.exists(), "GQA plot should create PDF file"
            assert output_path.stat().st_size > 0, "PDF should not be empty"

    def test_plot_gqa_default_output(self):
        """
        Test that plot_gqa creates default output file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)

                args = Namespace(
                    mode="gqa",
                    emb_dim=768,
                    n_heads=12,
                    n_layers=12,
                    batch_size=1,
                    dtype="bfloat16",
                    output=None
                )

                plot_gqa(args)

                default_path = Path(tmp_dir) / "kv_bytes_vs_context_length_gqa.pdf"
                assert default_path.exists(), "Should create default GQA output file"

            finally:
                os.chdir(original_cwd)

    def test_plot_mla_creates_file(self):
        """
        Test that plot_mla creates a PDF file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_mla.pdf"
            args = Namespace(
                mode="mla",
                emb_dim=768,
                n_heads=12,
                n_layers=12,
                latent_dims=[512, 256],
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_mla(args)

            assert output_path.exists(), "MLA plot should create PDF file"
            assert output_path.stat().st_size > 0, "PDF should not be empty"

    def test_plot_swa_creates_file(self):
        """
        Test that plot_swa creates a PDF file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_swa.pdf"
            args = Namespace(
                mode="swa",
                emb_dim=768,
                n_heads=12,
                n_layers=12,
                sliding_window_size=2048,
                swa_ratio="5:1",
                n_kv_groups=4,
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_swa(args)

            assert output_path.exists(), "SWA plot should create PDF file"
            assert output_path.stat().st_size > 0, "PDF should not be empty"

    def test_plot_moe_creates_file(self):
        """
        Test that plot_moe creates a PDF file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_moe.pdf"
            args = Namespace(
                mode="moe",
                emb_dim=768,
                hidden_dim=3072,
                ffn_type="swiglu",
                top_k=2,
                max_experts=16,
                match_dense=True,
                no_log=False,
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_moe(args)

            assert output_path.exists(), "MoE plot should create PDF file"
            assert output_path.stat().st_size > 0, "PDF should not be empty"

    def test_plot_moe_with_log_scale(self):
        """
        Test that plot_moe works with log scale enabled.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_moe_log.pdf"
            args = Namespace(
                mode="moe",
                emb_dim=768,
                hidden_dim=3072,
                ffn_type="gelu",
                top_k=2,
                max_experts=8,
                match_dense=False,
                no_log=False,  # Log scale enabled
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_moe(args)

            assert output_path.exists(), "MoE plot with log scale should create PDF"

    def test_plot_moe_without_log_scale(self):
        """
        Test that plot_moe works with log scale disabled.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_moe_linear.pdf"
            args = Namespace(
                mode="moe",
                emb_dim=768,
                hidden_dim=3072,
                ffn_type="swiglu",
                top_k=2,
                max_experts=8,
                match_dense=True,
                no_log=True,  # Log scale disabled
                batch_size=1,
                dtype="bfloat16",
                output=str(output_path)
            )

            plot_moe(args)

            assert output_path.exists(), "MoE plot without log scale should create PDF"


class TestDataTypeHandling:
    """
    Test that different data types are handled correctly.
    """

    def test_different_dtypes_gqa(self):
        """
        Test GQA plotting with different data types.
        """
        dtypes = ["float32", "bfloat16", "float16", "float8", "int8"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for dtype in dtypes:
                output_path = Path(tmp_dir) / f"test_gqa_{dtype}.pdf"
                args = Namespace(
                    mode="gqa",
                    emb_dim=768,
                    n_heads=12,
                    n_layers=12,
                    batch_size=1,
                    dtype=dtype,
                    output=str(output_path)
                )

                plot_gqa(args)

                assert output_path.exists(), f"Should create plot for {dtype}"

    def test_dtype_affects_memory_calculation(self):
        """
        Test that different dtypes result in different memory calculations.
        """
        fp32_bytes = kv_bytes_total_mha(1, 1024, 768, 12, bytes_per_elem=4)
        bf16_bytes = kv_bytes_total_mha(1, 1024, 768, 12, bytes_per_elem=2)
        fp8_bytes = kv_bytes_total_mha(1, 1024, 768, 12, bytes_per_elem=1)

        assert fp32_bytes > bf16_bytes > fp8_bytes, "Memory should decrease with smaller dtypes"
        assert fp32_bytes == 2 * bf16_bytes, "FP32 should use 2x memory of BF16"
        assert bf16_bytes == 2 * fp8_bytes, "BF16 should use 2x memory of FP8"
