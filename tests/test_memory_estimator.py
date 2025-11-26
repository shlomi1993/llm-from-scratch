"""
Tests for unified memory estimator tool.

This test suite covers MHA/GQA, MLA, SWA, and MoE estimation modes.
"""

import pytest
import subprocess
import sys

from src.configurations import GptConfig
from tools.memory_estimator.src import (
    estimate_mha,
    estimate_gqa,
    estimate_mla,
    estimate_swa,
    estimate_moe,
    bytes_convert,
    kv_bytes_total
)


class TestMhaEstimation:
    """
    Tests for MHA-only estimation mode.

    This tests the estimate_mha function which provides baseline MHA KV-cache
    calculations without GQA comparison.
    """

    def test_mha_basic_calculation(self):
        """
        Test MHA estimation with standard configuration.
        """
        cfg = GptConfig(
            emb_dim=4096,
            n_layers=32,
            n_heads=32,
            context_length=32768,
        )

        result = estimate_mha(cfg, batch_size=1, dtype="bfloat16")

        # Verify configuration
        assert result.bytes_per_elem == 2, "bf16 should be 2 bytes"
        assert result.head_dim == 128, "head_dim should be 4096/32 = 128"

        # Verify this is MHA-only (no compression)
        assert result.ratio == 1.0, "MHA ratio should be 1.0 (no compression)"
        assert result.savings == 0.0, "MHA savings should be 0.0 (baseline)"

        # Verify memory calculation matches expected value
        assert bytes_convert(result.total_mha) == "17.18 GB", "MHA total should be 17.18 GB"

        # For MHA-only, total_gqa should equal total_mha
        assert result.total_gqa == result.total_mha, "For MHA-only, GQA total should equal MHA total"

    def test_mha_different_dtypes(self):
        """
        Test MHA estimation with different data types.
        """
        cfg = GptConfig(emb_dim=2048, n_layers=12, n_heads=16, context_length=8192)

        # Test float32
        result_fp32 = estimate_mha(cfg, batch_size=1, dtype="float32")
        assert result_fp32.bytes_per_elem == 4, "float32 should be 4 bytes"

        # Test float16
        result_fp16 = estimate_mha(cfg, batch_size=1, dtype="float16")
        assert result_fp16.bytes_per_elem == 2, "float16 should be 2 bytes"

        # Test bfloat16
        result_bf16 = estimate_mha(cfg, batch_size=1, dtype="bfloat16")
        assert result_bf16.bytes_per_elem == 2, "bfloat16 should be 2 bytes"

        # Memory should scale with dtype size
        assert result_fp32.total_mha == 2 * result_fp16.total_mha, "fp32 should use 2x memory vs fp16"
        assert result_fp16.total_mha == result_bf16.total_mha, "fp16 and bf16 should use same memory"

    def test_mha_batch_scaling(self):
        """
        Test that MHA memory scales linearly with batch size.
        """
        cfg = GptConfig(emb_dim=1024, n_layers=12, n_heads=8, context_length=4096)

        result_batch1 = estimate_mha(cfg, batch_size=1, dtype="float16")
        result_batch4 = estimate_mha(cfg, batch_size=4, dtype="float16")
        result_batch8 = estimate_mha(cfg, batch_size=8, dtype="float16")

        # Memory should scale linearly with batch size
        assert result_batch4.total_mha == 4 * result_batch1.total_mha, "Batch size 4 should use 4x memory"
        assert result_batch8.total_mha == 8 * result_batch1.total_mha, "Batch size 8 should use 8x memory"

    def test_mha_context_scaling(self):
        """
        Test that MHA memory scales linearly with context length.
        """
        base_cfg = GptConfig(emb_dim=2048, n_layers=16, n_heads=16, context_length=2048)
        base_result = estimate_mha(base_cfg, batch_size=1, dtype="float16")

        # Double context length
        cfg_2x = GptConfig(emb_dim=2048, n_layers=16, n_heads=16, context_length=4096)
        result_2x = estimate_mha(cfg_2x, batch_size=1, dtype="float16")

        # Quadruple context length
        cfg_4x = GptConfig(emb_dim=2048, n_layers=16, n_heads=16, context_length=8192)
        result_4x = estimate_mha(cfg_4x, batch_size=1, dtype="float16")

        assert result_2x.total_mha == 2 * base_result.total_mha, "2x context length should use 2x memory"
        assert result_4x.total_mha == 4 * base_result.total_mha, "4x context length should use 4x memory"


class TestGqaEstimation:
    """
    Tests based on chapters/ch04/04_gqa/README.md example.

    Command from README:
    uv run memory_estimator_gqa.py --emb_dim 4096 --n_heads 32 --n_layers 32 \
        --context_length 32768 --n_kv_groups 4 --batch_size 1 --dtype bf16

    Expected output:
    MHA total KV cache  : 17.18 GB
    GQA total KV cache  : 4.29 GB
    Ratio (MHA / GQA)   : 4.00x
    Savings (GQA vs MHA): 75.00%
    """

    def test_gqa_readme_example(self):
        """
        Test GQA estimation matches README example output.
        """
        cfg = GptConfig(
            emb_dim=4096,
            n_layers=32,
            n_heads=32,
            context_length=32768,
        )

        result = estimate_gqa(cfg, batch_size=1, dtype="bfloat16", n_kv_groups=4)

        # Verify configuration
        assert result.bytes_per_elem == 2, "bf16 should be 2 bytes"
        assert result.head_dim == 128, "head_dim should be 4096/32 = 128"
        assert result.n_kv_heads_gqa == 8, "n_kv_heads should be 32/4 = 8"

        # Verify memory calculations match README
        assert bytes_convert(result.total_mha) == "17.18 GB", "MHA total should match README (17.18 GB)"
        assert bytes_convert(result.total_gqa) == "4.29 GB", "GQA total should match README (4.29 GB)"

        # Verify compression metrics
        assert abs(result.ratio - 4.0) < 0.01, "Ratio should be 4.00x"
        assert abs(result.savings - 0.75) < 0.01, "Savings should be 75%"

    def test_gqa_invalid_n_kv_groups(self):
        """
        Test that invalid n_kv_groups raises ValueError.
        """
        cfg = GptConfig(emb_dim=4096, n_layers=32, n_heads=32, context_length=32768)

        # n_kv_groups must divide n_heads exactly
        with pytest.raises(ValueError, match="n_kv_groups must divide n_heads"):
            estimate_gqa(cfg, batch_size=1, dtype="bfloat16", n_kv_groups=5)

    def test_gqa_scaling_with_parameters(self):
        """
        Test that GQA memory scales correctly with model parameters.
        """
        base_cfg = GptConfig(emb_dim=2048, n_layers=12, n_heads=16, context_length=8192)
        base_result = estimate_gqa(base_cfg, batch_size=1, dtype="float16", n_kv_groups=4)

        # Double context length -> double memory
        cfg_2x_context = GptConfig(emb_dim=2048, n_layers=12, n_heads=16, context_length=16384)
        result_2x = estimate_gqa(cfg_2x_context, batch_size=1, dtype="float16", n_kv_groups=4)
        assert result_2x.total_gqa == 2 * base_result.total_gqa, "2x context should double GQA memory"

        # Double layers -> double memory
        cfg_2x_layers = GptConfig(emb_dim=2048, n_layers=24, n_heads=16, context_length=8192)
        result_layers = estimate_gqa(cfg_2x_layers, batch_size=1, dtype="float16", n_kv_groups=4)
        assert result_layers.total_gqa == 2 * base_result.total_gqa, "2x layers should double GQA memory"


class TestMlaEstimation:
    """
    Tests based on chapters/ch04/05_mla/README.md example.

    Command from README:
    uv run memory_estimator_mla.py --context_length 8192 --emb_dim 2048 --n_heads 24 \
        --n_layers 48 --n_kv_groups 4 --batch_size 1 --dtype bf16 --latent_dim 1024

    Expected output:
    MHA total KV cache  : 3.25 GB
    GQA total KV cache  : 0.81 GB
    MLA total KV cache  : 0.81 GB
    Ratio (MHA / GQA)   : 4.00x
    Savings (GQA vs MHA): 75.00%
    Ratio (MHA / MLA)   : 4.03x
    Savings (MLA vs MHA): 75.19%
    """

    def test_mla_readme_example(self):
        """
        Test MLA estimation matches README example output.
        """
        cfg = GptConfig(
            context_length=8192,
            emb_dim=2048,
            n_heads=24,
            n_layers=48,
        )

        result = estimate_mla(
            cfg,
            batch_size=1,
            dtype="bfloat16",
            latent_dim=1024,
            n_kv_groups=4
        )

        # Verify configuration
        assert result.bytes_per_elem == 2, "bf16 should be 2 bytes"
        assert result.head_dim == 86, "head_dim should be ceil(2048/24) = 86"  # math.ceil
        assert result.n_kv_heads_gqa == 6, "n_kv_heads should be 24/4 = 6"
        assert result.latent_dim == 1024, "latent_dim should match input (1024)"

        # Verify memory calculations match README
        assert bytes_convert(result.total_mha) == "3.25 GB"
        assert bytes_convert(result.total_gqa) == "0.81 GB"
        assert bytes_convert(result.total_mla) == "0.81 GB"

        # Verify compression metrics for GQA
        assert abs(result.ratio - 4.0) < 0.01, "GQA ratio should be 4.00x"
        assert abs(result.savings - 0.75) < 0.01, "GQA savings should be 75%"

        # Verify compression metrics for MLA
        assert abs(result.ratio_mha_mla - 4.03) < 0.02, "MLA ratio should be ~4.03x"
        assert abs(result.savings_mla - 0.7519) < 0.01, "MLA savings should be ~75.19%"

    def test_mla_compression_effect(self):
        """
        Test that MLA compression (latent_dim) affects memory usage.
        """
        cfg = GptConfig(emb_dim=2048, n_layers=24, n_heads=16, context_length=4096)

        # Higher latent_dim = more memory
        result_high = estimate_mla(cfg, 1, "float16", latent_dim=2048, n_kv_groups=4)
        result_low = estimate_mla(cfg, 1, "float16", latent_dim=512, n_kv_groups=4)

        assert result_high.total_mla > result_low.total_mla, "Higher latent_dim should use more memory"
        assert result_low.total_mla == result_high.total_mla / 4, "latent_dim ratio should match memory ratio (2048/512 = 4)"

    def test_mla_better_than_gqa(self):
        """
        Test that MLA can achieve better compression than GQA with proper latent_dim.
        """
        cfg = GptConfig(emb_dim=4096, n_layers=32, n_heads=32, context_length=16384)

        # Small latent_dim should compress better than GQA
        result = estimate_mla(cfg, 1, "bfloat16", latent_dim=512, n_kv_groups=4)

        # MLA should use less memory than GQA
        assert result.total_mla < result.total_gqa, "MLA with small latent_dim should use less memory than GQA"
        assert result.savings_mla > result.savings, "MLA savings should be greater than GQA savings"


class TestSwaEstimation:
    """
    Tests based on chapters/ch04/06_swa/README.md example.

    Command from README:
    uv run memory_estimator_swa.py --emb_dim 4096 --n_heads 32 --n_layers 32 \
        --context_length 32768 --n_kv_groups 4 --batch_size 1 --dtype bf16 \
        --sliding_window_size 1024 --swa_ratio "5:1"

    Expected output:
    MHA KV total           : 17.18 GB
    GQA KV total           : 4.29 GB
    MHA + SWA (Ratio: 5:1) : 3.14 GB
    MHA + GQA (Ratio: 5:1) : 0.78 GB (this is GQA + SWA)
    """

    def test_swa_readme_example(self):
        """
        Test SWA estimation matches README example output.
        """
        cfg = GptConfig(
            emb_dim=4096,
            n_heads=32,
            n_layers=32,
            context_length=32768,
        )

        result = estimate_swa(
            cfg,
            batch_size=1,
            dtype="bfloat16",
            n_kv_groups=4,
            sliding_window_size=1024,
            swa_ratio="5:1"
        )

        # Verify configuration
        assert result.bytes_per_elem == 2, "bf16 should be 2 bytes"
        assert result.head_dim == 128, "head_dim should be 4096/32 = 128"
        assert result.n_kv_heads_gqa == 8, "n_kv_heads should be 32/4 = 8"
        assert result.eff_W == 1024, "Effective window should be min(32768, 1024) = 1024"

        # Verify layer distribution (5:1 ratio with 32 layers)
        # 32 layers / 6 = 5 blocks with 2 remaining
        # 5 blocks * 5 SWA + min(5, 2) = 25 + 2 = 27 SWA layers
        # 5 blocks * 1 Full + max(0, 2-5) = 5 + 0 = 5 Full layers
        assert result.n_swa_layers == 27, "Should have 27 SWA layers"
        assert result.n_full_layers == 5, "Should have 5 full layers"

        # Verify memory calculations match README
        assert bytes_convert(result.total_mha_all_full) == "17.18 GB"
        assert bytes_convert(result.total_gqa_all_full) == "4.29 GB"
        assert bytes_convert(result.total_mixed_mha) == "3.14 GB"
        assert bytes_convert(result.total_mixed_gqa) == "0.78 GB"

    def test_swa_all_swa_layers(self):
        """
        Test SWA with all layers using sliding window (1:0 ratio).
        """
        cfg = GptConfig(emb_dim=2048, n_heads=16, n_layers=24, context_length=16384)

        result = estimate_swa(
            cfg, 1, "float16",
            n_kv_groups=2,
            sliding_window_size=2048,
            swa_ratio="1:0"
        )

        assert result.n_swa_layers == 24, "All layers should be SWA"
        assert result.n_full_layers == 0, "No full attention layers"

        # Memory should be significantly lower than full attention
        assert result.total_mixed_mha < result.total_mha_all_full, "SWA should use less memory than full attention"
        # Ratio should be context_length / window_size = 16384 / 2048 = 8
        assert abs(result.total_mha_all_full / result.total_mixed_mha - 8.0) < 0.1, "Memory ratio should match context/window ratio (~8.0)"

    def test_swa_ratio_parsing(self):
        """
        Test different SWA:Full ratios.
        """
        cfg = GptConfig(emb_dim=1024, n_heads=8, n_layers=12, context_length=8192)

        # 1:1 ratio (equal split)
        result_1_1 = estimate_swa(cfg, 1, "float16", 2, 1024, "1:1")
        assert result_1_1.n_swa_layers == 6, "1:1 ratio should have 6 SWA layers"
        assert result_1_1.n_full_layers == 6, "1:1 ratio should have 6 full layers"

        # 3:1 ratio
        result_3_1 = estimate_swa(cfg, 1, "float16", 2, 1024, "3:1")
        assert result_3_1.n_swa_layers == 9, "3:1 ratio should have 9 SWA layers"  # 12/4 = 3 blocks, 3*3 = 9
        assert result_3_1.n_full_layers == 3, "3:1 ratio should have 3 full layers"


class TestMoeEstimation:
    """
    Tests based on chapters/ch04/07_moe/README.md example.

    Command from README:
    uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
        --num_experts 8 --top_k 2 --match_dense

    Expected output:
    Dense FFN params       : 308,281,344 (0.62 GB)
    Per-expert params      : 38,535,168 (0.08 GB)
    Router params          : 57,344 (0.00 GB)
    MoE TOTAL params       : 308,338,688 (0.62 GB)
    MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
    moe_hidden_size        : 1792
    """

    def test_moe_readme_example(self):
        """
        Test MoE estimation matches README example output.
        """
        result = estimate_moe(
            emb_dim=7168,
            hidden_dim=14336,
            ffn_type="swiglu",
            num_experts=8,
            top_k=2,
            dtype="bfloat16",
            match_dense=True
        )

        # Verify configuration
        assert result.bytes_per_elem == 2, "bf16 should be 2 bytes"
        assert result.moe_hidden_dim == 1792, "With match_dense, moe_hidden should be ~1792"

        # Verify parameter counts match README
        assert result.dense_params == 308_281_344, "Dense FFN params should be 308,281,344"
        assert result.per_expert_params == 38_535_168, "Per-expert params should be 38,535,168"
        assert result.router == 57_344, "Router params should be 57,344"
        assert result.moe_total == 308_338_688, "MoE total should be 308,338,688"
        assert result.moe_active_params_per_token == 77_127_680, "Active params should be 77,127,680"

    def test_moe_active_params_scaling(self):
        """
        Test that active params scale with top_k.
        """
        base_result = estimate_moe(
            emb_dim=4096,
            hidden_dim=8192,
            ffn_type="gelu",
            num_experts=16,
            top_k=2,
            dtype="float16",
            match_dense=False
        )

        # Double top_k -> roughly double active params (plus router overhead)
        result_4k = estimate_moe(
            emb_dim=4096,
            hidden_dim=8192,
            ffn_type="gelu",
            num_experts=16,
            top_k=4,
            dtype="float16",
            match_dense=False
        )

        # Active params should scale with top_k
        # active = router + top_k * per_expert
        expected_ratio = (base_result.router + 4 * base_result.per_expert_params) / \
                        (base_result.router + 2 * base_result.per_expert_params)
        actual_ratio = result_4k.moe_active_params_per_token / base_result.moe_active_params_per_token

        assert abs(actual_ratio - expected_ratio) < 0.01, "Active params ratio should match expected based on top_k"

    def test_moe_gelu_vs_swiglu(self):
        """
        Test that SwiGLU has more parameters than GELU (3 matrices vs 2).
        """
        result_gelu = estimate_moe(
            emb_dim=2048,
            hidden_dim=4096,
            ffn_type="gelu",
            num_experts=4,
            top_k=1,
            dtype="float16",
            match_dense=False
        )

        result_swiglu = estimate_moe(
            emb_dim=2048,
            hidden_dim=4096,
            ffn_type="swiglu",
            num_experts=4,
            top_k=1,
            dtype="float16",
            match_dense=False
        )

        # SwiGLU uses 3 parameter matrices, GELU uses 2
        # So SwiGLU should have 1.5x the parameters
        assert result_swiglu.per_expert_params == result_gelu.per_expert_params * 1.5, "SwiGLU should have 1.5x params vs GELU (3 matrices vs 2)"

    def test_moe_match_dense_mode(self):
        """
        Test that match_dense mode adjusts hidden_dim to match dense params.
        """
        dense_params = 100_000_000  # Approximate target

        result = estimate_moe(
            emb_dim=4096,
            hidden_dim=8192,
            ffn_type="swiglu",
            num_experts=8,
            top_k=2,
            dtype="float16",
            match_dense=True
        )

        # MoE total should be approximately equal to dense params
        # (within router params difference)
        assert abs(result.moe_total - result.dense_params) < result.router * 2, "MoE total should approximately match dense params in match_dense mode"


class TestCliIntegration:
    """Test CLI interface for all modes with README examples."""

    def test_cli_gqa_readme_example(self):
        """
        Test GQA mode CLI with README example.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.memory_estimator.memory_estimator",
                "--mode", "gqa",
                "--emb-dim", "4096",
                "--n-heads", "32",
                "--n-layers", "32",
                "--context-length", "32768",
                "--n-kv-groups", "4",
                "--batch-size", "1",
                "--dtype", "bfloat16"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "CLI should exit successfully"
        assert "17.18 GB" in result.stdout, "Output should contain MHA total (17.18 GB)"
        assert "4.29 GB" in result.stdout, "Output should contain GQA total (4.29 GB)"
        assert "4.00x" in result.stdout, "Output should contain ratio (4.00x)"
        assert "75.00%" in result.stdout, "Output should contain savings (75.00%)"

    def test_cli_mla_readme_example(self):
        """
        Test MLA mode CLI with README example.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.memory_estimator.memory_estimator",
                "--mode", "mla",
                "--emb-dim", "2048",
                "--n-heads", "24",
                "--n-layers", "48",
                "--context-length", "8192",
                "--n-kv-groups", "4",
                "--latent-dim", "1024",
                "--batch-size", "1",
                "--dtype", "bfloat16"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "CLI should exit successfully"
        assert "3.25 GB" in result.stdout, "Output should contain MHA total (3.25 GB)"
        assert "0.81 GB" in result.stdout, "Output should contain GQA and MLA totals (0.81 GB)"
        assert "75.19%" in result.stdout, "Output should contain MLA savings (75.19%)"

    def test_cli_swa_readme_example(self):
        """
        Test SWA mode CLI with README example.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.memory_estimator.memory_estimator",
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
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "CLI should exit successfully"
        assert "17.18 GB" in result.stdout, "Output should contain MHA total (17.18 GB)"
        assert "4.29 GB" in result.stdout, "Output should contain GQA total (4.29 GB)"
        assert "3.14 GB" in result.stdout, "Output should contain MHA+SWA (3.14 GB)"
        assert "0.78 GB" in result.stdout, "Output should contain GQA+SWA (0.78 GB)"

    def test_cli_moe_readme_example(self):
        """
        Test MoE mode CLI with README example.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.memory_estimator.memory_estimator",
                "--mode", "moe",
                "--emb-dim", "7168",
                "--hidden-dim", "14336",
                "--ffn-type", "swiglu",
                "--num-experts", "8",
                "--top-k", "2",
                "--dtype", "bfloat16",
                "--match-dense"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "CLI should exit successfully"
        assert "308,281,344" in result.stdout, "Output should contain dense params (308,281,344)"
        assert "308,338,688" in result.stdout, "Output should contain MoE total (308,338,688)"
        assert "77,127,680" in result.stdout, "Output should contain active params/token (77,127,680)"

    def test_cli_multi_mode(self):
        """
        Test multi-mode execution (MHA + GQA).
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.memory_estimator.memory_estimator",
                "--mode", "mha", "gqa",
                "--emb-dim", "2048",
                "--n-heads", "16",
                "--n-layers", "12",
                "--n-kv-groups", "4",
                "--context-length", "8192",
                "--batch-size", "1",
                "--dtype", "float16"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "CLI should exit successfully"
        # Both modes should produce output
        assert "MHA total KV cache" in result.stdout, "Output should contain MHA results"
        assert "GQA total KV cache" in result.stdout, "Output should contain GQA results"

    def test_bytes_convert_and_kv_bytes(self):
        """
        Test basic utility functions.
        """

        # Bytes conversion
        assert bytes_convert(1_000_000_000) == "1.00 GB", "Should convert 1B bytes to 1.00 GB"
        assert bytes_convert(17_179_869_184) == "17.18 GB", "Should convert bytes to 17.18 GB"

        # KV-cache: batch=1, context=1024, emb_dim=768, n_heads=12, n_layers=12, fp32
        result = kv_bytes_total(1, 1024, 768, 12, 12, 12, 4)
        assert result > 0, "KV cache bytes should be positive"
        # KV cache stores 2 tensors (K and V) per layer
        # Each is: batch * context * emb_dim * bytes_per_elem
        expected_bytes = 2 * 1 * 1024 * 768 * 12 * 4  # 2 (K,V) * batch * context * emb * n_layers * bytes
        assert result == expected_bytes, f"KV-cache bytes should be {expected_bytes}, got {result}"
