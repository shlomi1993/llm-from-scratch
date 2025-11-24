import pytest
import torch

from typing import Dict, List

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M


class TestPerformanceAnalysis:
    """
    Test suite for performance analysis and FLOPS computation.
    """

    @pytest.fixture
    def sample_configs(self) -> List[GptConfig]:
        """
        Sample configurations for testing.
        """
        return [
            GptConfig(emb_dim=64, n_layers=2, n_heads=2, vocab_size=1000, context_length=32, drop_rate=0.0, qkv_bias=False),
            GptConfig(emb_dim=128, n_layers=4, n_heads=4, vocab_size=2000, context_length=64, drop_rate=0.0, qkv_bias=False),
        ]

    @pytest.fixture
    def gpu_specs(self) -> Dict[str, Dict[torch.dtype, float]]:
        """
        GPU FLOPS specifications.
        """
        return {
            "H100": {
                torch.float32: 51.22e12,
                torch.float16: 204.9e12,
                torch.bfloat16: 204.9e12
            },
            "A100": {
                torch.float32: 19.49e12,
                torch.float16: 77.97e12,
                torch.bfloat16: 77.97e12
            },
            "RTX_3080": {
                torch.float32: 29.77e12,
                torch.float16: 29.77e12,
                torch.bfloat16: 29.77e12
            },
        }

    def calculate_attention_flops(self, emb_dim: int, n_heads: int, seq_len: int, batch_size: int) -> int:
        """
        Calculate FLOPS for multi-head attention layer.
        """
        head_dim = emb_dim // n_heads

        # QKV projections: 3 * (batch_size * seq_len * emb_dim * emb_dim)
        qkv_flops = 3 * batch_size * seq_len * emb_dim * emb_dim

        # Attention scores: batch_size * n_heads * seq_len * seq_len * head_dim
        attention_scores_flops = batch_size * n_heads * seq_len * seq_len * head_dim

        # Attention output: batch_size * n_heads * seq_len * seq_len * head_dim
        attention_output_flops = batch_size * n_heads * seq_len * seq_len * head_dim

        # Output projection: batch_size * seq_len * emb_dim * emb_dim
        output_proj_flops = batch_size * seq_len * emb_dim * emb_dim

        return qkv_flops + attention_scores_flops + attention_output_flops + output_proj_flops

    def calculate_feedforward_flops(self, emb_dim: int, seq_len: int, batch_size: int) -> int:
        """
        Calculate FLOPS for feed-forward network.
        """
        d_ff = 4 * emb_dim  # Standard 4x expansion

        # First linear layer: batch_size * seq_len * emb_dim * d_ff
        first_linear = batch_size * seq_len * emb_dim * d_ff

        # Second linear layer: batch_size * seq_len * d_ff * emb_dim
        second_linear = batch_size * seq_len * d_ff * emb_dim

        return first_linear + second_linear

    def calculate_total_flops(self, config: GptConfig, batch_size: int, include_backward: bool = False) -> Dict[str, int]:
        """
        Calculate total FLOPS for forward pass (and optionally backward pass).
        """
        emb_dim = config.emb_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        seq_len = config.context_length

        # Per-layer FLOPS
        attention_flops = self.calculate_attention_flops(emb_dim, n_heads, seq_len, batch_size)
        feedforward_flops = self.calculate_feedforward_flops(emb_dim, seq_len, batch_size)

        # Total for one transformer block
        block_flops = attention_flops + feedforward_flops

        # Total for all layers (forward pass)
        forward_flops = block_flops * n_layers

        # Add embedding and output projection FLOPS
        embedding_flops = batch_size * seq_len * config.vocab_size * emb_dim
        total_forward_flops = forward_flops + embedding_flops

        # Backward pass typically requires ~2x forward FLOPS
        backward_flops = 2 * total_forward_flops if include_backward else 0
        total_flops = total_forward_flops + backward_flops

        return {
            "attention_flops_per_layer": attention_flops,
            "feedforward_flops_per_layer": feedforward_flops,
            "block_flops": block_flops,
            "forward_flops": total_forward_flops,
            "backward_flops": backward_flops,
            "total_flops": total_flops
        }

    def calculate_model_parameters(self, config: GptConfig) -> Dict[str, int]:
        """
        Calculate model parameters for GPT configuration.
        """
        emb_dim = config.emb_dim
        vocab_size = config.vocab_size
        context_length = config.context_length
        n_layers = config.n_layers

        # Token embedding parameters
        token_embedding_params = vocab_size * emb_dim

        # Position embedding parameters
        pos_embedding_params = context_length * emb_dim

        # Attention parameters per layer (QKV + output projection)
        attention_params_per_layer = 4 * emb_dim * emb_dim

        # Feed-forward parameters per layer
        ff_params_per_layer = 2 * emb_dim * (4 * emb_dim)  # 4x expansion

        # Layer norm parameters per layer (2 layer norms per transformer block)
        ln_params_per_layer = 2 * 2 * emb_dim  # scale and shift for each layer norm

        # Total transformer block parameters
        block_params = attention_params_per_layer + ff_params_per_layer + ln_params_per_layer

        # Final layer norm
        final_ln_params = 2 * emb_dim

        # Output projection (often tied with token embedding)
        output_proj_params = vocab_size * emb_dim

        total_params = (
            token_embedding_params +
            pos_embedding_params +
            block_params * n_layers +
            final_ln_params +
            output_proj_params
        )

        # With weight tying (output projection shares weights with token embedding)
        tied_params = total_params - output_proj_params

        return {
            "token_embedding": token_embedding_params,
            "position_embedding": pos_embedding_params,
            "attention_per_layer": attention_params_per_layer,
            "feedforward_per_layer": ff_params_per_layer,
            "layernorm_per_layer": ln_params_per_layer,
            "block_params": block_params,
            "final_layernorm": final_ln_params,
            "output_projection": output_proj_params,
            "total_params": total_params,
            "tied_params": tied_params
        }

    def calculate_mfu(self, tokens_per_second: float, total_flops: int, tokens_processed: int, max_flops_per_second: float) -> float:
        """
        Calculate Model FLOPS Utilization (MFU).
        """
        if tokens_processed == 0 or max_flops_per_second == 0:
            return 0.0

        flops_per_token = total_flops / tokens_processed
        theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token

        if theoretical_max_tokens_per_second == 0:
            return 0.0

        return tokens_per_second / theoretical_max_tokens_per_second

    def test_flops_calculation_attention(self, sample_configs: List[GptConfig]) -> None:
        """
        Test FLOPS calculation for attention mechanism.
        """
        config = sample_configs[0]
        seq_len = 32
        batch_size = 4

        attention_flops = self.calculate_attention_flops(config.emb_dim, config.n_heads, seq_len, batch_size)

        # Attention FLOPS should be positive
        assert attention_flops > 0, "Attention FLOPS should be positive"

        # Should scale quadratically with sequence length
        attention_flops_longer = self.calculate_attention_flops(config.emb_dim, config.n_heads, seq_len * 2, batch_size)
        assert attention_flops_longer > attention_flops * 3, "FLOPS should scale more than linearly with sequence length due to O(n²) attention"

    def test_flops_calculation_feedforward(self, sample_configs: List[GptConfig]) -> None:
        """
        Test FLOPS calculation for feed-forward network.
        """
        config = sample_configs[0]
        seq_len = 32
        batch_size = 4

        ff_flops = self.calculate_feedforward_flops(config.emb_dim, seq_len, batch_size)

        # Feed-forward FLOPS should be positive
        assert ff_flops > 0, "Feed-forward FLOPS should be positive"

        # Should scale linearly with sequence length
        ff_flops_longer = self.calculate_feedforward_flops(config.emb_dim, seq_len * 2, batch_size)
        assert abs(ff_flops_longer - ff_flops * 2) < ff_flops * 0.01, "FF FLOPS should scale linearly with sequence length"

    def test_flops_scaling_with_batch_size(self) -> None:
        """
        Test that FLOPS scale linearly with batch size.
        """
        flops_batch_1 = self.calculate_total_flops(GPT_CONFIG_124M, 1)["total_flops"]
        flops_batch_4 = self.calculate_total_flops(GPT_CONFIG_124M, 4)["total_flops"]

        # Should scale linearly with batch size
        assert abs(flops_batch_4 - flops_batch_1 * 4) < flops_batch_1 * 0.01, "FLOPS should scale linearly with batch size"

    def test_flops_scaling_with_model_size(self) -> None:
        """
        Test that FLOPS scale appropriately with model size.
        """
        small_config = GPT_CONFIG_124M
        large_config = GPT_CONFIG_774M

        batch_size = 2

        small_flops = self.calculate_total_flops(small_config, batch_size)["total_flops"]
        large_flops = self.calculate_total_flops(large_config, batch_size)["total_flops"]

        # Larger model should have more FLOPS
        assert large_flops > small_flops, "Larger model should have more FLOPS"

        # Should scale roughly with parameter count squared
        small_emb = small_config.emb_dim
        large_emb = large_config.emb_dim
        param_ratio = (large_emb / small_emb) ** 2
        flops_ratio = large_flops / small_flops

        assert flops_ratio > param_ratio * 0.5, "FLOPS should scale with model size"

    @pytest.mark.parametrize("config,expected_params", [
        (GPT_CONFIG_124M, 124_000_000),
        (GPT_CONFIG_355M, 355_000_000),
        (GPT_CONFIG_774M, 774_000_000),
        (GPT_CONFIG_1558M, 1_558_000_000),
    ])
    def test_parameter_counts_standard_configs(self, config: GptConfig, expected_params: int) -> None:
        """
        Test parameter counts for standard GPT configurations.
        """

        params = self.calculate_model_parameters(config)

        # Test parameter count is in the right ballpark (within 20%)
        param_ratio = abs(params["tied_params"] - expected_params) / expected_params
        assert param_ratio < 0.2, f"Parameter count {params['tied_params']:,} differs from expected {expected_params:,} by {param_ratio:.1%} (should be within 20%)"

    def test_parameter_breakdown_consistency(self) -> None:
        """
        Test that parameter breakdown sums correctly.
        """
        params = self.calculate_model_parameters(GPT_CONFIG_124M)

        # Calculate sum of components
        calculated_total = (
            params["token_embedding"] +
            params["position_embedding"] +
            params["block_params"] * GPT_CONFIG_124M.n_layers +
            params["final_layernorm"] +
            params["output_projection"]
        )

        assert params["total_params"] == calculated_total, "Parameter breakdown should sum to total"

        # Tied parameters should exclude output projection
        expected_tied = calculated_total - params["output_projection"]
        assert params["tied_params"] == expected_tied, "Tied parameters calculation incorrect"

    def test_mfu_calculation(self, gpu_specs: Dict[str, Dict[torch.dtype, float]]) -> None:
        """
        Test Model FLOPS Utilization calculation.
        """
        # Test basic MFU calculation
        tokens_per_second = 1000.0
        total_flops = 1e12
        tokens_processed = 100
        max_flops_per_second = gpu_specs["A100"][torch.bfloat16]

        mfu = self.calculate_mfu(tokens_per_second, total_flops, tokens_processed, max_flops_per_second)

        assert 0 <= mfu <= 1, "MFU should be between 0 and 1"

        # Test edge cases
        assert self.calculate_mfu(0, total_flops, tokens_processed, max_flops_per_second) == 0, "MFU should be 0 when tokens_per_second is 0"
        assert self.calculate_mfu(tokens_per_second, total_flops, 0, max_flops_per_second) == 0, "MFU should be 0 when tokens_processed is 0"
        assert self.calculate_mfu(tokens_per_second, total_flops, tokens_processed, 0) == 0, "MFU should be 0 when max_flops_per_second is 0"

    def test_backward_pass_flops_estimation(self) -> None:
        """
        Test that backward pass FLOPS are estimated correctly.
        """
        batch_size = 2

        forward_only = self.calculate_total_flops(GPT_CONFIG_124M, batch_size, include_backward=False)
        forward_backward = self.calculate_total_flops(GPT_CONFIG_124M, batch_size, include_backward=True)

        # Backward pass should be ~2x forward pass
        expected_total = forward_only["forward_flops"] * 3  # forward + 2*forward
        actual_total = forward_backward["total_flops"]

        assert abs(actual_total - expected_total) < expected_total * 0.01, "Backward pass should be ~2x forward pass FLOPS"

    def test_model_size_in_memory(self) -> None:
        """
        Test model memory size calculation.
        """
        params = self.calculate_model_parameters(GPT_CONFIG_124M)

        # Calculate model size in MB (assuming float32 = 4 bytes)
        size_mb_float32 = params["tied_params"] * 4 / (1024 ** 2)
        size_mb_bfloat16 = params["tied_params"] * 2 / (1024 ** 2)

        # Rough sanity checks
        assert 400 < size_mb_float32 < 800, f"FP32 model size {size_mb_float32:.1f}MB seems wrong"
        assert 200 < size_mb_bfloat16 < 400, f"BF16 model size {size_mb_bfloat16:.1f}MB seems wrong"

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_flops_batch_scaling(self, sample_configs: List[GptConfig], batch_size: int) -> None:
        """
        Test FLOPS scaling with different batch sizes.
        """
        config = sample_configs[0]

        flops = self.calculate_total_flops(config, batch_size)

        # FLOPS should scale linearly with batch size
        expected_flops = self.calculate_total_flops(config, 1)["total_flops"] * batch_size
        actual_flops = flops["total_flops"]

        assert abs(actual_flops - expected_flops) < expected_flops * 0.01, \
            f"FLOPS should scale linearly with batch size {batch_size}"

    def test_attention_vs_feedforward_flops_ratio(self) -> None:
        """
        Test the ratio of attention to feedforward FLOPS.
        """
        batch_size = 1
        flops = self.calculate_total_flops(GPT_CONFIG_124M, batch_size)

        attention_flops = flops["attention_flops_per_layer"]
        feedforward_flops = flops["feedforward_flops_per_layer"]

        # For long sequences, attention should dominate
        # For short sequences, feedforward might dominate
        ratio = attention_flops / feedforward_flops

        assert ratio > 0, "Both attention and feedforward should have positive FLOPS"
        # The exact ratio depends on sequence length and model dimensions

    def test_context_length_scaling(self, sample_configs: List[GptConfig]) -> None:
        """
        Test how FLOPS scale with context length.
        """
        config = sample_configs[0]
        batch_size = 1

        short_context = GptConfig(
            emb_dim=config.emb_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            vocab_size=config.vocab_size,
            context_length=32,
            drop_rate=config.drop_rate,
            qkv_bias=config.qkv_bias
        )

        long_context = GptConfig(
            emb_dim=config.emb_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            vocab_size=config.vocab_size,
            context_length=64,
            drop_rate=config.drop_rate,
            qkv_bias=config.qkv_bias
        )

        short_flops = self.calculate_total_flops(short_context, batch_size)["total_flops"]
        long_flops = self.calculate_total_flops(long_context, batch_size)["total_flops"]

        # FLOPS should increase more than linearly due to O(n²) attention
        flops_ratio = long_flops / short_flops
        context_ratio = 64 / 32

        assert flops_ratio > context_ratio, "FLOPS should scale more than linearly with context length"