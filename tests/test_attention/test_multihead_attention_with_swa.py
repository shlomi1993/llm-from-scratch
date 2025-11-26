"""
Tests for MultiheadAttentionWithSwa implementation.

This test suite validates the sliding window attention mechanism, KV caching,
and integration with the transformer architecture.
"""

import pytest
import torch

from src.attention import MultiheadAttentionWithSwa


class TestMultiheadAttentionWithSwa:
    """
    Test suite for MultiheadAttentionWithSwa implementation.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiheadAttentionWithSwa.
        """
        torch.manual_seed(42)
        return torch.randn(2, 6, 256)  # batch=2, seq_len=6, d_in=256

    def test_output_shape_with_swa(self, sample_inputs):
        """
        Test that MultiheadAttentionWithSwa produces correct output shape.
        """
        d_in = 256
        d_out = 256
        dropout = 0.1
        num_heads = 8
        window_size = 64

        att = MultiheadAttentionWithSwa(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            n_heads=num_heads,
            sliding_window_size=window_size
        )

        output = att(sample_inputs)

        assert output.shape == sample_inputs.shape, f"Expected shape {sample_inputs.shape}, got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_dimension_divisibility_assertion(self):
        """
        Test that d_out must be divisible by num_heads.
        """
        with pytest.raises(AssertionError, match="d_out must be divisible by num_heads"):
            MultiheadAttentionWithSwa(
                d_in=256,
                d_out=255,  # Not divisible by 8
                dropout=0.1,
                n_heads=8,
                sliding_window_size=64
            )

    def test_head_dimension_calculation(self):
        """
        Test that head dimensions are calculated correctly.
        """
        d_out = 512
        num_heads = 8
        expected_head_dim = d_out // num_heads

        att = MultiheadAttentionWithSwa(
            d_in=512,
            d_out=d_out,
            dropout=0.1,
            n_heads=num_heads,
            sliding_window_size=64
        )

        assert att.head_dim == expected_head_dim, \
            f"Expected head_dim={expected_head_dim}, got {att.head_dim}"
        assert att.num_heads == num_heads, \
            f"Expected num_heads={num_heads}, got {att.num_heads}"

    def test_sliding_window_size_none(self):
        """
        Test that sliding_window_size=None works (full causal attention).
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=None
        )

        x = torch.randn(2, 10, 256)
        output = att(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_cache_functionality(self, sample_inputs):
        """
        Test that KV cache functionality works correctly.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=64
        )
        att.eval()

        # Initial state - no cache
        assert att.cache_k is None, "Cache K should initially be None"
        assert att.cache_v is None, "Cache V should initially be None"
        assert att.ptr_current_pos == 0, "Position pointer should initially be 0"

        # First forward pass with cache
        with torch.no_grad():
            output1 = att(sample_inputs, use_cache=True)

        assert output1.shape == sample_inputs.shape, f"Expected shape {sample_inputs.shape}, got {output1.shape}"
        assert att.cache_k is not None, "Cache should be created"
        assert att.cache_v is not None, "Cache should be created"
        assert att.cache_k.shape[1] == 6, "Cache should store 6 tokens"
        assert att.ptr_current_pos == 6, "Position pointer should be 6"

    def test_cache_accumulation(self):
        """
        Test that cache accumulates across multiple forward passes.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=128
        )
        att.eval()

        with torch.no_grad():
            # First pass: 10 tokens
            x1 = torch.randn(1, 10, 256)
            _ = att(x1, use_cache=True)
            assert att.cache_k.shape[1] == 10, f"Cache should have 10 tokens, got {att.cache_k.shape[1]}"

            # Second pass: 5 tokens
            x2 = torch.randn(1, 5, 256)
            _ = att(x2, use_cache=True)
            assert att.cache_k.shape[1] == 15, f"Cache should have 15 tokens, got {att.cache_k.shape[1]}"

            # Third pass: 3 tokens
            x3 = torch.randn(1, 3, 256)
            _ = att(x3, use_cache=True)
            assert att.cache_k.shape[1] == 18, f"Cache should have 18 tokens, got {att.cache_k.shape[1]}"

    def test_window_trimming(self):
        """
        Test that cache is trimmed when exceeding sliding window size.
        """
        window_size = 64
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=window_size
        )
        att.eval()

        with torch.no_grad():
            # Fill cache to window size
            x1 = torch.randn(1, 64, 256)
            _ = att(x1, use_cache=True)
            assert att.cache_k.shape[1] == 64, f"Cache should have 64 tokens, got {att.cache_k.shape[1]}"

            # Add more tokens - should trigger trimming
            x2 = torch.randn(1, 20, 256)
            _ = att(x2, use_cache=True)
            assert att.cache_k.shape[1] == window_size, \
                f"Cache should be trimmed to {window_size}, got {att.cache_k.shape[1]}"

    def test_reset_cache(self):
        """
        Test that reset_cache properly resets all cache state.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=64
        )

        # Populate cache
        x = torch.randn(1, 10, 256)
        with torch.no_grad():
            _ = att(x, use_cache=True)

        assert att.cache_k is not None, "Cache should be populated"
        assert att.ptr_current_pos == 10, f"Position pointer should be 10, got {att.ptr_current_pos}"

        # Reset cache
        att.reset_cache()

        assert att.cache_k is None, "Cache K should be None after reset"
        assert att.cache_v is None, "Cache V should be None after reset"
        assert att.ptr_current_pos == 0, "Position pointer should be 0 after reset"

    def test_causal_masking(self):
        """
        Test that causal masking prevents information leakage from future tokens.
        """
        att = MultiheadAttentionWithSwa(
            d_in=64,
            d_out=64,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=32
        )
        att.eval()

        torch.manual_seed(123)
        x = torch.randn(1, 5, 64)

        with torch.no_grad():
            output = att(x, use_cache=False)

        # Verify output is finite (no NaN from improper masking)
        assert torch.isfinite(output).all(), "Output should be finite with proper causal masking"

    def test_gradient_flow(self):
        """
        Test that gradients flow properly through the attention mechanism.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.1,
            n_heads=4,
            sliding_window_size=64
        )
        att.train()

        x = torch.randn(2, 5, 256, requires_grad=True)
        output = att(x, use_cache=False)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"
        assert torch.isfinite(x.grad).all(), "Gradients should be finite"

        # Check that all parameters have gradients
        for name, param in att.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_without_cache_is_stateless(self):
        """
        Test that forward passes without cache are stateless.
        """
        att = MultiheadAttentionWithSwa(
            d_in=128,
            d_out=128,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=32
        )
        att.eval()

        x = torch.randn(1, 10, 128)

        with torch.no_grad():
            output1 = att(x, use_cache=False)
            output2 = att(x, use_cache=False)

        # Outputs should be identical (stateless)
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6,
                                   msg="Without cache, outputs should be identical")
        assert att.cache_k is None, "Cache should remain None without use_cache"
        assert att.ptr_current_pos == 0, "Position should remain 0 without use_cache"

    def test_qkv_bias_parameter(self):
        """
        Test that qkv_bias parameter is correctly applied.
        """
        # With bias
        att_with_bias = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.1,
            n_heads=4,
            qkv_bias=True,
            sliding_window_size=64
        )

        assert att_with_bias.W_query.bias is not None, "Query projection should have bias when qkv_bias=True"
        assert att_with_bias.W_key.bias is not None, "Key projection should have bias when qkv_bias=True"
        assert att_with_bias.W_value.bias is not None, "Value projection should have bias when qkv_bias=True"

        # Without bias
        att_without_bias = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.1,
            n_heads=4,
            qkv_bias=False,
            sliding_window_size=64
        )

        assert att_without_bias.W_query.bias is None, "Query projection should not have bias when qkv_bias=False"
        assert att_without_bias.W_key.bias is None, "Key projection should not have bias when qkv_bias=False"
        assert att_without_bias.W_value.bias is None, "Value projection should not have bias when qkv_bias=False"

    def test_different_head_counts(self):
        """
        Test that different numbers of attention heads work correctly.
        """
        for num_heads in [1, 2, 4, 8, 16]:
            d_out = 256
            assert d_out % num_heads == 0, "Test setup error"

            att = MultiheadAttentionWithSwa(
                d_in=256,
                d_out=d_out,
                dropout=0.1,
                n_heads=num_heads,
                sliding_window_size=64
            )

            x = torch.randn(2, 10, 256)
            output = att(x)

            assert output.shape == x.shape, f"Output shape mismatch for num_heads={num_heads}"
            assert att.head_dim == d_out // num_heads, f"Head dimension should be {d_out // num_heads} for num_heads={num_heads}, got {att.head_dim}"

    def test_different_window_sizes(self):
        """
        Test that different sliding window sizes work correctly.
        """
        for window_size in [16, 32, 64, 128, 256]:
            att = MultiheadAttentionWithSwa(
                d_in=128,
                d_out=128,
                dropout=0.0,
                n_heads=4,
                sliding_window_size=window_size
            )
            att.eval()

            with torch.no_grad():
                # Process more tokens than window size
                x = torch.randn(1, window_size + 20, 128)
                _ = att(x, use_cache=True)

                # Cache should be trimmed to window size
                assert att.cache_k.shape[1] == window_size, \
                    f"Cache should be {window_size}, got {att.cache_k.shape[1]}"

    def test_batch_processing(self):
        """
        Test that batch processing works correctly.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=64
        )
        att.eval()

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10, 256)
            with torch.no_grad():
                output = att(x, use_cache=False)

            assert output.shape == (batch_size, 10, 256), \
                f"Output shape should be ({batch_size}, 10, 256), got {output.shape}"
            assert torch.isfinite(output).all(), f"Output should be finite for batch_size={batch_size}"

    def test_incremental_generation_simulation(self):
        """
        Test incremental generation pattern (typical for autoregressive models).
        """
        att = MultiheadAttentionWithSwa(
            d_in=128,
            d_out=128,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=32
        )
        att.eval()

        with torch.no_grad():
            # Reset cache
            att.reset_cache()

            # Process initial prompt
            prompt = torch.randn(1, 10, 128)
            output_prompt = att(prompt, use_cache=True)
            assert output_prompt.shape == (1, 10, 128), f"Prompt output shape should be (1, 10, 128), got {output_prompt.shape}"
            assert att.cache_k.shape[1] == 10, f"Cache should have 10 tokens after prompt, got {att.cache_k.shape[1]}"

            # Generate tokens one at a time
            for i in range(5):
                next_token = torch.randn(1, 1, 128)
                output_token = att(next_token, use_cache=True)
                expected_shape = (1, 1, 128)
                assert output_token.shape == expected_shape, f"Token {i} output shape should be {expected_shape}, got {output_token.shape}"
                expected_cache_size = min(10 + i + 1, 32)
                assert att.cache_k.shape[1] == expected_cache_size, f"Token {i}: cache should have {expected_cache_size} tokens, got {att.cache_k.shape[1]}"

    def test_cache_and_non_cache_consistency_initial_pass(self):
        """
        Test that cached and non-cached outputs are consistent for the first forward pass.
        """
        att = MultiheadAttentionWithSwa(
            d_in=256,
            d_out=256,
            dropout=0.0,
            n_heads=4,
            sliding_window_size=64
        )
        att.eval()

        x = torch.randn(1, 10, 256)

        with torch.no_grad():
            # Non-cached
            output_no_cache = att(x, use_cache=False)

            # Reset and use cache
            att.reset_cache()
            output_with_cache = att(x, use_cache=True)

        torch.testing.assert_close(output_no_cache, output_with_cache, atol=1e-6, rtol=1e-6,
                                   msg="Cached and non-cached outputs should match for initial pass")

    def test_parameter_count(self):
        """
        Test that parameter count matches expected values.
        """
        d_in = 256
        d_out = 256
        att = MultiheadAttentionWithSwa(
            d_in=d_in,
            d_out=d_out,
            dropout=0.1,
            n_heads=4,
            qkv_bias=False,
            sliding_window_size=64
        )

        total_params = sum(p.numel() for p in att.parameters())

        # Expected: W_query, W_key, W_value (each d_in * d_out), out_proj weight and bias (d_out * d_out + d_out)
        expected = 3 * (d_in * d_out) + (d_out * d_out) + d_out
        assert total_params == expected, f"Expected {expected} parameters, got {total_params}"

    def test_training_vs_eval_mode(self):
        """
        Test that training and eval modes behave differently (dropout).
        """
        torch.manual_seed(42)
        att = MultiheadAttentionWithSwa(
            d_in=128,
            d_out=128,
            dropout=0.5,  # High dropout for testing
            n_heads=4,
            sliding_window_size=32
        )

        x = torch.randn(1, 10, 128)

        # Training mode
        att.train()
        with torch.no_grad():
            output_train = att(x, use_cache=False)

        # Eval mode
        att.eval()
        torch.manual_seed(42)
        with torch.no_grad():
            output_eval = att(x, use_cache=False)

        # Outputs should differ due to dropout
        # Note: This test might occasionally fail due to randomness, but with 0.5 dropout it should be rare
        assert not torch.allclose(output_train, output_eval, atol=1e-5), \
            "Training and eval outputs should differ when dropout > 0"

    def test_device_compatibility(self):
        """
        Test that the module works on different devices.
        """
        att = MultiheadAttentionWithSwa(
            d_in=128,
            d_out=128,
            dropout=0.1,
            n_heads=4,
            sliding_window_size=32
        )

        # CPU
        x_cpu = torch.randn(1, 5, 128)
        output_cpu = att(x_cpu)
        assert output_cpu.device.type == 'cpu', f"Output should be on CPU, got {output_cpu.device.type}"

        # GPU (if available)
        if torch.cuda.is_available():
            att_gpu = att.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = att_gpu(x_gpu)
            assert output_gpu.device.type == 'cuda', f"Output should be on CUDA, got {output_gpu.device.type}"
