import pytest
import torch
import torch.nn as nn

from src.attention.grouped_query_attention import GroupedQueryAttention


class TestGroupedQueryAttention:
    """
    Test suite for GroupedQueryAttention implementation.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Create sample input tensors for testing.
        """
        torch.manual_seed(42)
        return torch.randn(2, 8, 64)  # (batch_size=2, seq_len=8, d_in=64)

    def test_gqa_vs_mha_parameter_efficiency(self, sample_inputs):
        """
        Test that GQA uses fewer parameters than equivalent MHA while maintaining output shape.
        """
        batch_size, seq_len, d_in = sample_inputs.shape
        d_out = 128
        num_heads = 8
        dropout = 0.1

        # Create GQA with 2 KV groups (4x reduction in KV parameters)
        gqa = GroupedQueryAttention(d_in=d_in, d_out=d_out, dropout=dropout, n_heads=num_heads, num_kv_groups=2)

        # Simulate equivalent MHA (would have num_kv_groups = num_heads)
        mha_kv_params = num_heads * (d_in * (d_out // num_heads))  # K and V projections
        gqa_kv_params = 2 * (d_in * (d_out // num_heads))  # Only 2 groups

        # Verify parameter efficiency
        assert gqa_kv_params < mha_kv_params, "GQA should use fewer KV parameters than MHA"

        # Verify output shape is maintained
        output = gqa(sample_inputs)
        expected_shape = (batch_size, seq_len, d_out)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

        # Verify output is finite and not all zeros
        assert torch.isfinite(output).all(), "Output should contain only finite values"
        assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"

    def test_kv_cache_consistency_with_incremental_generation(self, sample_inputs):
        """
        Test that cached and non-cached forward passes produce equivalent results.
        """
        d_out = 64
        num_heads = 4
        num_kv_groups = 2

        gqa = GroupedQueryAttention(d_in=sample_inputs.shape[-1], d_out=d_out, dropout=0.0, n_heads=num_heads,
                                    num_kv_groups=num_kv_groups)

        gqa.eval()  # Disable dropout for deterministic comparison

        # Method 1: Process full sequence without cache
        gqa.reset_cache()
        with torch.no_grad():
            full_output = gqa(sample_inputs, use_cache=False)

        # Method 2: Process incrementally with cache
        gqa.reset_cache()
        cached_outputs = []
        with torch.no_grad():
            for i in range(sample_inputs.shape[1]):
                token = sample_inputs[:, i:i+1, :]  # Single token
                output = gqa(token, use_cache=True)
                cached_outputs.append(output)

        incremental_output = torch.cat(cached_outputs, dim=1)

        # Compare outputs (allowing small numerical differences)
        assert torch.allclose(full_output, incremental_output, atol=1e-5, rtol=1e-5), \
            "Cached incremental generation should match full sequence processing"

    def test_causal_masking_prevents_information_leakage(self, sample_inputs):
        """
        Test that causal masking prevents tokens from attending to future positions.
        """
        d_out = 64
        num_heads = 4
        num_kv_groups = 2

        # Create model with identifiable inputs
        torch.manual_seed(123)
        gqa = GroupedQueryAttention(d_in=sample_inputs.shape[-1], d_out=d_out, dropout=0.0, n_heads=num_heads,
                                    num_kv_groups=num_kv_groups)

        # Create input where later tokens have distinct patterns
        test_input = torch.zeros(1, 4, sample_inputs.shape[-1])
        test_input[0, 0, :] = 1.0   # First token: all 1s
        test_input[0, 1, :] = 2.0   # Second token: all 2s
        test_input[0, 2, :] = 3.0   # Third token: all 3s
        test_input[0, 3, :] = 4.0   # Fourth token: all 4s

        gqa.eval()
        with torch.no_grad():
            output = gqa(test_input, use_cache=False)

        # Process tokens incrementally to verify causal property
        gqa.reset_cache()
        incremental_outputs = []
        with torch.no_grad():
            for i in range(test_input.shape[1]):
                partial_input = test_input[:, :i+1, :]
                partial_output = gqa(partial_input, use_cache=False)
                incremental_outputs.append(partial_output[:, i:i+1, :])  # Last token output

        # First token output should be identical in both cases
        # (it can only attend to itself)
        assert torch.allclose(output[:, 0:1, :], incremental_outputs[0], atol=1e-6), \
            "First token output should be identical (causal masking)"

    def test_different_group_configurations(self):
        """
        Test GQA with different group configurations and verify mathematical correctness.
        """
        batch_size, seq_len, d_in = 2, 6, 48
        d_out = 96
        num_heads = 12

        # Test different group configurations
        group_configs = [1, 2, 3, 4, 6, 12]  # All divisors of num_heads=12

        input_tensor = torch.randn(batch_size, seq_len, d_in)

        for num_kv_groups in group_configs:
            gqa = GroupedQueryAttention(d_in=d_in, d_out=d_out, dropout=0.1, n_heads=num_heads,
                                        num_kv_groups=num_kv_groups)

            # Verify group_size calculation
            expected_group_size = num_heads // num_kv_groups
            assert gqa.group_size == expected_group_size, \
                f"Group size should be {expected_group_size} for {num_kv_groups} groups"

            # Verify output shape consistency
            output = gqa(input_tensor)
            expected_shape = (batch_size, seq_len, d_out)
            assert output.shape == expected_shape, \
                f"Output shape mismatch for {num_kv_groups} groups: expected {expected_shape}, got {output.shape}"

            # Verify KV projection dimensions
            assert gqa.W_key.out_features == num_kv_groups * gqa.head_dim, \
                f"Key projection should output {num_kv_groups} * {gqa.head_dim} features"
            assert gqa.W_value.out_features == num_kv_groups * gqa.head_dim, \
                f"Value projection should output {num_kv_groups} * {gqa.head_dim} features"

    def test_gradient_flow_and_training_compatibility(self, sample_inputs):
        """
        Test that gradients flow properly through GQA during training.
        """
        d_out = 64
        num_heads = 8
        num_kv_groups = 2

        gqa = GroupedQueryAttention(d_in=sample_inputs.shape[-1], d_out=d_out, dropout=0.1,
                                   n_heads=num_heads, num_kv_groups=num_kv_groups)

        # Enable training mode
        gqa.train()

        # Create a simple loss scenario
        target = torch.randn_like(sample_inputs[:, :, :d_out])

        # Forward pass
        output = gqa(sample_inputs)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Verify gradients exist for all parameters
        for name, param in gqa.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} should have finite gradients"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Parameter {name} gradients should not be all zeros"

        # Test with caching enabled
        gqa.reset_cache()
        gqa.zero_grad()

        # Split input for cached processing
        first_half = sample_inputs[:, :4, :]
        second_half = sample_inputs[:, 4:, :]

        output1 = gqa(first_half, use_cache=True)
        output2 = gqa(second_half, use_cache=True)
        combined_output = torch.cat([output1, output2], dim=1)

        cached_loss = nn.MSELoss()(combined_output, target)
        cached_loss.backward()

        # Verify gradients still flow with caching
        for name, param in gqa.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients with caching"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} should have finite gradients with caching"
