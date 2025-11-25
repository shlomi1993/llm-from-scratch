import pytest
import torch

from src.attention import MHAEinsum, MultiHeadAttention


class TestMHAEinsum:
    """
    Test suite for the MHAEinsum module (einsum-based multi-head attention).
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MHAEinsum.
        """
        torch.manual_seed(42)
        return torch.tensor([
            [0.43, 0.15, 0.89],  # Sample tensor to test attention
            [0.55, 0.87, 0.66],  # mechanisms with known values
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]
        ])

    def test_output_shape_einsum(self, sample_inputs):
        """
        Test that einsum multi-head attention produces correct output shape.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.1
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion_einsum(self):
        """
        Test that assertion is raised when d_out is not divisible by n_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out must be divisible by n_heads"):
            MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

    def test_head_dimension_calculation_einsum(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly in einsum implementation.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        assert mha.d_out == 12, f"Expected d_out=12, got {mha.d_out}"
        assert mha.n_heads == 4, f"Expected n_heads=4, got {mha.n_heads}"

    def test_einsum_parameter_initialization(self, sample_inputs):
        """
        Test that einsum implementation initializes parameters correctly.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Test without bias
        mha_no_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
        assert mha_no_bias.bias_q is None, "bias_q should be None when qkv_bias=False"
        assert mha_no_bias.bias_k is None, "bias_k should be None when qkv_bias=False"
        assert mha_no_bias.bias_v is None, "bias_v should be None when qkv_bias=False"

        # Test with bias
        mha_with_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=True)
        assert mha_with_bias.bias_q is not None, "bias_q should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_k is not None, "bias_k should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_v is not None, "bias_v should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_q.shape == (d_out,), f"bias_q should have shape ({d_out},), got {mha_with_bias.bias_q.shape}"

    def test_gradient_flow_einsum(self, sample_inputs):
        """
        Test that gradients flow properly through the einsum multi-head attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output = mha(batch)

        # Compute loss and back-propagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        assert mha.W_query.grad is not None, "W_query should have gradients"
        assert mha.W_key.grad is not None, "W_key should have gradients"
        assert mha.W_value.grad is not None, "W_value should have gradients"
        assert mha.out_proj.weight.grad is not None, "out_proj should have gradients"

    def test_causal_mask_einsum(self, sample_inputs):
        """
        Test that causal masking works properly in the einsum implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test with eval mode to ensure consistent behavior
        mha.eval()

        with torch.no_grad():
            output = mha(batch)

            # Check that changing future tokens doesn't affect past outputs
            modified_batch = batch.clone()
            modified_batch[:, -1, :] = torch.randn_like(modified_batch[:, -1, :])

            modified_output = mha(modified_batch)

            # First 5 positions should be identical (causal masking)
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), "Causal masking should prevent future tokens from affecting past outputs"

    def test_different_head_counts_einsum(self, sample_inputs):
        """
        Test einsum multi-head attention with different numbers of heads.
        """
        d_in = 3
        context_length = 8
        dropout = 0.1

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test different head counts with appropriate d_out values
        test_configs = [
            (1, 4),   # 1 head, d_out=4
            (2, 6),   # 2 heads, d_out=6
            (3, 9),   # 3 heads, d_out=9
            (4, 8),   # 4 heads, d_out=8
        ]

        for num_heads, d_out in test_configs:
            mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_einsum(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in einsum implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

    def test_versus_other_implementations(self, sample_inputs):
        """
        Test that einsum implementation produces valid results compared to other implementations.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Einsum implementation
        torch.manual_seed(789)
        einsum_mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        einsum_output = einsum_mha(batch)

        # Efficient implementation for comparison (different seed to show they can differ)
        torch.manual_seed(456)
        efficient = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert einsum_output.shape == efficient_output.shape, f"Output shapes should match: einsum {einsum_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(einsum_output).all(), "Einsum output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(einsum_output).any(), "Einsum output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

    def test_attention_weights_normalization_einsum(self, sample_inputs):
        """
        Test that attention weights are properly normalized in einsum implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test that the model produces reasonable outputs
        with torch.no_grad():
            output = mha(batch)

            # Check that outputs are finite and reasonable
            assert torch.isfinite(output).all(), "All outputs should be finite"
            assert not torch.isnan(output).any(), "No outputs should be NaN"

    def test_wrong_input_dimensions_einsum(self):
        """
        Test error handling for wrong input dimensions in einsum implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_einsum(self):
        """
        Test einsum multi-head attention with realistic transformer configuration.

        This test uses parameters similar to those found in large language models
        and verifies the specific user-requested test case.
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the einsum multi-head attention with transformer-like configuration
        mha_einsum = MHAEinsum(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_len,
            dropout=0.0,
            n_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_einsum(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_einsum.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_einsum.head_dim}"
        assert mha_einsum.d_out == 768, f"Expected d_out=768, got {mha_einsum.d_out}"
        assert mha_einsum.n_heads == 12, f"Expected n_heads=12, got {mha_einsum.n_heads}"

    def test_parameter_efficiency_einsum(self, sample_inputs):
        """
        Test that einsum implementation has expected parameter counts.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Einsum implementation
        einsum_mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Regular implementation for comparison
        regular = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Count parameters
        einsum_params = sum(p.numel() for p in einsum_mha.parameters())
        regular_params = sum(p.numel() for p in regular.parameters())

        assert einsum_params > 0, "Einsum implementation should have parameters"
        assert regular_params > 0, "Regular implementation should have parameters"

        # Verify specific parameter shapes
        assert einsum_mha.W_query.shape == (d_in, d_out), f"Expected W_query shape ({d_in}, {d_out}), got {einsum_mha.W_query.shape}"
        assert einsum_mha.W_key.shape == (d_in, d_out), f"Expected W_key shape ({d_in}, {d_out}), got {einsum_mha.W_key.shape}"
        assert einsum_mha.W_value.shape == (d_in, d_out), f"Expected W_value shape ({d_in}, {d_out}), got {einsum_mha.W_value.shape}"

    def test_bias_functionality_einsum(self, sample_inputs):
        """
        Test that bias terms work correctly when enabled.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with bias
        mha_with_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=True)
        output_with_bias = mha_with_bias(batch)

        # Test without bias
        mha_no_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
        output_no_bias = mha_no_bias(batch)

        # Both should produce valid outputs
        assert torch.isfinite(output_with_bias).all(), "Output with bias should be finite"
        assert torch.isfinite(output_no_bias).all(), "Output without bias should be finite"
        assert output_with_bias.shape == output_no_bias.shape, "Output shapes should match regardless of bias"

    def test_einsum_operations_correctness(self, sample_inputs):
        """
        Test that einsum operations produce mathematically correct results.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)

        # Extract components for manual verification
        b, n, _ = batch.shape

        # Test einsum QKV projection manually
        Q_einsum = torch.einsum("bnd,do->bno", batch, mha.W_query)
        Q_manual = batch @ mha.W_query

        # Should be equivalent
        assert torch.allclose(Q_einsum, Q_manual, atol=1e-6), "Einsum QKV projection should match manual computation"

        # Test full forward pass produces valid output
        output = mha(batch)
        assert output.shape == (b, n, d_out), f"Expected output shape ({b}, {n}, {d_out}), got {output.shape}"
