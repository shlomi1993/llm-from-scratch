import pytest
import torch

from src.attention import MHAPyTorchScaledDotProduct, MultiHeadAttention


class TestMHAPyTorchScaledDotProduct:
    """
    Test suite for the MHAPyTorchScaledDotProduct module (PyTorch built-in scaled dot-product attention).
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MHAPyTorchScaledDotProduct.
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

    def test_output_shape_pytorch_scaled(self, sample_inputs):
        """
        Test that PyTorch scaled dot-product attention produces correct output shape.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.1
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion_pytorch_scaled(self):
        """
        Test that assertion is raised when d_out is not divisible by n_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out.*divisible.*n_heads"):
            MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

    def test_head_dimension_calculation_pytorch_scaled(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        assert mha.d_out == 12, f"Expected d_out=12, got {mha.d_out}"
        assert mha.n_heads == 4, f"Expected n_heads=4, got {mha.n_heads}"

    def test_pytorch_scaled_qkv_projection(self, sample_inputs):
        """
        Test that QKV projection works correctly in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test the QKV projection creates correct shape
        qkv_output = mha.qkv(batch)
        expected_qkv_shape = (2, 6, 3 * d_out)  # 3 * d_out for Q, K, V
        assert qkv_output.shape == expected_qkv_shape, f"Expected QKV shape {expected_qkv_shape}, got {qkv_output.shape}"

        # Test full forward pass
        output = mha(batch)
        assert output.shape == (2, 6, d_out), f"Expected output shape (2, 6, {d_out}), got {output.shape}"

    def test_gradient_flow_pytorch_scaled(self, sample_inputs):
        """
        Test that gradients flow properly through the PyTorch scaled attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output = mha(batch)

        # Compute loss and back-propagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_causal_mask_pytorch_scaled(self, sample_inputs):
        """
        Test that causal masking works properly in the PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

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

    def test_different_head_counts_pytorch_scaled(self, sample_inputs):
        """
        Test PyTorch scaled attention with different numbers of heads.
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
            mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_pytorch_scaled(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

    def test_versus_other_implementations(self, sample_inputs):
        """
        Test that PyTorch scaled implementation produces valid results compared to other implementations.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # PyTorch scaled implementation
        torch.manual_seed(789)
        pytorch_scaled = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        pytorch_output = pytorch_scaled(batch)

        # Efficient implementation for comparison (different seed to show they can differ)
        torch.manual_seed(456)
        efficient = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert pytorch_output.shape == efficient_output.shape, f"Output shapes should match: pytorch_scaled {pytorch_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(pytorch_output).all(), "PyTorch scaled output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(pytorch_output).any(), "PyTorch scaled output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

    def test_training_vs_eval_mode(self, sample_inputs):
        """
        Test that training and evaluation modes work correctly.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.2  # High dropout to see the effect
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Training mode
        mha.train()
        torch.manual_seed(123)
        output_train = mha(batch)

        # Evaluation mode
        mha.eval()
        torch.manual_seed(123)
        output_eval = mha(batch)

        # In eval mode, dropout should be disabled, so results might be different
        # but both should be finite and valid
        assert torch.isfinite(output_train).all(), "Training output should be finite"
        assert torch.isfinite(output_eval).all(), "Evaluation output should be finite"
        assert not torch.isnan(output_train).any(), "Training output should not have NaN"
        assert not torch.isnan(output_eval).any(), "Evaluation output should not have NaN"

    def test_wrong_input_dimensions_pytorch_scaled(self):
        """
        Test error handling for wrong input dimensions in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_pytorch_scaled(self):
        """
        Test PyTorch scaled attention with realistic transformer configuration.

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

        # Initialize the PyTorch scaled attention with transformer-like configuration
        mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
            d_in=embed_dim,
            d_out=embed_dim,
            n_heads=12,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_pytorch_scaled(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_pytorch_scaled.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_pytorch_scaled.head_dim}"
        assert mha_pytorch_scaled.d_out == 768, f"Expected d_out=768, got {mha_pytorch_scaled.d_out}"
        assert mha_pytorch_scaled.n_heads == 12, f"Expected n_heads=12, got {mha_pytorch_scaled.n_heads}"

    def test_parameter_efficiency_pytorch_scaled(self, sample_inputs):
        """
        Test that PyTorch scaled implementation has expected parameter counts.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # PyTorch scaled implementation
        pytorch_scaled = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Regular implementation for comparison
        regular = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Count parameters
        pytorch_params = sum(p.numel() for p in pytorch_scaled.parameters())
        regular_params = sum(p.numel() for p in regular.parameters())

        assert pytorch_params > 0, "PyTorch scaled implementation should have parameters"
        assert regular_params > 0, "Regular implementation should have parameters"

        # Both should have similar parameter counts (QKV + proj)
        # The difference should be minimal since both use similar architectures
        assert abs(pytorch_params - regular_params) < pytorch_params * 0.1, "Parameter counts should be similar between implementations"

    def test_bias_functionality_pytorch_scaled(self, sample_inputs):
        """
        Test that bias terms work correctly when enabled in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with bias
        mha_with_bias = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout, qkv_bias=True)
        output_with_bias = mha_with_bias(batch)

        # Test without bias
        mha_no_bias = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout, qkv_bias=False)
        output_no_bias = mha_no_bias(batch)

        # Both should produce valid outputs
        assert torch.isfinite(output_with_bias).all(), "Output with bias should be finite"
        assert torch.isfinite(output_no_bias).all(), "Output without bias should be finite"
        assert output_with_bias.shape == output_no_bias.shape, "Output shapes should match regardless of bias"

        # Check that bias parameters exist when requested
        assert mha_with_bias.qkv.bias is not None, "QKV bias should exist when qkv_bias=True"
        assert mha_no_bias.qkv.bias is None, "QKV bias should not exist when qkv_bias=False"

    def test_pytorch_scaled_attention_backend(self, sample_inputs):
        """
        Test that PyTorch scaled_dot_product_attention is being used correctly.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test that we can successfully run the forward pass
        # (implicitly testing that scaled_dot_product_attention is working)
        output = mha(batch)

        b, n, _ = batch.shape
        assert output.shape == (b, n, d_out), f"Expected output shape ({b}, {n}, {d_out}), got {output.shape}"

        # Test with different training modes
        mha.train()
        train_output = mha(batch)

        mha.eval()
        eval_output = mha(batch)

        # Both should produce valid outputs
        assert torch.isfinite(train_output).all(), "Training output should be finite"
        assert torch.isfinite(eval_output).all(), "Evaluation output should be finite"

    def test_pytorch_scaled_performance_characteristics(self, sample_inputs):
        """
        Test performance-related characteristics of PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with different batch sizes to ensure scalability
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            # Create batch of the specified size
            test_batch = torch.stack([sample_inputs] * batch_size, dim=0)

            mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
            output = mha(test_batch)

            expected_shape = (batch_size, 6, d_out)
            assert output.shape == expected_shape, f"For batch_size={batch_size}, expected shape {expected_shape}, got {output.shape}"

            # Verify output is finite and valid
            assert torch.isfinite(output).all(), f"Output should be finite for batch_size={batch_size}"
            assert not torch.isnan(output).any(), f"Output should not contain NaN for batch_size={batch_size}"
