import pytest
import torch

from src.attention import MultiHeadAttention, MultiHeadAttentionWrapper


class TestMultiHeadAttention:
    """
    Test suite for the efficient MultiHeadAttention module.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiHeadAttention.
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

    def test_gradient_flow_efficiency(self, sample_inputs):
        """
        Test that gradients flow properly through the efficient multi-head attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output = mha(batch)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_output_projection_effect(self, sample_inputs):
        """
        Test that output projection layer affects the results.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Create two identical models
        torch.manual_seed(123)
        mha1 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        torch.manual_seed(123)
        mha2 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Zero out one model's output projection
        with torch.no_grad():
            mha2.out_proj.weight.zero_()
            mha2.out_proj.bias.zero_()

        output1 = mha1(batch)
        output2 = mha2(batch)

        # Results should be different due to output projection
        assert not torch.allclose(output1, output2), "Output projection should affect results"

    def test_causal_mask_efficiency(self, sample_inputs):
        """
        Test that causal masking works properly in the efficient implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

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

    def test_different_head_counts_efficiency(self, sample_inputs):
        """
        Test multi-head attention with different numbers of heads for efficiency.
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
            mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_with_seed_efficiency(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in efficient implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

    def test_versus_wrapper_implementation(self, sample_inputs):
        """
        Test that efficient implementation produces different but valid results compared to wrapper.
        """
        d_in, d_out_per_head = 3, 2
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Wrapper implementation (d_out per head)
        torch.manual_seed(789)
        wrapper = MultiHeadAttentionWrapper(d_in, d_out_per_head, context_length, dropout, num_heads)
        wrapper_output = wrapper(batch)

        # Efficient implementation (total d_out)
        torch.manual_seed(789)
        efficient = MultiHeadAttention(d_in, d_out_per_head * num_heads, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert wrapper_output.shape == efficient_output.shape, f"Output shapes should match: wrapper {wrapper_output.shape} vs efficient {efficient_output.shape}"

        # Results will be different due to different architectures and output projection
        # but both should be valid attention outputs
        assert not torch.allclose(wrapper_output, efficient_output), "Different implementations should produce different results"

    def test_attention_weights_normalization(self, sample_inputs):
        """
        Test that attention weights are properly normalized (though not directly accessible).
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Test that the model produces reasonable outputs
        with torch.no_grad():
            output = mha(batch)
            assert torch.isfinite(output).all(), "All outputs should be finite"
            assert not torch.isnan(output).any(), "No outputs should be NaN"

    def test_wrong_input_dimensions_efficiency(self):
        """
        Test error handling for wrong input dimensions in efficient implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_efficient(self):
        """
        Test efficient MultiHeadAttention with realistic transformer configuration.

        This test uses parameters similar to those found in large language models:
        - embed_dim=768 (common embedding dimension)
        - 12 attention heads
        - context_length=1024 (typical context window)
        - batch_size=8

        The key difference from MultiHeadAttentionWrapper is that this efficient implementation:
        - Uses d_out=768 total (not per head)
        - Computes all heads simultaneously
        - Includes output projection layer
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the efficient multi-head attention with transformer-like configuration
        # Note: d_out=embed_dim (total), not per head like in MultiHeadAttentionWrapper
        mha_efficient = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,  # Total output dimension (768)
            context_length=context_len,
            dropout=0.0,
            n_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_efficient(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out (768 total)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_efficient.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_efficient.head_dim}"
        assert mha_efficient.d_out == 768, f"Expected d_out=768, got {mha_efficient.d_out}"
        assert mha_efficient.n_heads == 12, f"Expected n_heads=12, got {mha_efficient.n_heads}"
