import pytest
import torch
import torch.nn as nn

from src.normalization import LayerNorm
from src.configurations import GptConfig


class TestLayerNorm:
    """
    Test suite for the LayerNorm implementation.
    """

    @pytest.fixture
    def sample_config(self):
        """
        Create a small test configuration for faster testing.
        """
        return GptConfig(
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=32,
            drop_rate=0.1,
            qkv_bias=False
        )

    @pytest.fixture
    def layer_norm(self, sample_config):
        """
        Create a LayerNorm layer for testing.
        """
        return LayerNorm(sample_config.emb_dim)

    def test_layer_norm_mathematical_correctness(self, layer_norm, sample_config):
        """
        Test LayerNorm against mathematical definition.
        """
        # Test with known input where we can verify the computation
        input_tensor = torch.randn(2, 3, sample_config.emb_dim)
        output = layer_norm(input_tensor)

        # Manually compute expected output
        # LayerNorm normalizes over the last dimension
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        expected = (input_tensor - mean) / torch.sqrt(var + layer_norm.eps)
        expected = expected * layer_norm.scale + layer_norm.shift

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6, msg="LayerNorm output should match manual computation within tolerance")

    def test_layer_norm_mean_and_variance(self, layer_norm, sample_config):
        """
        Test that output has correct mean and variance.
        """
        input_tensor = torch.randn(4, 10, sample_config.emb_dim)
        output = layer_norm(input_tensor)

        # Check mean (should be approximately zero for each vector)
        output_mean = output.mean(dim=-1)
        expected_mean = layer_norm.shift.mean()  # Average of shift parameters

        # Mean should be close to the mean of shift parameters
        # (exactly equal when shift is all zeros in initialization)
        assert torch.allclose(output_mean, expected_mean * torch.ones_like(output_mean), atol=1e-5), "Mean should be approximately shift mean"

        # Check variance (should be approximately 1 for each vector when scale is ones)
        output_var = output.var(dim=-1, unbiased=False)
        expected_var = layer_norm.scale.var(unbiased=False) + 1.0  # Approximately 1 when scale is ones

        # For initialized parameters (scale=1, shift=0), variance should be close to 1
        if torch.allclose(layer_norm.scale, torch.ones_like(layer_norm.scale)):
            assert torch.allclose(output_var, torch.ones_like(output_var), atol=1e-4), "Variance should be approximately 1"

    def test_layer_norm_zero_variance_handling(self, layer_norm, sample_config):
        """
        Test LayerNorm behavior with zero variance input (constant values).
        """
        # Input with same values across the normalized dimension
        constant_value = 5.0
        input_tensor = torch.full((2, 3, sample_config.emb_dim), constant_value)

        output = layer_norm(input_tensor)

        # With constant input, mean should equal the input value
        # After normalization: (x - mean) / sqrt(0 + eps) = 0 / sqrt(eps) = 0
        # Then: 0 * scale + shift = shift
        expected_output = layer_norm.shift.unsqueeze(0).unsqueeze(0).expand_as(input_tensor)

        torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6, msg="LayerNorm output should equal shift parameter for constant input")

    def test_layer_norm_vs_pytorch_reference(self, sample_config):
        """
        Test LayerNorm against PyTorch's built-in LayerNorm.
        """
        emb_dim = sample_config.emb_dim

        # Create both implementations
        custom_ln = LayerNorm(emb_dim)
        pytorch_ln = nn.LayerNorm(emb_dim, eps=1e-5)

        # Initialize with same parameters
        with torch.no_grad():
            pytorch_ln.weight.copy_(custom_ln.scale)
            pytorch_ln.bias.copy_(custom_ln.shift)

        # Test with same input
        input_tensor = torch.randn(3, 7, emb_dim)

        custom_output = custom_ln(input_tensor)
        pytorch_output = pytorch_ln(input_tensor)

        torch.testing.assert_close(custom_output, pytorch_output, atol=1e-6, rtol=1e-6, msg="LayerNorm output should match PyTorch's LayerNorm output within tolerance")

    def test_layer_norm_different_epsilon_values(self, sample_config):
        """
        Test LayerNorm with different epsilon values.
        """
        emb_dim = sample_config.emb_dim
        eps_values = [1e-8, 1e-5, 1e-3]

        # Create input that might be sensitive to epsilon
        input_tensor = torch.randn(2, 5, emb_dim) * 0.1  # Small values

        outputs = []
        for eps in eps_values:
            ln = LayerNorm(emb_dim)
            ln.eps = eps
            output = ln(input_tensor)
            outputs.append(output)

            # All should be finite
            assert torch.isfinite(output).all(), f"Output should be finite for eps={eps}"

        # Outputs should be similar but not identical
        for i in range(len(outputs) - 1):
            # Should be close but might have small differences due to epsilon
            # We mainly test that all outputs are finite and reasonable
            assert torch.allclose(outputs[i], outputs[i+1], atol=0.1, rtol=0.1), f"Outputs with different eps should be reasonably close"

    def test_layer_norm_parameter_learning(self, layer_norm, sample_config):
        """
        Test that LayerNorm parameters can be learned.
        """
        input_tensor = torch.randn(4, 8, sample_config.emb_dim)

        # Create a realistic target: apply LayerNorm with specific scale/shift values
        target_scale = torch.ones_like(layer_norm.scale) * 2.0  # Scale of 2
        target_shift = torch.ones_like(layer_norm.shift) * 0.5  # Shift of 0.5

        # Create target output by manually applying LayerNorm with target parameters
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (input_tensor - mean) / torch.sqrt(var + layer_norm.eps)
        target = target_scale * norm_x + target_shift

        # Optimize parameters
        optimizer = torch.optim.Adam([layer_norm.scale, layer_norm.shift], lr=0.01)

        initial_loss = None
        for i in range(200):  # More iterations for convergence
            optimizer.zero_grad()
            output = layer_norm(input_tensor)
            loss = nn.MSELoss()(output, target)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Loss should decrease during training"
        assert final_loss < 0.1, "Should achieve reasonable fit"

    def test_layer_norm_numerical_stability(self, layer_norm, sample_config):
        """
        Test LayerNorm numerical stability with extreme inputs.
        """
        # Test with very large values
        large_input = torch.randn(2, 3, sample_config.emb_dim) * 1000
        large_output = layer_norm(large_input)
        assert torch.isfinite(large_output).all(), "Should handle large inputs"

        # Test with very small values
        small_input = torch.randn(2, 3, sample_config.emb_dim) * 1e-6
        small_output = layer_norm(small_input)
        assert torch.isfinite(small_output).all(), "Should handle small inputs"

        # Test with mixed extreme values
        mixed_input = torch.randn(2, 3, sample_config.emb_dim)
        mixed_input[:, :, :sample_config.emb_dim//2] *= 1000
        mixed_input[:, :, sample_config.emb_dim//2:] *= 1e-6
        mixed_output = layer_norm(mixed_input)
        assert torch.isfinite(mixed_output).all(), "Should handle mixed extreme inputs"

    def test_layer_norm_with_different_dimensions(self):
        """
        Test LayerNorm with different embedding dimensions.
        """
        dims_to_test = [64, 128, 256, 512, 768, 1024]

        for dim in dims_to_test:
            ln = LayerNorm(dim)
            input_tensor = torch.randn(2, 5, dim)
            output = ln(input_tensor)

            assert output.shape == input_tensor.shape, f"Shape should be preserved for dim={dim}"
            assert torch.isfinite(output).all(), f"Output should be finite for dim={dim}"

            # Check that normalization is working
            output_mean = output.mean(dim=-1)
            output_var = output.var(dim=-1, unbiased=False)

            # Should have approximately zero mean and unit variance (with initialization)
            assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-4), f"Mean should be ~0 for dim={dim}"
            assert torch.allclose(output_var, torch.ones_like(output_var), atol=1e-3), f"Variance should be ~1 for dim={dim}"
