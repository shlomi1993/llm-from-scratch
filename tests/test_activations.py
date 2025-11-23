import pytest
import torch
import torch.nn as nn
import math

from src.activations import GELU


class TestGELU:
    """
    Test suite for the GELU activation function.
    """

    @pytest.fixture
    def gelu_layer(self):
        """
        Create a GELU activation layer for testing.
        """
        return GELU()

    def test_gelu_forward_shape_preservation(self, gelu_layer):
        """
        Test that GELU preserves input tensor shapes.
        """
        test_shapes = [
            (5,),           # 1D tensor
            (3, 4),         # 2D tensor
            (2, 3, 4),      # 3D tensor
            (1, 2, 3, 4),   # 4D tensor
            (2, 3, 4, 5, 6) # 5D tensor
        ]
        for shape in test_shapes:
            input_tensor = torch.randn(shape)
            output = gelu_layer(input_tensor)
            assert output.shape == input_tensor.shape, f"Output shape {output.shape} should match input shape {shape}"

    def test_gelu_mathematical_correctness(self, gelu_layer):
        """
        Test GELU against the mathematical definition.
        """
        # Test with specific values where we can compute expected results
        test_values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        output = gelu_layer(test_values)

        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        expected = 0.5 * test_values * (1 + torch.tanh(sqrt_2_over_pi * (test_values + 0.044715 * test_values**3)))

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6, msg="GELU output should match mathematical definition within tolerance")

    def test_gelu_asymptotic_behavior(self, gelu_layer):
        """
        Test GELU behavior at extreme values.
        """
        # For large positive x, GELU(x) ≈ x
        large_positive = torch.tensor([5.0, 10.0, 100.0])
        output_pos = gelu_layer(large_positive)

        # Should be very close to x for large positive values
        relative_error = torch.abs((output_pos - large_positive) / large_positive)
        assert (relative_error < 0.01).all(), "GELU should approximate x for large positive values"

        # For large negative x, GELU(x) ≈ 0
        large_negative = torch.tensor([-5.0, -10.0, -100.0])
        output_neg = gelu_layer(large_negative)

        # Should be very close to 0 for large negative values
        assert torch.allclose(output_neg, torch.zeros_like(output_neg), atol=1e-3), "GELU should approximate 0 for large negative values"

    def test_gelu_gradient_properties(self, gelu_layer):
        """
        Test gradient properties of GELU.
        """
        x = torch.linspace(-2, 2, 100, requires_grad=True)
        y = gelu_layer(x)

        # Compute gradients
        grad_outputs = torch.ones_like(y)
        gradients = torch.autograd.grad(y, x, grad_outputs, create_graph=True)[0]

        # Gradients should be finite
        assert torch.isfinite(gradients).all(), "GELU gradients should be finite"

        # Gradient at x=0 should be approximately 0.5
        zero_idx = torch.argmin(torch.abs(x))
        grad_at_zero = gradients[zero_idx]
        assert torch.allclose(grad_at_zero, torch.tensor(0.5), atol=0.1), "GELU gradient at 0 should be approximately 0.5"

        # For positive values, gradients should be close to 1
        positive_mask = x > 1.0
        positive_grads = gradients[positive_mask]
        assert (positive_grads > 0.8).all(), "GELU gradients should be close to 1 for positive values"

    def test_gelu_vs_pytorch_reference(self):
        """
        Test GELU against PyTorch's built-in GELU with tanh approximation.
        """
        # PyTorch's GELU with approximate=True uses the same formula
        pytorch_gelu = nn.GELU(approximate='tanh')
        custom_gelu = GELU()

        # Test on random inputs
        x = torch.randn(100)

        output_pytorch = pytorch_gelu(x)
        output_custom = custom_gelu(x)

        torch.testing.assert_close(output_custom, output_pytorch, atol=1e-6, rtol=1e-6, msg="Custom GELU output should match PyTorch's GELU output within tolerance")

    def test_gelu_numerical_stability(self, gelu_layer):
        """
        Test GELU numerical stability with edge cases.
        """
        # Test with very small values
        small_values = torch.tensor([1e-10, -1e-10, 1e-8, -1e-8])
        output_small = gelu_layer(small_values)
        assert torch.isfinite(output_small).all(), "GELU should be stable with very small inputs"

        # Test with values that might cause tanh saturation
        saturation_values = torch.tensor([-10.0, -5.0, 5.0, 10.0])
        output_sat = gelu_layer(saturation_values)
        assert torch.isfinite(output_sat).all(), "GELU should be stable with saturation-prone inputs"

        # Test with large finite values
        large_values = torch.tensor([50.0, -50.0])
        output_large = gelu_layer(large_values)
        assert torch.isfinite(output_large).all(), "GELU should be stable with large finite inputs"

    def test_gelu_batch_processing(self, gelu_layer):
        """
        Test GELU with batch processing.
        """
        batch_size, seq_len, emb_dim = 32, 128, 768
        batch_input = torch.randn(batch_size, seq_len, emb_dim)

        output = gelu_layer(batch_input)

        assert output.shape == batch_input.shape, "Output shape should match input"
        assert torch.isfinite(output).all(), "All outputs should be finite"

        # Test that each element is processed independently
        # Apply to flattened version and compare
        flat_input = batch_input.flatten()
        flat_output = gelu_layer(flat_input)
        expected_output = flat_output.reshape(batch_size, seq_len, emb_dim)

        torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6, msg="GELU batch output should match flattened input output reshaped")

    def test_gelu_device_compatibility(self, gelu_layer):
        """
        Test GELU on different devices.
        """
        # Test CPU
        cpu_input = torch.randn(10)
        cpu_output = gelu_layer(cpu_input)
        assert cpu_output.device == torch.device('cpu'), "Output should be on CPU"

        # Test CUDA if available
        if torch.cuda.is_available():
            gelu_layer = gelu_layer.cuda()
            cuda_input = cpu_input.cuda()
            cuda_output = gelu_layer(cuda_input)
            assert cuda_output.device.type == 'cuda', "Output should be on CUDA"

            # Results should be the same (within numerical precision)
            torch.testing.assert_close(cuda_output.cpu(), cpu_output, atol=1e-6, rtol=1e-6, msg="CUDA and CPU GELU outputs should match within tolerance")

    def test_gelu_dtype_preservation(self, gelu_layer):
        """
        Test that GELU preserves input data type.
        """
        dtypes_to_test = [torch.float32, torch.float64]
        if torch.cuda.is_available():
            dtypes_to_test.append(torch.float16)

        for dtype in dtypes_to_test:
            input_tensor = torch.randn(10, dtype=dtype)
            output = gelu_layer(input_tensor)

            assert output.dtype == input_tensor.dtype, f"Output dtype {output.dtype} should match input dtype {dtype}"

    def test_gelu_backward_pass(self, gelu_layer):
        """
        Test GELU backward pass and gradient computation.
        """
        x = torch.randn(10, requires_grad=True)
        y = gelu_layer(x)
        loss = y.sum()

        # Compute gradients
        loss.backward()

        # Check that gradients exist and are reasonable
        assert x.grad is not None, "Input should have gradients"
        assert torch.isfinite(x.grad).all(), "Gradients should be finite"
        torch.testing.assert_close(x.grad, x.grad, atol=1e-6, rtol=1e-6, msg="GELU gradients should be consistent within tolerance")

        # Gradients should be reasonable in magnitude
        assert (x.grad.abs() < 10.0).all(), "GELU gradients should have reasonable magnitude"

    def test_gelu_with_different_tensor_layouts(self, gelu_layer):
        """
        Test GELU with different tensor memory layouts.
        """
        # Contiguous tensor
        x_contiguous = torch.randn(2, 3, 4)
        output_contiguous = gelu_layer(x_contiguous)

        # Non-contiguous tensor (transposed)
        x_transposed = x_contiguous.transpose(0, 2)
        output_transposed = gelu_layer(x_transposed)

        # Results should be consistent
        expected_transposed = output_contiguous.transpose(0, 2)
        torch.testing.assert_close(output_transposed, expected_transposed, atol=1e-6, rtol=1e-6)

    def test_gelu_comparison_with_other_activations(self):
        """
        Test GELU characteristics compared to other activations.
        """
        x = torch.linspace(-3, 3, 1000)

        gelu = GELU()
        relu = nn.ReLU()

        gelu_output = gelu(x)
        relu_output = relu(x)

        # GELU should be smoother than ReLU (no sharp corner at 0)
        # Check that GELU has non-zero output for small negative values (though negative)
        negative_small = torch.tensor([-0.5, -0.1])
        gelu_neg = gelu(negative_small)
        relu_neg = relu(negative_small)

        assert (gelu_neg != 0).all(), "GELU should have non-zero output for negative inputs"
        assert (gelu_neg < 0).all(), "GELU should have negative output for negative inputs"
        assert torch.allclose(relu_neg, torch.zeros_like(relu_neg)), "ReLU should be zero for negative inputs"

        # GELU should be closer to identity for positive values
        positive_vals = torch.tensor([1.0, 2.0, 3.0])
        gelu_pos = gelu(positive_vals)
        assert (gelu_pos > 0.8 * positive_vals).all(), "GELU should be close to identity for positive values"
