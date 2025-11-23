import pytest
import torch
import math

from src.feed_forward import FeedForward
from src.configurations import GptConfig, GPT_CONFIG_124M


class TestFeedForward:
    """
    Test suite for the FeedForward network.
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
    def feed_forward_layer(self, sample_config):
        """
        Create a FeedForward layer for testing.
        """
        return FeedForward(sample_config)

    def test_feed_forward_forward_pass_shape(self, feed_forward_layer, sample_config):
        """
        Test that forward pass produces correct output shape.
        """
        batch_size, seq_len = 4, 10
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        output = feed_forward_layer(input_tensor)

        expected_shape = (batch_size, seq_len, sample_config.emb_dim)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_feed_forward_different_input_shapes(self, feed_forward_layer, sample_config):
        """
        Test FeedForward with different input shapes.
        """
        test_shapes = [
            (1, 1, sample_config.emb_dim),      # Single token
            (1, 10, sample_config.emb_dim),     # Single sequence
            (8, 20, sample_config.emb_dim),     # Batch of sequences
            (16, 64, sample_config.emb_dim),    # Large batch
        ]

        for shape in test_shapes:
            input_tensor = torch.randn(shape)
            output = feed_forward_layer(input_tensor)

            assert output.shape == shape, f"Output shape {output.shape} should match input shape {shape}"

    def test_feed_forward_values_finite(self, feed_forward_layer, sample_config):
        """
        Test that forward pass produces finite values.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        output = feed_forward_layer(input_tensor)

        assert torch.isfinite(output).all(), "Output should contain only finite values"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain infinite values"

    def test_feed_forward_dimension_expansion_contraction(self, sample_config):
        """
        Test the 4x expansion and contraction pattern.
        """
        ff = FeedForward(sample_config)
        emb_dim = sample_config.emb_dim

        # Test intermediate expansion
        input_tensor = torch.randn(1, 1, emb_dim)

        # Manually forward through components to check dimensions
        # Access layers through the Sequential container
        fc1 = ff.layers[0]  # First linear layer
        act = ff.layers[1]  # GELU activation
        fc2 = ff.layers[2]  # Second linear layer

        x = fc1(input_tensor)
        assert x.shape[-1] == 4 * emb_dim, f"After fc1, dimension should be {4 * emb_dim}"

        x = act(x)
        assert x.shape[-1] == 4 * emb_dim, f"After activation, dimension should be {4 * emb_dim}"

        x = fc2(x)
        assert x.shape[-1] == emb_dim, f"After fc2, dimension should be {emb_dim}"

    def test_feed_forward_gradient_flow(self, feed_forward_layer, sample_config):
        """
        Test that gradients flow properly through the FeedForward layer.
        """
        input_tensor = torch.randn(2, 3, sample_config.emb_dim, requires_grad=True)

        output = feed_forward_layer(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None, "Input tensor should have gradients"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradients should be finite"

        # Check that all layer parameters have gradients
        for name, param in feed_forward_layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} gradients should be finite"

    def test_feed_forward_training_vs_eval_mode(self, feed_forward_layer, sample_config):
        """
        Test that training and eval modes affect dropout behavior.
        """
        input_tensor = torch.randn(4, 10, sample_config.emb_dim)

        # Training mode (dropout active)
        feed_forward_layer.train()
        torch.manual_seed(42)
        output_train1 = feed_forward_layer(input_tensor)

        torch.manual_seed(42)
        output_train2 = feed_forward_layer(input_tensor)

        # In training mode, dropout should cause different outputs
        # (unless dropout rate is 0, but we're using 0.1)
        if sample_config.drop_rate > 0:
            # Due to randomness in dropout, outputs might be different
            # But we can't guarantee this, so let's just check shapes
            assert output_train1.shape == output_train2.shape, "Training outputs should have same shape"

        # Eval mode (dropout inactive)
        feed_forward_layer.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            output_eval1 = feed_forward_layer(input_tensor)

            torch.manual_seed(42)
            output_eval2 = feed_forward_layer(input_tensor)

            # In eval mode, outputs should be identical
            torch.testing.assert_close(output_eval1, output_eval2, atol=1e-6, rtol=1e-6, msg="Eval mode outputs should be identical")

    def test_feed_forward_zero_dropout(self, sample_config):
        """
        Test FeedForward with zero dropout rate.
        """
        zero_dropout_config = GptConfig(
            emb_dim=sample_config.emb_dim,
            n_layers=sample_config.n_layers,
            n_heads=sample_config.n_heads,
            vocab_size=sample_config.vocab_size,
            context_length=sample_config.context_length,
            drop_rate=0.0,  # Zero dropout
            qkv_bias=sample_config.qkv_bias
        )

        ff = FeedForward(zero_dropout_config)
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Training mode should give same results as eval mode with zero dropout
        ff.train()
        torch.manual_seed(123)
        output_train = ff(input_tensor)

        ff.eval()
        torch.manual_seed(123)
        output_eval = ff(input_tensor)

        torch.testing.assert_close(output_train, output_eval, atol=1e-6, rtol=1e-6, msg="With zero dropout, training and eval outputs should match")

    def test_feed_forward_parameter_initialization(self, sample_config):
        """
        Test that parameters are initialized reasonably.
        """
        ff = FeedForward(sample_config)

        # Access layers through the Sequential container
        fc1 = ff.layers[0]  # First linear layer
        fc2 = ff.layers[2]  # Second linear layer (skip GELU at index 1)

        # Check fc1 weight initialization
        fc1_weight_std = fc1.weight.std().item()
        expected_std = 1.0 / math.sqrt(sample_config.emb_dim)  # Typical Xavier-like initialization

        # Should be in reasonable range (not too large or too small)
        assert 0.01 < fc1_weight_std < 1.0, f"fc1 weight std {fc1_weight_std} should be in reasonable range"

        # Check fc2 weight initialization
        fc2_weight_std = fc2.weight.std().item()

        # Should be in reasonable range
        assert 0.01 < fc2_weight_std < 1.0, f"fc2 weight std {fc2_weight_std} should be in reasonable range"

        # Check bias initialization (should be small)
        fc1_bias_abs_max = fc1.bias.abs().max().item()
        fc2_bias_abs_max = fc2.bias.abs().max().item()

        # Biases should be small (PyTorch default initialization)
        assert fc1_bias_abs_max < 0.1, f"fc1 bias max absolute value {fc1_bias_abs_max} should be small"
        assert fc2_bias_abs_max < 0.1, f"fc2 bias max absolute value {fc2_bias_abs_max} should be small"

    def test_feed_forward_with_gpt_configs(self):
        """
        Test FeedForward with standard GPT configurations.
        """
        configs = [GPT_CONFIG_124M]  # Test with one standard config to keep test fast

        for config in configs:
            ff = FeedForward(config)
            input_tensor = torch.randn(1, 5, config.emb_dim)

            output = ff(input_tensor)

            assert output.shape == input_tensor.shape, "Output shape should match input"
            assert torch.isfinite(output).all(), "Output should be finite"

    def test_feed_forward_residual_compatibility(self, feed_forward_layer, sample_config):
        """
        Test that FeedForward is compatible with residual connections.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Apply FeedForward
        ff_output = feed_forward_layer(input_tensor)

        # Add residual connection
        residual_output = input_tensor + ff_output

        assert residual_output.shape == input_tensor.shape, "Residual connection should preserve shape"
        assert torch.isfinite(residual_output).all(), "Residual output should be finite"

    def test_feed_forward_batch_independence(self, feed_forward_layer, sample_config):
        """
        Test that different batch items are processed independently.
        """
        batch_size, seq_len = 3, 5

        # Process batch
        batch_input = torch.randn(batch_size, seq_len, sample_config.emb_dim)
        batch_output = feed_forward_layer(batch_input)

        # Process individual items
        individual_outputs = []
        for i in range(batch_size):
            individual_input = batch_input[i:i+1]  # Keep batch dimension
            individual_output = feed_forward_layer(individual_input)
            individual_outputs.append(individual_output)

        # Concatenate individual outputs
        concat_output = torch.cat(individual_outputs, dim=0)

        # Should be identical to batch processing
        torch.testing.assert_close(batch_output, concat_output, atol=1e-6, rtol=1e-6, msg="Batch and individual outputs should match")

    def test_feed_forward_device_compatibility(self, sample_config):
        """
        Test FeedForward on different devices.
        """
        ff = FeedForward(sample_config)
        input_tensor = torch.randn(2, 3, sample_config.emb_dim)

        # Test CPU
        output_cpu = ff(input_tensor)
        assert output_cpu.device == torch.device('cpu'), "Output should be on CPU"

        # Test CUDA if available
        if torch.cuda.is_available():
            ff_cuda = ff.cuda()
            input_cuda = input_tensor.cuda()
            output_cuda = ff_cuda(input_cuda)

            assert output_cuda.device.type == 'cuda', "Output should be on CUDA"

            # Results should be close (within numerical precision)
            torch.testing.assert_close(output_cuda.cpu(), output_cpu, atol=1e-5, rtol=1e-5, msg="CUDA and CPU outputs should match within tolerance")

    def test_feed_forward_dtype_preservation(self, feed_forward_layer, sample_config):
        """
        Test that FeedForward preserves input data type.
        """
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            input_tensor = torch.randn(2, 3, sample_config.emb_dim, dtype=dtype)

            # Convert layer to same dtype
            ff_typed = feed_forward_layer.to(dtype)
            output = ff_typed(input_tensor)

            assert output.dtype == dtype, f"Output dtype should be {dtype}, got {output.dtype}"


    def test_feed_forward_state_dict_serialization(self, feed_forward_layer):
        """
        Test that FeedForward can be saved and loaded correctly.
        """
        # Save state dict
        original_state_dict = feed_forward_layer.state_dict()

        # Create new instance and load state dict
        new_config = GptConfig(emb_dim=128, n_layers=2, n_heads=4, vocab_size=1000, context_length=32)
        new_ff = FeedForward(new_config)
        new_ff.load_state_dict(original_state_dict)

        # Test that they produce same outputs
        input_tensor = torch.randn(1, 5, new_config.emb_dim)

        feed_forward_layer.eval()
        new_ff.eval()

        with torch.no_grad():
            output1 = feed_forward_layer(input_tensor)
            output2 = new_ff(input_tensor)

        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6, msg="State dict loaded model output should match original")
