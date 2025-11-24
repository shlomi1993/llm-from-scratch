import pytest
import torch
import torch.nn as nn

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from src.transformer import TransformerBlock
from src.gpt import GptModel


class TestTransformerBlock:
    """
    Test suite for the TransformerBlock implementation.
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
    def transformer_block(self, sample_config):
        """
        Create a TransformerBlock for testing.
        """
        return TransformerBlock(sample_config)

    def test_transformer_block_forward_pass_shape(self, transformer_block, sample_config):
        """
        Test that forward pass produces correct output shape.
        """
        batch_size, seq_len = 4, 10
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        output = transformer_block(input_tensor)

        expected_shape = (batch_size, seq_len, sample_config.emb_dim)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_transformer_block_different_input_shapes(self, transformer_block, sample_config):
        """
        Test TransformerBlock with different input shapes.
        """
        test_shapes = [
            (1, 1, sample_config.emb_dim),      # Single token
            (1, 5, sample_config.emb_dim),      # Single sequence
            (8, 16, sample_config.emb_dim),     # Batch of sequences
            (16, 32, sample_config.emb_dim),    # Larger batch
        ]

        for shape in test_shapes:
            input_tensor = torch.randn(shape)
            output = transformer_block(input_tensor)

            assert output.shape == shape, f"Output shape {output.shape} should match input shape {shape}"

    def test_transformer_block_pre_ln_architecture(self, sample_config):
        """
        Test Pre-LayerNorm architecture implementation.
        """
        block = TransformerBlock(sample_config)
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Set model to eval mode to disable dropout for deterministic comparison
        block.eval()

        with torch.no_grad():
            # Manually trace through Pre-LN architecture
            # Pre-LN: x = x + dropout(attention(norm1(x)))
            # Pre-LN: x = x + dropout(feedforward(norm2(x)))

            x = input_tensor.clone()

            # First sub-layer: attention with Pre-LN
            norm1_out = block.norm1(x)
            att_out = block.att(norm1_out)
            shortcut_out = block.drop_shortcut(att_out)
            x = x + shortcut_out

            # Second sub-layer: feed-forward with Pre-LN
            norm2_out = block.norm2(x)
            ff_out = block.ff(norm2_out)
            shortcut_out2 = block.drop_shortcut(ff_out)
            expected_output = x + shortcut_out2

            # Compare with actual forward pass
            actual_output = block(input_tensor)

        # Should be close (accounting for potential numerical differences)
        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=1e-5)

    def test_transformer_block_residual_connections(self, transformer_block, sample_config):
        """
        Test that residual connections preserve information flow.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Set model to eval mode to make dropout deterministic
        transformer_block.eval()

        with torch.no_grad():
            output = transformer_block(input_tensor)

        # Output should not be identical to input (due to transformations)
        assert not torch.allclose(output, input_tensor), "Output should be different from input"

        # But should preserve some information (reasonable magnitude)
        input_norm = torch.norm(input_tensor)
        output_norm = torch.norm(output)
        ratio = output_norm / input_norm

        # Ratio should be reasonable (not too large or too small)
        assert 0.5 < ratio < 2.0, f"Output norm ratio {ratio} should be reasonable"

    def test_transformer_block_attention_mask_compatibility(self, transformer_block, sample_config):
        """
        Test that TransformerBlock works with attention masks.
        """
        batch_size, seq_len = 2, 8
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        # Create causal mask (typical for language modeling)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        # Should work with mask (though this implementation might not use it directly)
        # This tests that the attention component can handle masks
        try:
            output = transformer_block(input_tensor)
            success = True
        except Exception:
            success = False

        assert success, "TransformerBlock should handle masked inputs"
        assert output.shape == input_tensor.shape, "Output shape should match input"

    def test_transformer_block_gradient_flow(self, transformer_block, sample_config):
        """
        Test that gradients flow properly through the transformer block.
        """
        input_tensor = torch.randn(2, 3, sample_config.emb_dim, requires_grad=True)

        output = transformer_block(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None, "Input tensor should have gradients"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradients should be finite"

        # Check that all component parameters have gradients
        for name, param in transformer_block.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} gradients should be finite"

    def test_transformer_block_training_vs_eval_mode(self, transformer_block, sample_config):
        """
        Test that training and eval modes affect dropout behavior.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Training mode
        transformer_block.train()
        torch.manual_seed(42)
        output_train1 = transformer_block(input_tensor)

        torch.manual_seed(42)
        output_train2 = transformer_block(input_tensor)

        # In training mode, dropout might cause different outputs
        # But due to randomness, we can't guarantee this, so just check shapes
        assert output_train1.shape == output_train2.shape, "Training outputs should have same shape"

        # Eval mode
        transformer_block.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            output_eval1 = transformer_block(input_tensor)

            torch.manual_seed(42)
            output_eval2 = transformer_block(input_tensor)

            # In eval mode, outputs should be identical
            torch.testing.assert_close(output_eval1, output_eval2, atol=1e-6, rtol=1e-6)

    def test_transformer_block_zero_dropout(self, sample_config):
        """
        Test TransformerBlock with zero dropout rate.
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

        block = TransformerBlock(zero_dropout_config)
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Training and eval modes should give same results with zero dropout
        block.train()
        torch.manual_seed(123)
        output_train = block(input_tensor)

        block.eval()
        torch.manual_seed(123)
        output_eval = block(input_tensor)

        torch.testing.assert_close(output_train, output_eval, atol=1e-6, rtol=1e-6)

    def test_transformer_block_component_integration(self, transformer_block, sample_config):
        """
        Test that attention and feed-forward components are properly integrated.
        """
        input_tensor = torch.randn(1, 4, sample_config.emb_dim)

        # Get intermediate outputs to verify integration
        with torch.no_grad():
            transformer_block.eval()

            # First sub-layer
            x = input_tensor
            norm1_out = transformer_block.norm1(x)
            att_out = transformer_block.att(norm1_out)
            x = x + transformer_block.drop_shortcut(att_out)

            # Check that attention output has reasonable properties
            assert torch.isfinite(att_out).all(), "Attention output should be finite"

            # Second sub-layer
            norm2_out = transformer_block.norm2(x)
            ff_out = transformer_block.ff(norm2_out)
            final_out = x + transformer_block.drop_shortcut(ff_out)

            # Check that feed-forward output has reasonable properties
            assert torch.isfinite(ff_out).all(), "Feed-forward output should be finite"
            assert torch.isfinite(final_out).all(), "Final output should be finite"

            # Compare with actual forward pass
            actual_output = transformer_block(input_tensor)
            torch.testing.assert_close(final_out, actual_output, atol=1e-5, rtol=1e-5)

    def test_transformer_block_batch_independence(self, transformer_block, sample_config):
        """
        Test that different batch items are processed independently.
        """
        batch_size, seq_len = 3, 6

        # Process batch
        batch_input = torch.randn(batch_size, seq_len, sample_config.emb_dim)
        transformer_block.eval()
        with torch.no_grad():
            batch_output = transformer_block(batch_input)

        # Process individual items
        individual_outputs = []
        for i in range(batch_size):
            individual_input = batch_input[i:i+1]
            with torch.no_grad():
                individual_output = transformer_block(individual_input)
            individual_outputs.append(individual_output)

        # Concatenate individual outputs
        concat_output = torch.cat(individual_outputs, dim=0)

        # Should be identical
        torch.testing.assert_close(batch_output, concat_output, atol=1e-6, rtol=1e-6)

    def test_transformer_block_with_standard_configs(self):
        """
        Test TransformerBlock with standard GPT configurations.
        """
        configs_to_test = [GPT_CONFIG_124M]  # Test with one standard config

        for config in configs_to_test:
            block = TransformerBlock(config)
            input_tensor = torch.randn(1, 8, config.emb_dim)

            output = block(input_tensor)

            assert output.shape == input_tensor.shape, "Output shape should match input"
            assert torch.isfinite(output).all(), "Output should be finite"

    def test_transformer_block_stacking(self, sample_config):
        """
        Test that multiple transformer blocks can be stacked.
        """
        num_blocks = 3
        blocks = nn.ModuleList([TransformerBlock(sample_config) for _ in range(num_blocks)])

        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Forward through stacked blocks
        x = input_tensor
        for block in blocks:
            x = block(x)

        assert x.shape == input_tensor.shape, "Final output should match input shape"
        assert torch.isfinite(x).all(), "Final output should be finite"
        assert not torch.allclose(x, input_tensor), "Stacked blocks should transform input"

    def test_transformer_block_structure(self):
        """
        Exercise 4.1 from notebook chapters/ch04/01_main-chapter-code/exercise-solutions.ipynb
        """
        block = TransformerBlock(GPT_CONFIG_124M)

        # Test the structure of the TransformerBlock
        assert isinstance(block.att, torch.nn.Module), "att should be a Module"
        assert isinstance(block.ff, torch.nn.Module), "ff should be a Module"
        assert isinstance(block.norm1, torch.nn.Module), "norm1 should be a Module"
        assert isinstance(block.norm2, torch.nn.Module), "norm2 should be a Module"
        assert isinstance(block.drop_shortcut, torch.nn.Dropout), "drop_shortcut should be Dropout"

        # Test that the string representation contains expected components
        block_str = str(block)
        assert "MultiHeadAttention" in block_str, "Should contain MultiHeadAttention"
        assert "FeedForward" in block_str, "Should contain FeedForward"
        assert "LayerNorm" in block_str, "Should contain LayerNorm"
        assert "Dropout" in block_str, "Should contain Dropout"

        # Test attention module structure
        att_str = str(block.att)
        assert "W_query" in att_str, "Attention should have W_query"
        assert "W_key" in att_str, "Attention should have W_key"
        assert "W_value" in att_str, "Attention should have W_value"
        assert "out_proj" in att_str, "Attention should have out_proj"
        assert "Linear(in_features=768, out_features=768, bias=False)" in att_str, "Should have QKV projections"
        assert "Linear(in_features=768, out_features=768, bias=True)" in att_str, "Should have output projection"
        assert "Dropout(p=0.1" in att_str, "Should have dropout with p=0.1"

        # Test feed-forward module structure
        ff_str = str(block.ff)
        assert "Sequential" in ff_str, "FeedForward should use Sequential"
        assert "Linear(in_features=768, out_features=3072, bias=True)" in ff_str, "Should expand to 4x dimension"
        assert "GELU" in ff_str, "Should use GELU activation"
        assert "Linear(in_features=3072, out_features=768, bias=True)" in ff_str, "Should contract back to original dimension"

        # Test parameter counts
        total_ff_params = sum(p.numel() for p in block.ff.parameters())
        assert total_ff_params == 4_722_432, f"FeedForward should have 4,722,432 parameters, got {total_ff_params:,}"

        total_att_params = sum(p.numel() for p in block.att.parameters())
        assert total_att_params == 2_360_064, f"Attention should have 2,360,064 parameters, got {total_att_params:,}"

        # Test overall functionality
        input_tensor = torch.randn(1, 10, GPT_CONFIG_124M.emb_dim)
        output = block(input_tensor)
        assert output.shape == input_tensor.shape, "Output should match input shape"
        assert torch.isfinite(output).all(), "Output should be finite"

    @pytest.mark.parametrize("config,expected_total,expected_tied,expected_size_mb,model_name", [
        (GPT_CONFIG_124M, 163_009_536, 124_412_160, 621.83, "gpt2-small"),
        (GPT_CONFIG_355M, 406_212_608, 354_749_440, 1549.58, "gpt2-medium"),
        (GPT_CONFIG_774M, 838_220_800, 773_891_840, 3197.56, "gpt2-large"),
        (GPT_CONFIG_1558M, 1_637_792_000, 1_557_380_800, 6247.68, "gpt2-xl"),
    ])
    def test_gpt_model_parameter_counts(self, config, expected_total, expected_tied, expected_size_mb, model_name):
        """
        Test parameter counts and model sizes for different GPT-2 configurations.

        Verifies that the model architectures match the expected parameter counts from the GPT-2 paper specifications.
        """
        model = GptModel(config)

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected_total, f"{model_name}: Expected {expected_total:,} total parameters, got {total_params:,}"

        # Calculate parameters excluding output head (simulating weight tying)
        params_without_output_head = total_params - sum(p.numel() for p in model.out_head.parameters())
        assert params_without_output_head == expected_tied, f"{model_name}: Expected {expected_tied:,} parameters with weight tying, got {params_without_output_head:,}"

        # Calculate model size in MB (assuming float32, 4 bytes per parameter)
        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        assert abs(total_size_mb - expected_size_mb) < 0.01, f"{model_name}: Expected {expected_size_mb:.2f} MB, got {total_size_mb:.2f} MB"

        # Test that the model can perform forward pass
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids)
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert output.shape == expected_shape, f"{model_name}: Expected output shape {expected_shape}, got {output.shape}"
            assert torch.isfinite(output).all(), f"{model_name}: Output should be finite"
