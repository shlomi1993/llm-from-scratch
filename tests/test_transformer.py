import pytest
import torch
import torch.nn as nn

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from src.gpt import GptModel
from src.transformer import TransformerBlock, TransformerBlockCached


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

    def test_forward_pass_basic(self, transformer_block, sample_config):
        """
        Test forward pass with various input shapes.
        """
        test_cases = [
            (1, 1, sample_config.emb_dim),
            (1, 5, sample_config.emb_dim),
            (4, 10, sample_config.emb_dim),
            (8, 16, sample_config.emb_dim),
            (2, sample_config.context_length, sample_config.emb_dim),
        ]

        transformer_block.eval()
        for shape in test_cases:
            input_tensor = torch.randn(shape)
            output = transformer_block(input_tensor)

            assert output.shape == shape, f"Failed for shape {shape}: expected {shape}, got {output.shape}"
            assert torch.isfinite(output).all(), f"Output should be finite for shape {shape}"
            assert not torch.isnan(output).any(), f"Output should not contain NaN for shape {shape}"

    def test_pre_ln_architecture(self, sample_config):
        """
        Test Pre-LayerNorm architecture: normalization before attention/FF, not after.
        """
        block = TransformerBlock(sample_config)
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        block.eval()

        with torch.no_grad():
            # Manually trace through Pre-LN architecture
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

        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=1e-5,
                                   msg="Pre-LN architecture not correctly implemented")

    def test_residual_connections(self, transformer_block, sample_config):
        """
        Test that residual connections preserve gradient flow and information.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)
        transformer_block.eval()

        with torch.no_grad():
            output = transformer_block(input_tensor)

        # Output should differ from input due to transformations
        assert not torch.allclose(output, input_tensor), "Output should differ from input after transformations"

        # But maintain reasonable magnitude (residual connections help)
        input_norm = torch.norm(input_tensor)
        output_norm = torch.norm(output)
        ratio = output_norm / input_norm

        assert 0.5 < ratio < 2.0, f"Output norm ratio {ratio:.2f} should be reasonable (0.5-2.0)"

    def test_gradient_flow(self, transformer_block, sample_config):
        """
        Test that gradients flow properly through all transformer components.
        """
        transformer_block.train()
        input_tensor = torch.randn(2, 3, sample_config.emb_dim, requires_grad=True)

        output = transformer_block(input_tensor)
        loss = output.sum()
        loss.backward()

        # Verify input gradients
        assert input_tensor.grad is not None, "Input tensor should have gradients"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradients should be finite"
        assert not torch.isnan(input_tensor.grad).any(), "Input gradients should not be NaN"

        # Verify all parameters have gradients
        for name, param in transformer_block.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

    def test_train_eval_modes(self, transformer_block, sample_config):
        """
        Test that train and eval modes affect dropout behavior correctly.
        """
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Training mode - dropout should cause variation
        transformer_block.train()
        outputs_train = []
        for _ in range(5):
            with torch.no_grad():
                output = transformer_block(input_tensor)
                outputs_train.append(output)

        # Check if outputs vary (dropout is stochastic)
        all_same = all(torch.equal(outputs_train[0], out) for out in outputs_train[1:])
        assert not all_same, "Training mode outputs should vary due to dropout"

        # Eval mode - should be deterministic
        transformer_block.eval()
        outputs_eval = []
        for _ in range(3):
            with torch.no_grad():
                output = transformer_block(input_tensor)
                outputs_eval.append(output)

        all_same = all(torch.equal(outputs_eval[0], out) for out in outputs_eval[1:])
        assert all_same, "Eval mode outputs should be deterministic"

    def test_zero_dropout(self, sample_config):
        """
        Test transformer with zero dropout for deterministic behavior.
        """
        zero_dropout_config = GptConfig(
            emb_dim=sample_config.emb_dim,
            n_layers=sample_config.n_layers,
            n_heads=sample_config.n_heads,
            vocab_size=sample_config.vocab_size,
            context_length=sample_config.context_length,
            drop_rate=0.0,
            qkv_bias=sample_config.qkv_bias
        )

        block = TransformerBlock(zero_dropout_config)
        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # With zero dropout, train and eval should give identical results
        torch.manual_seed(123)
        block.train()
        output_train = block(input_tensor.clone())

        torch.manual_seed(123)
        block.eval()
        output_eval = block(input_tensor.clone())

        torch.testing.assert_close(output_train, output_eval, atol=1e-6, rtol=1e-6,
                                   msg="Zero dropout should produce identical outputs in train and eval modes")

    def test_batch_independence(self, transformer_block, sample_config):
        """
        Test that batch items are processed independently (no cross-contamination).
        """
        batch_size, seq_len = 3, 6
        batch_input = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        transformer_block.eval()
        with torch.no_grad():
            batch_output = transformer_block(batch_input)

        # Process each item individually
        individual_outputs = []
        for i in range(batch_size):
            with torch.no_grad():
                individual_output = transformer_block(batch_input[i:i+1])
            individual_outputs.append(individual_output)

        concat_output = torch.cat(individual_outputs, dim=0)

        torch.testing.assert_close(batch_output, concat_output, atol=1e-6, rtol=1e-6,
                                   msg="Batch processing should be equivalent to individual processing")

    def test_stacked_blocks(self, sample_config):
        """
        Test stacking multiple transformer blocks.
        """
        num_blocks = 3
        blocks = nn.ModuleList([TransformerBlock(sample_config) for _ in range(num_blocks)])

        input_tensor = torch.randn(2, 5, sample_config.emb_dim)

        # Forward through stacked blocks
        x = input_tensor
        for block in blocks:
            x = block(x)

        assert x.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {x.shape}"
        assert torch.isfinite(x).all(), "Stacked blocks output should be finite"
        assert not torch.allclose(x, input_tensor), "Stacked blocks should transform input"

    def test_component_structure(self):
        """
        Test the structure and components of TransformerBlock (Exercise 4.1).
        """
        block = TransformerBlock(GPT_CONFIG_124M)

        # Verify component types
        assert isinstance(block.att, torch.nn.Module), "att should be a Module"
        assert isinstance(block.ff, torch.nn.Module), "ff should be a Module"
        assert isinstance(block.norm1, torch.nn.Module), "norm1 should be a Module"
        assert isinstance(block.norm2, torch.nn.Module), "norm2 should be a Module"
        assert isinstance(block.drop_shortcut, torch.nn.Dropout), "drop_shortcut should be Dropout"

        # Verify string representation contains expected components
        block_str = str(block)
        assert "MultiheadAttention" in block_str, "Should contain MultiheadAttention"
        assert "FeedForward" in block_str, "Should contain FeedForward"
        assert "LayerNorm" in block_str, "Should contain LayerNorm"
        assert "Dropout" in block_str, "Should contain Dropout"

        # Verify attention structure
        att_str = str(block.att)
        assert "W_query" in att_str, "Attention should have W_query"
        assert "W_key" in att_str, "Attention should have W_key"
        assert "W_value" in att_str, "Attention should have W_value"
        assert "out_proj" in att_str, "Attention should have out_proj"
        assert "Linear(in_features=768, out_features=768, bias=False)" in att_str, "Should have QKV projections"
        assert "Linear(in_features=768, out_features=768, bias=True)" in att_str, "Should have output projection"
        assert "Dropout(p=0.1" in att_str, "Should have dropout with p=0.1"

        # Verify feed-forward structure
        ff_str = str(block.ff)
        assert "Sequential" in ff_str, "FeedForward should use Sequential"
        assert "Linear(in_features=768, out_features=3072, bias=True)" in ff_str, "Should expand to 4x dimension"
        assert "GELU" in ff_str, "Should use GELU activation"
        assert "Linear(in_features=3072, out_features=768, bias=True)" in ff_str, "Should contract back to original dimension"

        # Verify parameter counts
        total_ff_params = sum(p.numel() for p in block.ff.parameters())
        assert total_ff_params == 4_722_432, f"FeedForward should have 4,722,432 parameters, got {total_ff_params:,}"

        total_att_params = sum(p.numel() for p in block.att.parameters())
        assert total_att_params == 2_360_064, f"Attention should have 2,360,064 parameters, got {total_att_params:,}"

        # Verify functionality
        input_tensor = torch.randn(1, 10, GPT_CONFIG_124M.emb_dim)
        output = block(input_tensor)
        assert output.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output.shape}"
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
        """
        model = GptModel(config)

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected_total, \
            f"{model_name}: Expected {expected_total:,} total parameters, got {total_params:,}"

        # Calculate parameters excluding output head (simulating weight tying)
        params_without_output_head = total_params - sum(p.numel() for p in model.out_head.parameters())
        assert params_without_output_head == expected_tied, \
            f"{model_name}: Expected {expected_tied:,} parameters with weight tying, got {params_without_output_head:,}"

        # Calculate model size in MB (float32 = 4 bytes per parameter)
        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        assert abs(total_size_mb - expected_size_mb) < 0.01, \
            f"{model_name}: Expected {expected_size_mb:.2f} MB, got {total_size_mb:.2f} MB"

        # Test forward pass
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids)
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert output.shape == expected_shape, \
                f"{model_name}: Expected output shape {expected_shape}, got {output.shape}"
            assert torch.isfinite(output).all(), f"{model_name}: Output should be finite"


class TestTransformerBlockCached:
    """
    Test suite for the TransformerBlockCached implementation with KV cache support.
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
    def cached_transformer_block(self, sample_config):
        """
        Create a TransformerBlockCached for testing.
        """
        return TransformerBlockCached(sample_config)

    def test_forward_pass_basic(self, cached_transformer_block, sample_config):
        """
        Test forward pass with and without cache.
        """
        batch_size, seq_len = 4, 10
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        # Test uncached forward pass
        output_uncached = cached_transformer_block(input_tensor, use_cache=False)
        expected_shape = (batch_size, seq_len, sample_config.emb_dim)
        assert output_uncached.shape == expected_shape, \
            f"Uncached: Expected output shape {expected_shape}, got {output_uncached.shape}"
        assert torch.isfinite(output_uncached).all(), "Uncached output should be finite"

        # Test cached forward pass
        cached_transformer_block.att.reset_cache()
        output_cached = cached_transformer_block(input_tensor, use_cache=True)
        assert output_cached.shape == expected_shape, \
            f"Cached: Expected output shape {expected_shape}, got {output_cached.shape}"
        assert torch.isfinite(output_cached).all(), "Cached output should be finite"

    def test_inheritance_from_transformer_block(self, cached_transformer_block):
        """
        Test that TransformerBlockCached inherits from TransformerBlock.
        """
        # TransformerBlockCached doesn't inherit from TransformerBlock in this implementation
        # It's a separate class with similar structure
        assert hasattr(cached_transformer_block, 'att'), "Should have attention layer"
        assert hasattr(cached_transformer_block, 'ff'), "Should have feed-forward layer"
        assert hasattr(cached_transformer_block, 'norm1'), "Should have first normalization layer"
        assert hasattr(cached_transformer_block, 'norm2'), "Should have second normalization layer"
        assert hasattr(cached_transformer_block, 'drop_shortcut'), "Should have dropout layer"

    def test_cached_vs_uncached_consistency(self, cached_transformer_block, sample_config):
        """
        Test that cached and uncached outputs are consistent for initial forward pass.
        """
        batch_size, seq_len = 2, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        cached_transformer_block.eval()

        with torch.no_grad():
            # Reset cache and run uncached
            cached_transformer_block.att.reset_cache()
            output_uncached = cached_transformer_block(input_tensor, use_cache=False)

            # Reset cache and run cached
            cached_transformer_block.att.reset_cache()
            output_cached = cached_transformer_block(input_tensor, use_cache=True)

        torch.testing.assert_close(output_uncached, output_cached, atol=1e-6, rtol=1e-6,
                                   msg="Cached and uncached outputs should be close for initial forward pass")

    def test_kv_cache_propagation(self, cached_transformer_block, sample_config):
        """
        Test that use_cache parameter properly propagates to attention layer.
        """
        batch_size, seq_len = 2, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        # Forward pass with cache enabled
        cached_transformer_block.att.reset_cache()
        cached_transformer_block(input_tensor, use_cache=True)

        # Verify cache was populated
        assert cached_transformer_block.att.cache_k is not None, "Cache K should exist after cached forward pass"
        assert cached_transformer_block.att.cache_v is not None, "Cache V should exist after cached forward pass"
        assert cached_transformer_block.att.ptr_cur == seq_len, \
            f"Pointer should be {seq_len} after cached forward pass, got {cached_transformer_block.att.ptr_cur}"

        # Store cache state
        cache_k_before = cached_transformer_block.att.cache_k.clone()
        cache_v_before = cached_transformer_block.att.cache_v.clone()

        # Forward pass without cache
        cached_transformer_block(input_tensor, use_cache=False)

        # Cache tensors should remain but ptr_cur is reset
        assert cached_transformer_block.att.cache_k is not None, "Cache K should still exist after uncached forward pass"
        assert cached_transformer_block.att.cache_v is not None, "Cache V should still exist after uncached forward pass"
        assert cached_transformer_block.att.ptr_cur == 0, \
            f"Pointer should be reset to 0 after uncached forward pass, got {cached_transformer_block.att.ptr_cur}"

        # Cache tensors should remain unchanged
        torch.testing.assert_close(cached_transformer_block.att.cache_k, cache_k_before,
                                   msg="Cache K should remain unchanged after uncached forward pass")
        torch.testing.assert_close(cached_transformer_block.att.cache_v, cache_v_before,
                                   msg="Cache V should remain unchanged after uncached forward pass")

    def test_incremental_generation_with_cache(self, cached_transformer_block, sample_config):
        """
        Test that cache enables efficient incremental generation.
        """
        batch_size = 1
        cached_transformer_block.eval()

        with torch.no_grad():
            # Reset cache and process initial sequence
            cached_transformer_block.att.reset_cache()
            initial_seq = torch.randn(batch_size, 3, sample_config.emb_dim)
            output1 = cached_transformer_block(initial_seq, use_cache=True)

            assert output1.shape == (batch_size, 3, sample_config.emb_dim), \
                f"Expected shape (1, 3, {sample_config.emb_dim}), got {output1.shape}"

            # Process next token using cache
            next_token = torch.randn(batch_size, 1, sample_config.emb_dim)
            output2 = cached_transformer_block(next_token, use_cache=True)

            assert output2.shape == (batch_size, 1, sample_config.emb_dim), \
                f"Expected shape (1, 1, {sample_config.emb_dim}), got {output2.shape}"
            assert torch.isfinite(output2).all(), "Output for next token should be finite"

            # Verify cache grew
            assert cached_transformer_block.att.ptr_cur == 4, \
                f"Cache pointer should be 4 after processing 3+1 tokens, got {cached_transformer_block.att.ptr_cur}"

    def test_cache_reset(self, cached_transformer_block, sample_config):
        """
        Test that cache can be properly reset.
        """
        input_tensor = torch.randn(1, 5, sample_config.emb_dim)

        # Use cache
        cached_transformer_block.att.reset_cache()
        cached_transformer_block(input_tensor, use_cache=True)

        assert cached_transformer_block.att.ptr_cur == 5, "Cache pointer should be 5"
        assert cached_transformer_block.att.cache_k is not None, "Cache K should exist"
        assert cached_transformer_block.att.cache_v is not None, "Cache V should exist"

        # Reset cache
        cached_transformer_block.att.reset_cache()

        # Cache tensors are reset to None
        assert cached_transformer_block.att.cache_k is None, "Cache K should be None after reset"
        assert cached_transformer_block.att.cache_v is None, "Cache V should be None after reset"
        # Note: ptr_cur is not reset by reset_cache(), it's only reset when use_cache=False in forward

    def test_attention_type_selection(self, sample_config):
        """
        Test that correct attention type is selected based on n_kv_groups.
        """
        # Test MHA selection (n_kv_groups == 1)
        mha_config = GptConfig(
            emb_dim=sample_config.emb_dim,
            n_layers=sample_config.n_layers,
            n_heads=sample_config.n_heads,
            vocab_size=sample_config.vocab_size,
            context_length=sample_config.context_length,
            drop_rate=sample_config.drop_rate,
            qkv_bias=sample_config.qkv_bias,
            n_kv_groups=1
        )
        mha_block = TransformerBlockCached(mha_config)
        assert "MultiheadAttentionCached" in str(type(mha_block.att).__name__), \
            "Should use MultiheadAttentionCached when n_kv_groups == 1"

        # Test GQA selection (n_kv_groups > 1)
        gqa_config = GptConfig(
            emb_dim=sample_config.emb_dim,
            n_layers=sample_config.n_layers,
            n_heads=8,
            vocab_size=sample_config.vocab_size,
            context_length=sample_config.context_length,
            drop_rate=sample_config.drop_rate,
            qkv_bias=sample_config.qkv_bias,
            n_kv_groups=2
        )
        gqa_block = TransformerBlockCached(gqa_config)
        assert "GroupedQueryAttention" in str(type(gqa_block.att).__name__), \
            "Should use GroupedQueryAttention when n_kv_groups > 1"

    def test_gradient_flow_with_cache(self, cached_transformer_block, sample_config):
        """
        Test gradient flow through cached transformer block.
        """
        cached_transformer_block.train()
        input_tensor = torch.randn(2, 3, sample_config.emb_dim, requires_grad=True)

        # Test with cache disabled (more common for training)
        output = cached_transformer_block(input_tensor, use_cache=False)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None, "Input should have gradients"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradients should be finite"

        for name, param in cached_transformer_block.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"

    def test_batch_processing_with_cache(self, cached_transformer_block, sample_config):
        """
        Test that caching works correctly with batch processing.
        """
        batch_size, seq_len = 3, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        cached_transformer_block.eval()
        with torch.no_grad():
            cached_transformer_block.att.reset_cache()
            output = cached_transformer_block(input_tensor, use_cache=True)

        expected_shape = (batch_size, seq_len, sample_config.emb_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Batch output with cache should be finite"
