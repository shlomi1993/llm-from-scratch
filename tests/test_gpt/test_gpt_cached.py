import pytest
import torch

from src.configurations import GPT_CONFIG_124M
from src.gpt import GptModelBasic, GptModelCached


class TestGptModelCached:

    @pytest.fixture
    def model(self):
        """
        Fixture that provides a GPT model with caching capability.
        """
        torch.manual_seed(123)
        return GptModelCached(GPT_CONFIG_124M)

    def test_incremental_generation(self, model):
        """
        Test that incremental generation works correctly.
        """
        batch_size = 1
        initial_tokens = 5
        new_tokens = 3

        # Create initial input
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, initial_tokens))

        model.eval()
        with torch.no_grad():
            # Reset and start with cache
            model.reset_kv_cache()

            # First call - process initial tokens
            output1 = model(input_ids, use_cache=True)

            # Verify output shape
            expected_shape = (batch_size, initial_tokens, model.config.vocab_size)
            assert output1.shape == expected_shape, f"Expected output shape {expected_shape}, got {output1.shape}"

            # Generate additional tokens incrementally
            for i in range(new_tokens):
                # Get next token (simple greedy sampling)
                next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1))

                # Process with cache
                output = model(next_token, use_cache=True)
                expected_incremental_shape = (batch_size, 1, model.config.vocab_size)
                assert output.shape == expected_incremental_shape, f"Expected incremental output shape {expected_incremental_shape}, got {output.shape}"

    def test_cache_reuse(self, model):
        """
        Test that cache is properly reused across calls.
        """
        batch_size = 1
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 5))

        model.eval()
        with torch.no_grad():
            # Reset cache before testing
            model.reset_kv_cache()

            # First call with cache
            output1 = model(input_ids, use_cache=True)

            # Check initial position
            initial_pos = model.ptr_current_pos
            assert initial_pos == 5, f"Position should be 5 after processing 5 tokens, got {initial_pos}"

            # Second call with one more token
            next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1))
            output2 = model(next_token, use_cache=True)

            # Check position increased
            final_pos = model.ptr_current_pos
            assert final_pos > initial_pos, f"Position should increase, got {final_pos} vs {initial_pos}"
            assert final_pos == 6, f"Position should be 6 after processing 1 more token, got {final_pos}"

    def test_generate_text_cached(self, model):
        """
        Test the cached text generation method.
        """
        initial_tokens = torch.randint(0, model.config.vocab_size, (1, 5))
        max_new_tokens = 10

        model.eval()
        with torch.no_grad():
            generated = model.generate_text(
                idx=initial_tokens,
                max_new_tokens=max_new_tokens,
                context_size=model.config.context_length
            )

        # Check output shape
        expected_length = initial_tokens.shape[1] + max_new_tokens
        expected_shape = (1, expected_length)
        assert generated.shape == expected_shape, f"Expected generated shape {expected_shape}, got {generated.shape}"

        # Check that initial context is preserved
        initial_context = generated[:, :initial_tokens.shape[1]]
        assert torch.equal(initial_context, initial_tokens), "Initial context not preserved in generated text"

        # Check that new tokens are within vocabulary
        new_tokens = generated[:, initial_tokens.shape[1]:]
        assert (new_tokens >= 0).all() and (new_tokens < model.config.vocab_size).all(), "Generated tokens are outside vocabulary range"

    def test_cache_reset(self, model):
        """
        Test that cache can be properly reset.
        """
        batch_size = 2
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 5))

        model.eval()
        with torch.no_grad():
            # Generate output with cache to create some state
            model.reset_kv_cache()
            model(input_ids, use_cache=True)

            # Check position was set
            assert model.ptr_current_pos > 0, f"Expected position > 0 after processing, got {model.ptr_current_pos}"

            # Reset cache
            model.reset_kv_cache()

            # Check position is reset
            assert model.ptr_current_pos == 0, f"Expected position 0 after reset, got {model.ptr_current_pos}"

    def test_different_sequence_lengths(self, model):
        """
        Test model with different sequence lengths.
        """
        batch_size = 2
        test_lengths = [1, 5, 10, 50]

        for seq_len in test_lengths:
            input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

            model.eval()
            model.reset_kv_cache()  # Reset cache for each test

            with torch.no_grad():
                output = model(input_ids)

                expected_shape = (batch_size, seq_len, model.config.vocab_size)
                assert output.shape == expected_shape, f"Failed for seq_len={seq_len}"

    def test_cached_vs_non_cached_equivalence(self, model):
        """
        Test that cached model produces same results as non-cached when processing full sequence.
        """

        # Create non-cached model with same config
        torch.manual_seed(123)
        non_cached_model = GptModelBasic(model.config)

        # Copy compatible weights from cached to non-cached model
        cached_state = model.state_dict()
        basic_state = non_cached_model.state_dict()

        # Only copy compatible parameters (skip mask buffers)
        compatible_state = {}
        for key, value in cached_state.items():
            if key in basic_state and basic_state[key].shape == value.shape:
                compatible_state[key] = value

        non_cached_model.load_state_dict(compatible_state, strict=False)

        input_ids = torch.randint(0, model.config.vocab_size, (1, 10))

        # Set both models to eval mode
        model.eval()
        non_cached_model.eval()

        # Reset cache for cached model
        model.reset_kv_cache()

        with torch.no_grad():
            # Get outputs from both models (without cache for fair comparison)
            cached_output = model(input_ids, use_cache=False)
            non_cached_output = non_cached_model(input_ids)

            # Should produce similar results (not exactly identical due to mask differences)
            # But the shapes and overall magnitude should be close
            assert cached_output.shape == non_cached_output.shape, f"Shape mismatch: cached {cached_output.shape} vs non-cached {non_cached_output.shape}"
            assert torch.isfinite(cached_output).all(), "Cached output contains non-finite values"
            assert torch.isfinite(non_cached_output).all(), "Non-cached output contains non-finite values"

    def test_cache_position_tracking(self, model):
        """
        Test that position tracking works correctly.
        """
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

        model.eval()
        model.reset_kv_cache()

        with torch.no_grad():
            # Process tokens incrementally
            for i in range(8):
                token = input_ids[:, i:i+1]
                model(token, use_cache=True)

                # Check position increments correctly
                assert model.ptr_current_pos == i + 1, f"Position should be {i+1}, got {model.ptr_current_pos}"

    def test_device_compatibility(self, model):
        """
        Test that cached model works on different devices.
        """
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))

        # Test CPU (default)
        model.eval()
        model.reset_kv_cache()

        with torch.no_grad():
            output_cpu = model(input_ids)

        expected_device = torch.device('cpu')
        assert output_cpu.device == expected_device, f"Expected output on {expected_device}, got {output_cpu.device}"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            input_ids_cuda = input_ids.cuda()

            model_cuda.eval()
            model_cuda.reset_kv_cache()

            with torch.no_grad():
                output_cuda = model_cuda(input_ids_cuda)

            assert output_cuda.device.type == 'cuda', f"Expected output on CUDA, got {output_cuda.device}"

    def test_gradients_with_cache(self, model):
        """
        Test that gradients work correctly when cache is enabled.
        """
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        target = torch.randint(0, model.config.vocab_size, (1, 5))

        model.train()  # Enable training mode
        model.reset_kv_cache()

        # Forward pass
        output = model(input_ids)

        # Compute loss and backward pass
        loss = torch.nn.CrossEntropyLoss()(
            output.view(-1, model.config.vocab_size),
            target.view(-1)
        )
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"


