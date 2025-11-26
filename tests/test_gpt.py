import pytest
import torch
import torch.nn as nn
import tiktoken

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M
from src.gpt import GptModel


class TestGptModel:
    """
    Test suite for the GptModel class.
    """

    @pytest.fixture
    def sample_config(self) -> GptConfig:
        """
        Create a small test configuration for faster testing.
        """
        return GptConfig(
            emb_dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=32,
            drop_rate=0.1,
            qkv_bias=False
        )

    @pytest.fixture
    def sample_model(self, sample_config: GptConfig) -> GptModel:
        """
        Create a GptModel instance for testing.
        """
        torch.manual_seed(42)
        return GptModel(sample_config)

    def test_forward_pass_basic(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test basic forward pass with various batch sizes and sequence lengths.
        """
        test_cases = [
            (1, 1),
            (1, 10),
            (2, 5),
            (4, sample_config.context_length),
        ]

        sample_model.eval()
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
            output = sample_model(input_ids)

            expected_shape = (batch_size, seq_len, sample_config.vocab_size)
            assert output.shape == expected_shape, f"Failed for batch_size={batch_size}, seq_len={seq_len}"
            assert torch.isfinite(output).all(), "Output should contain only finite values"
            assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_forward_with_cache(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test forward pass with KV cache enabled.
        """
        sample_model.eval()
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        # Reset cache and run with cache
        sample_model.reset_kv_cache()
        output_cached = sample_model(input_ids, use_cache=True)

        # Run without cache
        output_no_cache = sample_model(input_ids, use_cache=False)

        # Shapes should match
        assert output_cached.shape == output_no_cache.shape, "Output shapes should match between cached and non-cached runs"
        assert output_cached.shape == (batch_size, seq_len, sample_config.vocab_size), "Output shape should match expected dimensions"

        # Verify cache position tracking
        assert sample_model.current_pos == seq_len, "Current position should be updated when using cache"

        # Feed another token with cache
        next_token = torch.randint(0, sample_config.vocab_size, (batch_size, 1))
        output_next = sample_model(next_token, use_cache=True)
        assert output_next.shape == (batch_size, 1, sample_config.vocab_size), "Output shape for next token should match expected dimensions"
        assert sample_model.current_pos == seq_len + 1, "Current position should increment after next token"

    def test_reset_kv_cache(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that reset_kv_cache properly resets the cache state.
        """
        sample_model.eval()
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        # Use cache
        sample_model(input_ids, use_cache=True)
        assert sample_model.current_pos == 5, "Current position should be 5 after first use with cache"

        # Reset cache
        sample_model.reset_kv_cache()

        # Verify position is reset correctly
        assert sample_model.current_pos == 0, "Current position should be reset to 0"

        # Use cache again
        sample_model(input_ids, use_cache=True)
        assert sample_model.current_pos == 5, "Current position should be 5 after second use with cache"

    def test_generate_text_simple(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_simple method with greedy decoding.
        """
        sample_model.eval()
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        generated = sample_model.generate_text_simple(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        # Verify output shape
        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), "Generated output shape should match expected dimensions"

        # Verify initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved in generated output"

        # Verify new tokens are valid
        new_tokens = generated[:, initial_context.shape[1]:]
        assert (new_tokens >= 0).all() and (new_tokens < sample_config.vocab_size).all(), "New tokens should be within valid vocabulary range"

    def test_generate_text_simple_batch(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_simple with batch size > 1.
        """
        sample_model.eval()
        batch_size = 3
        initial_context = torch.randint(0, sample_config.vocab_size, (batch_size, 4))
        max_new_tokens = 3

        generated = sample_model.generate_text_simple(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        expected_shape = (batch_size, initial_context.shape[1] + max_new_tokens)
        assert generated.shape == expected_shape, "Generated output shape should match expected dimensions"

        # Verify each batch item preserves its context
        for i in range(batch_size):
            assert torch.equal(generated[i, :initial_context.shape[1]], initial_context[i]), f"Initial context should be preserved for batch item {i}"

    def test_generate_text_simple_context_cropping(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that generate_text_simple properly crops context when it exceeds context_size.
        """
        sample_model.eval()
        # Create context longer than context_length
        long_context = torch.randint(0, sample_config.vocab_size, (1, sample_config.context_length + 5))
        max_new_tokens = 2

        generated = sample_model.generate_text_simple(
            idx=long_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        # Should successfully generate tokens
        assert generated.shape[1] == long_context.shape[1] + max_new_tokens, "Generated length should account for context and new tokens"

    def test_generate_text_softmax(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_softmax method.
        """
        sample_model.eval()
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        generated = sample_model.generate_text_softmax(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        # Verify output shape
        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), "Generated output shape should match expected dimensions"

        # Verify initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved in generated output"

        # Verify new tokens are valid
        new_tokens = generated[:, initial_context.shape[1]:]
        assert (new_tokens >= 0).all() and (new_tokens < sample_config.vocab_size).all(), "New tokens should be within valid vocabulary range"

    def test_generate_text_simple_vs_softmax(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that generate_text_simple and generate_text_softmax produce identical results (greedy sampling).
        """
        sample_model.eval()
        torch.manual_seed(123)
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 3

        # Generate with simple method
        torch.manual_seed(456)
        generated_simple = sample_model.generate_text_simple(
            idx=initial_context.clone(),
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        # Generate with softmax method
        torch.manual_seed(456)
        generated_softmax = sample_model.generate_text_softmax(
            idx=initial_context.clone(),
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length
        )

        # Both should produce identical results for greedy sampling
        assert torch.equal(generated_simple, generated_softmax), "generate_text_simple and generate_text_softmax should produce identical results for greedy sampling"

    def test_generate_text_simple_cached_with_cache(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_simple_cached with cache enabled.
        """
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        generated = sample_model.generate_text_simple_cached(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length,
            use_cache=True
        )

        # Verify output shape
        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), "Generated output shape should match expected dimensions"

        # Verify initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved in generated output"

        # Verify new tokens are valid
        new_tokens = generated[:, initial_context.shape[1]:]
        assert (new_tokens >= 0).all() and (new_tokens < sample_config.vocab_size).all(), "New tokens should be within valid vocabulary range"
    def test_generate_text_simple_cached_without_cache(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_simple_cached with cache disabled.
        """
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        generated = sample_model.generate_text_simple_cached(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length,
            use_cache=False
        )

        # Verify output shape
        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), "Generated output shape should match expected dimensions"

        # Verify initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved in generated output"
    def test_generate_text_cached_vs_uncached(self, sample_config: GptConfig) -> None:
        """
        Test that cached and uncached generation produce identical results.
        """
        torch.manual_seed(789)
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 4

        # Generate with cache
        torch.manual_seed(999)
        model1 = GptModel(sample_config)
        generated_cached = model1.generate_text_simple_cached(
            idx=initial_context.clone(),
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length,
            use_cache=True
        )

        # Generate without cache
        torch.manual_seed(999)
        model2 = GptModel(sample_config)
        generated_uncached = model2.generate_text_simple_cached(
            idx=initial_context.clone(),
            max_new_tokens=max_new_tokens,
            context_size=sample_config.context_length,
            use_cache=False
        )

        # Both should produce identical results
        assert torch.equal(generated_cached, generated_uncached), "Cached and uncached generation should produce identical results"

    def test_generate_text_simple_cached_default_context(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test generate_text_simple_cached with None context_size (uses pos_emb.num_embeddings).
        """
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 3

        generated = sample_model.generate_text_simple_cached(
            idx=initial_context,
            max_new_tokens=max_new_tokens,
            context_size=None,
            use_cache=True
        )

        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), "Generated output shape should match expected dimensions"

    def test_parameter_counting(self) -> None:
        """
        Test parameter counting for standard GPT configurations.
        """
        # Test GPT-124M
        model_124m = GptModel(GPT_CONFIG_124M)
        total_params_124m = sum(p.numel() for p in model_124m.parameters())
        expected_params_124m = 163009536
        assert total_params_124m == expected_params_124m, f"Expected {expected_params_124m} parameters, got {total_params_124m}"

        # Test parameter count without output head (simulates weight tying)
        params_without_head = total_params_124m - sum(p.numel() for p in model_124m.out_head.parameters())
        expected_without_head = 124412160
        assert params_without_head == expected_without_head, f"Expected {expected_without_head} parameters without output head, got {params_without_head}"

        # Test GPT-355M
        model_355m = GptModel(GPT_CONFIG_355M)
        total_params_355m = sum(p.numel() for p in model_355m.parameters())
        assert total_params_355m > total_params_124m, "355M model should have more parameters than 124M"

    def test_gradient_flow(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that gradients flow properly through all parameters.
        """
        sample_model.train()
        input_ids = torch.randint(0, sample_config.vocab_size, (2, 5))
        target = torch.randint(0, sample_config.vocab_size, (2, 5))

        # Forward pass
        output = sample_model(input_ids)

        # Compute loss and backward pass
        loss = nn.CrossEntropyLoss()(output.view(-1, sample_config.vocab_size), target.view(-1))
        loss.backward()

        # Verify gradients exist and are valid for all parameters
        for name, param in sample_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"

    def test_train_eval_modes(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that train and eval modes affect model behavior (dropout).
        """
        torch.manual_seed(100)
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        # Training mode - run multiple times
        sample_model.train()
        outputs_train = []
        for _ in range(3):
            with torch.no_grad():
                output = sample_model(input_ids)
                outputs_train.append(output)

        # At least some outputs should differ due to dropout
        all_same = all(torch.equal(outputs_train[0], out) for out in outputs_train[1:])
        assert not all_same, "Training mode outputs should vary due to dropout"

        # Eval mode - run multiple times
        sample_model.eval()
        outputs_eval = []
        for _ in range(3):
            with torch.no_grad():
                output = sample_model(input_ids)
                outputs_eval.append(output)

        # All outputs should be identical in eval mode
        all_same = all(torch.equal(outputs_eval[0], out) for out in outputs_eval[1:])
        assert all_same, "Eval mode outputs should be deterministic"

    def test_model_state_dict_serialization(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test that model state can be saved and loaded correctly.
        """
        # Save state dict
        original_state_dict = sample_model.state_dict()

        # Create new model and load state
        new_model = GptModel(sample_config)
        new_model.load_state_dict(original_state_dict)

        # Verify outputs match
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))
        sample_model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = sample_model(input_ids)
            output2 = new_model(input_ids)

        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6, msg="Outputs should match after loading state dict")

    def test_reproducibility(self, sample_config: GptConfig) -> None:
        """
        Test that model outputs are reproducible with same seed.
        """
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        # First run
        torch.manual_seed(42)
        model1 = GptModel(sample_config)
        model1.eval()
        with torch.no_grad():
            output1 = model1(input_ids)

        # Second run with same seed
        torch.manual_seed(42)
        model2 = GptModel(sample_config)
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_ids)

        assert torch.equal(output1, output2), "Outputs should be identical for same seed"

    def test_device_compatibility(self, sample_model: GptModel, sample_config: GptConfig) -> None:
        """
        Test model works on CPU and CUDA (if available).
        """
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))
        output_cpu = sample_model(input_ids)
        assert output_cpu.device == torch.device('cpu'), "Output should be on CPU"

        if torch.cuda.is_available():
            sample_model = sample_model.cuda()
            input_ids = input_ids.cuda()
            output_cuda = sample_model(input_ids)
            assert output_cuda.device.type == 'cuda', "Output should be on CUDA device"

    def test_integration_with_tokenizer(self) -> None:
        """
        Test GPT model with actual tokenizer for end-to-end generation.
        """
        pytest.importorskip("tiktoken", reason="tiktoken library required")

        torch.manual_seed(123)
        model = GptModel(GPT_CONFIG_124M)
        model.eval()

        start_context = "Hello, I am"
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        # Generate text
        generated = model.generate_text_simple(
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M.context_length
        )

        # Verify output
        assert generated.shape == (1, len(encoded) + 10), "Generated output shape should match expected dimensions"
        assert torch.isfinite(generated).all(), "Generated output should contain finite values"
        assert (generated >= 0).all() and (generated < GPT_CONFIG_124M.vocab_size).all(), "Generated tokens should be within valid vocabulary range"
        # Decode and verify
        decoded_text = tokenizer.decode(generated.squeeze(0).tolist())
        assert isinstance(decoded_text, str), "Decoded text should be a string"
        assert start_context in decoded_text, "Decoded text should contain the start context"
        assert len(decoded_text) > len(start_context), "Decoded text should be longer than the start context"

        # Verify deterministic generation
        torch.manual_seed(123)
        model2 = GptModel(GPT_CONFIG_124M)
        model2.eval()
        generated2 = model2.generate_text_simple(
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M.context_length
        )
        assert torch.equal(generated, generated2), "Deterministic generation should produce identical outputs"
