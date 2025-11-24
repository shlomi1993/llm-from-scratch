import pytest
import torch
import torch.nn as nn
import tiktoken

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M
from src.gpt import GptModelBasic


class TestGptModelBasic:
    """
    Test suite for the GptModelBasic class.
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
    def sample_model(self, sample_config: GptConfig) -> GptModelBasic:
        """
        Create a GptModelBasic instance for testing.
        """
        torch.manual_seed(42)
        return GptModelBasic(sample_config)

    def test_forward_pass_shape(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that forward pass produces correct output shape.
        """
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        output = sample_model(input_ids)

        expected_shape = (batch_size, seq_len, sample_config.vocab_size)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_different_sequence_lengths(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test model with different sequence lengths.
        """
        batch_size = 2
        test_lengths = [1, 5, 10, sample_config.context_length]

        for seq_len in test_lengths:
            if seq_len <= sample_config.context_length:
                input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
                output = sample_model(input_ids)

                expected_shape = (batch_size, seq_len, sample_config.vocab_size)
                assert output.shape == expected_shape, f"Failed for seq_len={seq_len}: expected {expected_shape}, got {output.shape}"

    def test_generate_text_simple(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test the greedy text generation method.
        """
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        with torch.no_grad():
            generated = sample_model.generate_text(
                idx=initial_context,
                max_new_tokens=max_new_tokens,
                context_size=sample_config.context_length
            )

        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), f"Expected shape (1, {expected_length}), got {generated.shape}"

        # Check that initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved"

        # Check that new tokens are within vocabulary
        new_tokens = generated[:, initial_context.shape[1]:]
        assert (new_tokens >= 0).all() and (new_tokens < sample_config.vocab_size).all(), "Generated tokens should be within vocabulary range"

    def test_generate_text_simple_softmax(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test the softmax-based text generation method.
        """
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 5

        with torch.no_grad():
            generated = sample_model.generate_text(
                idx=initial_context,
                max_new_tokens=max_new_tokens,
                context_size=sample_config.context_length,
                use_softmax=True
            )

        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length), f"Expected shape (1, {expected_length}), got {generated.shape}"

        # Check that initial context is preserved
        assert torch.equal(generated[:, :initial_context.shape[1]], initial_context), "Initial context should be preserved"

    def test_generation_methods_equivalence(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that both generation methods produce same results for greedy sampling.
        """
        torch.manual_seed(123)
        initial_context = torch.randint(0, sample_config.vocab_size, (1, 3))
        max_new_tokens = 3

        with torch.no_grad():
            # Generate with first method
            torch.manual_seed(456)
            generated1 = sample_model.generate_text(
                idx=initial_context.clone(),
                max_new_tokens=max_new_tokens,
                context_size=sample_config.context_length
            )

            # Generate with second method
            torch.manual_seed(456)
            generated2 = sample_model.generate_text(
                idx=initial_context.clone(),
                max_new_tokens=max_new_tokens,
                context_size=sample_config.context_length,
                use_softmax=True
            )

        # Both methods should produce identical results for greedy sampling
        assert torch.equal(generated1, generated2), "Both generation methods should produce identical results"

    def test_context_length_cropping(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that context is properly cropped when it exceeds context_size.
        """
        # Create context longer than context_length
        long_context = torch.randint(0, sample_config.vocab_size, (1, sample_config.context_length + 5))
        max_new_tokens = 2
        context_size = sample_config.context_length

        with torch.no_grad():
            generated = sample_model.generate_text(
                idx=long_context,
                max_new_tokens=max_new_tokens,
                context_size=context_size
            )

        # Should generate new tokens successfully
        assert generated.shape[1] == long_context.shape[1] + max_new_tokens, "Should generate requested number of tokens"

    def test_batch_generation(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test text generation with batch size > 1.
        """
        batch_size = 3
        initial_context = torch.randint(0, sample_config.vocab_size, (batch_size, 4))
        max_new_tokens = 3

        with torch.no_grad():
            generated = sample_model.generate_text(
                idx=initial_context,
                max_new_tokens=max_new_tokens,
                context_size=sample_config.context_length
            )

        expected_shape = (batch_size, initial_context.shape[1] + max_new_tokens)
        assert generated.shape == expected_shape, f"Expected shape {expected_shape}, got {generated.shape}"

        # Check each batch item
        for i in range(batch_size):
            assert torch.equal(generated[i, :initial_context.shape[1]], initial_context[i]), f"Batch item {i} context should be preserved"

    def test_parameter_counting(self) -> None:
        """
        Test parameter counting matches expected values for standard configs.
        """
        # Test with GPT_CONFIG_124M
        model = GptModelBasic(GPT_CONFIG_124M)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected parameters for GPT-2 124M without weight tying
        expected_params = 163009536
        assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"

        # Test parameter count excluding output head (simulating weight tying)
        params_without_output_head = total_params - sum(p.numel() for p in model.out_head.parameters())
        expected_without_head = 124412160
        assert params_without_output_head == expected_without_head, f"Expected {expected_without_head} parameters without output head, got {params_without_output_head}"



    def test_gradient_flow(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that gradients flow properly through the model.
        """
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        output = sample_model(input_ids)

        # Compute loss and backward pass
        target = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        loss = nn.CrossEntropyLoss()(output.view(-1, sample_config.vocab_size), target.view(-1))
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in sample_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} should not have NaN gradients"

    def test_eval_mode(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that eval mode affects model behavior (dropout).
        """
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        # Training mode
        sample_model.train()
        with torch.no_grad():
            output_train = sample_model(input_ids)

        # Eval mode
        sample_model.eval()
        with torch.no_grad():
            output_eval = sample_model(input_ids)

        # Outputs might be different due to dropout, but should have same shape
        assert output_train.shape == output_eval.shape, "Train and eval outputs should have same shape"

    def test_different_configurations(self) -> None:
        """
        Test model with different configurations.
        """
        configs = [GPT_CONFIG_124M, GPT_CONFIG_355M]

        for config in configs:
            model = GptModelBasic(config)
            input_ids = torch.randint(0, config.vocab_size, (1, 10))

            output = model(input_ids)
            expected_shape = (1, 10, config.vocab_size)
            assert output.shape == expected_shape, f"Failed for config with {config.emb_dim} emb_dim: expected {expected_shape}, got {output.shape}"

    def test_device_compatibility(self, sample_model: GptModelBasic, sample_config: GptConfig) -> None:
        """
        Test that model works on different devices.
        """
        # Test CPU (default)
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))
        output_cpu = sample_model(input_ids)
        assert output_cpu.device == torch.device('cpu'), "Output should be on CPU"

        # Test CUDA if available
        if torch.cuda.is_available():
            sample_model = sample_model.cuda()
            input_ids = input_ids.cuda()
            output_cuda = sample_model(input_ids)
            assert output_cuda.device.type == 'cuda', "Output should be on CUDA"

    def test_model_state_dict_serialization(self, sample_model: GptModelBasic) -> None:
        """
        Test that model can be saved and loaded correctly.
        """
        # Save state dict
        original_state_dict = sample_model.state_dict()

        # Create new model instance and load state dict
        sample_config = GptConfig(
            emb_dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=32,
            drop_rate=0.1,
            qkv_bias=False
        )
        new_model = GptModelBasic(sample_config)
        new_model.load_state_dict(original_state_dict)

        # Test that they produce same outputs
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        sample_model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = sample_model(input_ids)
            output2 = new_model(input_ids)

        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6, msg="State dict loaded model output should match original")

    def test_reproducibility(self, sample_config: GptConfig) -> None:
        """
        Test that model outputs are reproducible with same seed.
        """
        input_ids = torch.randint(0, sample_config.vocab_size, (1, 5))

        # First run
        torch.manual_seed(42)
        model1 = GptModelBasic(sample_config)
        model1.eval()
        with torch.no_grad():
            output1 = model1(input_ids)

        # Second run with same seed
        torch.manual_seed(42)
        model2 = GptModelBasic(sample_config)
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_ids)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6), "Model outputs should be reproducible with same seed"

    def test_gpt_demo_integration(self) -> None:
        """
        Test GPT model text generation demonstration (converted from main function).
        """
        pytest.importorskip("tiktoken", reason="tiktoken library required for tokenization")

        torch.manual_seed(123)
        model = GptModelBasic(GPT_CONFIG_124M)
        model.eval()  # disable dropout

        start_context = "Hello, I am"
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        # Verify input processing
        assert isinstance(encoded, list), "Encoded should be a list of token IDs"
        assert len(encoded) > 0, "Encoded should not be empty"
        assert encoded_tensor.shape == (1, len(encoded)), f"Expected shape (1, {len(encoded)}), got {encoded_tensor.shape}"

        # Generate text
        with torch.no_grad():
            out = model.generate_text(idx=encoded_tensor, max_new_tokens=10, context_size=GPT_CONFIG_124M.context_length)

        # Verify output
        assert out.shape == (1, len(encoded) + 10), f"Expected output length {len(encoded) + 10}, got {out.shape[1]}"
        assert torch.isfinite(out).all(), "Generated tokens should be finite"
        assert (out >= 0).all() and (out < GPT_CONFIG_124M.vocab_size).all(), "Generated tokens should be within vocabulary range"

        # Test decoding
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        assert isinstance(decoded_text, str), "Decoded text should be a string"
        assert start_context in decoded_text, "Generated text should contain the original context"
        assert len(decoded_text) > len(start_context), "Generated text should be longer than input context"

        # Verify that generation is deterministic with same seed
        torch.manual_seed(123)
        model2 = GptModelBasic(GPT_CONFIG_124M)
        model2.eval()
        with torch.no_grad():
            out2 = model2.generate_text(idx=encoded_tensor, max_new_tokens=10, context_size=GPT_CONFIG_124M.context_length)

        assert torch.equal(out, out2), "Generation should be deterministic with same seed"
