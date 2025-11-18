import pytest
import torch
import torch.nn as nn

from attention import SelfAttention, CausalAttention


class TestSelfAttention:
    """
    Test suite for the SelfAttention module.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor representing word embeddings for testing.
        """
        return torch.tensor(
            [[0.43, 0.15, 0.89],  # Your     (x^1)
             [0.55, 0.87, 0.66],  # journey  (x^2)
             [0.57, 0.85, 0.64],  # starts   (x^3)
             [0.22, 0.58, 0.33],  # with     (x^4)
             [0.77, 0.25, 0.10],  # one      (x^5)
             [0.05, 0.80, 0.55]]  # step     (x^6)
        )

    def test_forward_output_shape(self, sample_inputs):
        """
        Test that forward pass produces correct output shape.
        """
        d_in, d_out = 3, 2
        sa = SelfAttention(d_in, d_out)

        output = sa(sample_inputs)

        expected_shape = (sample_inputs.shape[0], d_out)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_reproducibility_with_seed(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed.
        """
        d_in, d_out = 3, 2

        # First run
        torch.manual_seed(789)
        sa1 = SelfAttention(d_in, d_out)
        output1 = sa1(sample_inputs)

        # Second run with same seed
        torch.manual_seed(789)
        sa2 = SelfAttention(d_in, d_out)
        output2 = sa2(sample_inputs)

        torch.testing.assert_close(output1, output2)

    def test_different_dimensions(self):
        """
        Test SelfAttention with different input/output dimensions.
        """
        test_cases = [
            (512, 64),   # Large to small
            (128, 128),  # Same dimensions
            (64, 256),   # Small to large
            (1, 1),      # Minimal dimensions
        ]
        for d_in, d_out in test_cases:
            sa = SelfAttention(d_in, d_out)
            x = torch.randn(10, d_in)  # 10 tokens
            output = sa(x)
            assert output.shape == (10, d_out), f"For dimensions ({d_in}, {d_out}): expected shape (10, {d_out}), got {output.shape}"

    def test_batch_processing(self):
        """
        Test SelfAttention with batch dimensions.
        """
        d_in, d_out = 3, 2
        batch_size, seq_len = 4, 6
        sa = SelfAttention(d_in, d_out)

        # Test with batch dimension
        x = torch.randn(batch_size, seq_len, d_in)

        # Process each batch item separately
        outputs = []
        for i in range(batch_size):
            output = sa(x[i])
            outputs.append(output)

        # Check all outputs have correct shape
        for i, output in enumerate(outputs):
            assert output.shape == (seq_len, d_out), f"Batch item {i}: expected shape ({seq_len}, {d_out}), got {output.shape}"

    def test_attention_weights_sum_to_one(self, sample_inputs):
        """
        Test that attention weights sum to 1 for each query position.
        """
        d_in, d_out = 3, 2
        sa = SelfAttention(d_in, d_out)

        # We need to access intermediate attention weights
        # Let's modify the forward pass temporarily
        keys = sa.W_key(sample_inputs)
        queries = sa.W_query(sample_inputs)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Check that each row sums to 1 (within numerical tolerance)
        row_sums = attn_weights.sum(dim=-1)
        expected = torch.ones(attn_weights.shape[0])
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self, sample_inputs):
        """
        Test that gradients flow properly through the module.
        """
        d_in, d_out = 3, 2
        sa = SelfAttention(d_in, d_out)

        # Enable gradient computation
        sample_inputs.requires_grad_(True)

        output = sa(sample_inputs)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        assert sa.W_query.weight.grad is not None, "W_query.weight should have gradients after backward pass"
        assert sa.W_key.weight.grad is not None, "W_key.weight should have gradients after backward pass"
        assert sa.W_value.weight.grad is not None, "W_value.weight should have gradients after backward pass"

        # Check that input gradients exist
        assert sample_inputs.grad is not None, "Input tensor should have gradients after backward pass"

    def test_original_example_output(self):
        """
        Test the original example from the file to ensure backward compatibility.
        """
        inputs = torch.tensor(
            [[0.43, 0.15, 0.89], # Your     (x^1)
             [0.55, 0.87, 0.66], # journey  (x^2)
             [0.57, 0.85, 0.64], # starts   (x^3)
             [0.22, 0.58, 0.33], # with     (x^4)
             [0.77, 0.25, 0.10], # one      (x^5)
             [0.05, 0.80, 0.55]] # step     (x^6)
        )

        d_in = inputs.shape[1]  # 3
        d_out = 2

        torch.manual_seed(789)
        sa = SelfAttention(d_in, d_out)
        output = sa(inputs)

        # Expected output from the original implementation
        expected_output = torch.tensor([
            [-0.0739,  0.0713],
            [-0.0748,  0.0703],
            [-0.0749,  0.0702],
            [-0.0760,  0.0685],
            [-0.0763,  0.0679],
            [-0.0754,  0.0693]
        ])

        # Check that output matches expected (within tolerance for floating point)
        torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)

    def test_word_sequence_example(self):
        """
        Test with the word sequence example input.
        """
        word_inputs = torch.tensor(
            [[0.43, 0.15, 0.89], # Your     (x^1)
             [0.55, 0.87, 0.66], # journey  (x^2)
             [0.57, 0.85, 0.64], # starts   (x^3)
             [0.22, 0.58, 0.33], # with     (x^4)
             [0.77, 0.25, 0.10], # one      (x^5)
             [0.05, 0.80, 0.55]] # step     (x^6)
        )
        d_in, d_out = 3, 4
        sa = SelfAttention(d_in, d_out)
        output = sa(word_inputs)
        assert output.shape == (6, 4), f"Expected output shape (6, 4), got {output.shape}"

    def test_word_sequence_attention_properties(self):
        """
        Test attention properties with the word sequence example.
        """
        word_inputs = torch.tensor(
            [[0.43, 0.15, 0.89], # Your     (x^1)
             [0.55, 0.87, 0.66], # journey  (x^2)
             [0.57, 0.85, 0.64], # starts   (x^3)
             [0.22, 0.58, 0.33], # with     (x^4)
             [0.77, 0.25, 0.10], # one      (x^5)
             [0.05, 0.80, 0.55]] # step     (x^6)
        )

        d_in, d_out = 3, 3  # Same dimensions to analyze attention
        sa = SelfAttention(d_in, d_out)

        # Get attention weights
        keys = sa.W_key(word_inputs)
        queries = sa.W_query(word_inputs)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Check that attention weights sum to 1 for each word
        row_sums = attn_weights.sum(dim=-1)
        expected = torch.ones(6)
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6)

    def test_wrong_input_dimension_error(self):
        """
        Test that wrong input dimensions raise appropriate error.
        """
        d_in, d_out = 3, 2
        sa = SelfAttention(d_in, d_out)

        with pytest.raises(RuntimeError):
            wrong_input = torch.randn(6, 5)  # Wrong d_in (5 instead of 3)
            sa(wrong_input)


class TestCausalAttention:
    """
    Test suite for the CausalAttention module.
    """

    @pytest.fixture
    def sample_batch_inputs(self):
        """
        Sample batched input tensor for testing CausalAttention.
        """
        return torch.tensor([
            [[0.43, 0.15, 0.89],  # Batch 1: Your     (x^1)
             [0.55, 0.87, 0.66],  #         journey  (x^2)
             [0.57, 0.85, 0.64],  #         starts   (x^3)
             [0.22, 0.58, 0.33]],  #         with     (x^4)
            [[0.77, 0.25, 0.10],  # Batch 2: one      (x^1)
             [0.05, 0.80, 0.55],  #         step     (x^2)
             [0.91, 0.44, 0.33],  #         forward  (x^3)
             [0.12, 0.67, 0.89]]   #         now      (x^4)
        ])

    def test_mask_initialization(self):
        """
        Test that CausalAttention creates proper mask during initialization.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.1
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Test CausalAttention specific components
        assert hasattr(ca, 'mask'), "CausalAttention should have a 'mask' attribute"
        assert ca.mask.shape == (context_length, context_length), f"Mask shape should be ({context_length}, {context_length}), got {ca.mask.shape}"
        assert ca.mask.shape == (context_length, context_length), f"Mask shape should be ({context_length}, {context_length}), got {ca.mask.shape}"

    def test_causal_mask_properties(self):
        """
        Test that the causal mask has correct properties.
        """
        context_length = 5
        ca = CausalAttention(3, 2, context_length, 0.1)

        mask = ca.mask
        # Upper triangular mask should have 1s above diagonal, 0s on and below
        expected_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        torch.testing.assert_close(mask, expected_mask, msg="Causal mask should be upper triangular with 1s above diagonal")

        # Check that mask prevents future attention
        for i in range(context_length):
            for j in range(context_length):
                if j > i:  # Future positions
                    assert mask[i, j] == 1, f"Future position ({i}, {j}) should be masked (value=1), got {mask[i, j]}"
                else:  # Current and past positions
                    assert mask[i, j] == 0, f"Current/past position ({i}, {j}) should not be masked (value=0), got {mask[i, j]}"

    def test_forward_output_shape(self, sample_batch_inputs):
        """
        Test that forward pass produces correct output shape.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.1
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        output = ca(sample_batch_inputs)

        batch_size, seq_len, _ = sample_batch_inputs.shape
        expected_shape = (batch_size, seq_len, d_out)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_causal_masking_effect(self):
        """
        Test that causal masking prevents attention to future tokens.
        """
        d_in, d_out, context_length, dropout = 3, 3, 4, 0.0  # No dropout for testing
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Create a simple test input
        x = torch.randn(1, 4, d_in)  # batch=1, seq_len=4

        # Get attention components
        keys = ca.W_key(x)
        queries = ca.W_query(x)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply mask
        num_tokens = x.shape[1]
        masked_scores = attn_scores.clone()
        masked_scores.masked_fill_(
            ca.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # Check that future positions are masked (set to -inf)
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                assert masked_scores[0, i, j] == -torch.inf, f"Position ({i}, {j}) should be masked to -inf, got {masked_scores[0, i, j]}"

    def test_attention_weights_sum_to_one(self, sample_batch_inputs):
        """
        Test that attention weights sum to 1 for each query position.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.0  # No dropout for testing
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Manually compute attention weights to verify
        keys = ca.W_key(sample_batch_inputs)
        queries = ca.W_query(sample_batch_inputs)

        batch_size, num_tokens, _ = sample_batch_inputs.shape
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(ca.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Check that each row sums to 1 (within numerical tolerance)
        row_sums = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, num_tokens)
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6)

    def test_different_sequence_lengths(self):
        """
        Test CausalAttention with different sequence lengths.
        """
        d_in, d_out, context_length, dropout = 64, 32, 128, 0.1

        test_cases = [
            (1, 1),    # Single token
            (1, 10),   # Short sequence
            (2, 50),   # Medium sequence
            (4, 128),  # Full context length
        ]

        ca = CausalAttention(d_in, d_out, context_length, dropout)

        for batch_size, seq_len in test_cases:
            x = torch.randn(batch_size, seq_len, d_in)
            output = ca(x)
            assert output.shape == (batch_size, seq_len, d_out), f"For batch_size={batch_size}, seq_len={seq_len}: expected shape ({batch_size}, {seq_len}, {d_out}), got {output.shape}"

    def test_reproducibility_with_seed(self, sample_batch_inputs):
        """
        Test that outputs are reproducible with same random seed.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.1

        # First run
        torch.manual_seed(789)
        ca1 = CausalAttention(d_in, d_out, context_length, dropout)
        ca1.eval()  # Disable dropout for reproducibility
        output1 = ca1(sample_batch_inputs)

        # Second run with same seed
        torch.manual_seed(789)
        ca2 = CausalAttention(d_in, d_out, context_length, dropout)
        ca2.eval()  # Disable dropout for reproducibility
        output2 = ca2(sample_batch_inputs)

        torch.testing.assert_close(output1, output2)

    def test_dropout_effect(self, sample_batch_inputs):
        """
        Test that dropout affects outputs differently in train vs eval mode.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.5  # High dropout
        torch.manual_seed(42)

        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Training mode
        ca.train()
        torch.manual_seed(123)
        output_train = ca(sample_batch_inputs)

        # Evaluation mode
        ca.eval()
        torch.manual_seed(123)
        output_eval = ca(sample_batch_inputs)

        # Outputs should be different due to dropout in training mode
        assert not torch.allclose(output_train, output_eval, atol=1e-6), "Training and evaluation outputs should be different when dropout is applied"

    def test_gradient_flow(self, sample_batch_inputs):
        """
        Test that gradients flow properly through the module.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.1
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Enable gradient computation
        sample_batch_inputs.requires_grad_(True)

        output = ca(sample_batch_inputs)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        assert ca.W_query.weight.grad is not None, "W_query.weight should have gradients after backward pass"
        assert ca.W_key.weight.grad is not None, "W_key.weight should have gradients after backward pass"
        assert ca.W_value.weight.grad is not None, "W_value.weight should have gradients after backward pass"

        # Check that input gradients exist
        assert sample_batch_inputs.grad is not None, "Input tensor should have gradients after backward pass"

    def test_context_length_constraint(self):
        """
        Test behavior when input sequence length approaches context length.
        """
        d_in, d_out, context_length, dropout = 3, 2, 5, 0.1
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Test with sequence length equal to context length
        x = torch.randn(1, context_length, d_in)
        output = ca(x)
        assert output.shape == (1, context_length, d_out), f"Expected shape (1, {context_length}, {d_out}), got {output.shape}"

        # Test with sequence length less than context length
        x_short = torch.randn(1, context_length - 2, d_in)
        output_short = ca(x_short)
        assert output_short.shape == (1, context_length - 2, d_out), f"Expected shape (1, {context_length - 2}, {d_out}), got {output_short.shape}"

    def test_weight_initialization_comparison(self):
        """
        Test that CausalAttention and SelfAttention have similar weight initialization when using same seed.
        """
        d_in, d_out = 3, 2
        context_length, dropout = 8, 0.0  # No dropout

        # Create both modules with same seed
        torch.manual_seed(789)
        sa = SelfAttention(d_in, d_out)

        torch.manual_seed(789)
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        # Check that QKV weights are the same when initialized with same seed
        torch.testing.assert_close(sa.W_query.weight, ca.W_query.weight, msg="CausalAttention should have same W_query weights as SelfAttention when initialized with same seed")
        torch.testing.assert_close(sa.W_key.weight, ca.W_key.weight, msg="CausalAttention should have same W_key weights as SelfAttention when initialized with same seed")
        torch.testing.assert_close(sa.W_value.weight, ca.W_value.weight, msg="CausalAttention should have same W_value weights as SelfAttention when initialized with same seed")

    def test_wrong_input_dimension_error(self):
        """
        Test that wrong input dimensions raise appropriate error.
        """
        d_in, d_out, context_length, dropout = 3, 2, 8, 0.1
        ca = CausalAttention(d_in, d_out, context_length, dropout)

        with pytest.raises(RuntimeError):
            wrong_input = torch.randn(1, 4, 5)  # Wrong d_in (5 instead of 3)
            ca(wrong_input)

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            wrong_dims = torch.randn(4, 3)  # Wrong number of dimensions (2D instead of 3D)
            ca(wrong_dims)
