import torch

from attention import SelfAttention, CausalAttention


class TestCausalAttention:
    """
    Test suite for the CausalAttention module.
    """

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
        masked_scores.masked_fill_(ca.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

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
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6, msg="Causal attention weights should sum to 1 for each query position")

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
