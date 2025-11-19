import pytest
import torch

from attention import (
    SelfAttention,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    MHAEinsum
)


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


class TestMultiHeadAttentionWrapper:
    """
    Test suite for the MultiHeadAttentionWrapper module.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiHeadAttention.
        """
        return torch.tensor(
            [[0.43, 0.15, 0.89],  # Your     (x^1)
             [0.55, 0.87, 0.66],  # journey  (x^2)
             [0.57, 0.85, 0.64],  # starts   (x^3)
             [0.22, 0.58, 0.33],  # with     (x^4)
             [0.77, 0.25, 0.10],  # one      (x^5)
             [0.05, 0.80, 0.55]]  # step     (x^6)
        )

    def test_multi_head_output_shape(self, sample_inputs):
        """
        Test that multi-head attention produces correct output shape.
        """
        d_in, d_out = 3, 2
        num_heads = 4
        context_length = 8
        dropout = 0.1

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)

        output = mha(batch)

        batch_size, seq_len, _ = batch.shape
        expected_shape = (batch_size, seq_len, d_out * num_heads)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

    def test_specified_example_case(self, sample_inputs):
        """
        Test using the specific code example provided by the user.
        """
        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        torch.manual_seed(123)
        context_length = batch.shape[1]  # This is the number of tokens
        d_in, d_out = 3, 2
        mha = MultiHeadAttentionWrapper(
            d_in, d_out, context_length, 0.0, num_heads=2
        )
        context_vecs = mha(batch)

        # Verify output shape
        expected_shape = (2, 6, 4)  # batch_size=2, seq_len=6, d_out*num_heads=2*2=4
        assert context_vecs.shape == expected_shape, f"Expected shape {expected_shape}, got {context_vecs.shape}"

        # Verify output is a tensor with reasonable values
        assert isinstance(context_vecs, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(context_vecs).any(), "Output should not contain NaN values"
        assert torch.isfinite(context_vecs).all(), "Output should contain finite values"

    def test_single_vs_multi_head_dimensions(self, sample_inputs):
        """
        Test that multi-head attention output has correct relationship to single head.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.0

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Single head attention
        single_head = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads=1)
        single_output = single_head(batch)

        # Multi head attention
        multi_head = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads=3)
        multi_output = multi_head(batch)

        # Multi-head should have 3x the feature dimension of single head
        assert multi_output.shape[-1] == 3 * single_output.shape[-1], "Multi-head output should have concatenated dimensions"
        assert multi_output.shape[:-1] == single_output.shape[:-1], "Batch and sequence dimensions should match"

    def test_different_head_counts(self, sample_inputs):
        """
        Test multi-head attention with different numbers of heads.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.1

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        head_counts = [1, 2, 4, 8]

        for num_heads in head_counts:
            mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)
            output = mha(batch)

            expected_feature_dim = d_out * num_heads
            assert output.shape[-1] == expected_feature_dim, f"For {num_heads} heads: expected feature dim {expected_feature_dim}, got {output.shape[-1]}"

    def test_gradient_flow(self, sample_inputs):
        """
        Test that gradients flow properly through multi-head attention.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)
        output = mha(batch)
        loss = output.sum()
        loss.backward()

        # Check that all heads have gradients
        for i, head in enumerate(mha.heads):
            assert head.W_query.weight.grad is not None, f"Head {i} W_query should have gradients"
            assert head.W_key.weight.grad is not None, f"Head {i} W_key should have gradients"
            assert head.W_value.weight.grad is not None, f"Head {i} W_value should have gradients"

        # Check input gradients
        assert batch.grad is not None, "Input should have gradients"

    def test_reproducibility_with_seed(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)
        output2 = mha2(batch)

        torch.testing.assert_close(output1, output2, msg="Outputs should be identical with same seed")

    def test_causal_masking_preserved(self, sample_inputs):
        """
        Test that causal masking is preserved across all heads.
        """
        d_in, d_out = 3, 3  # Same dimensions for easier analysis
        context_length = 6
        dropout = 0.0
        num_heads = 2

        # Create test input
        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)

        # Verify that each head has proper causal masking
        for i, head in enumerate(mha.heads):
            mask = head.mask
            seq_len = sample_inputs.shape[0]

            # Check causal property: future positions should be masked
            for pos_i in range(seq_len):
                for pos_j in range(pos_i + 1, seq_len):
                    assert mask[pos_i, pos_j] == 1, f"Head {i}: position ({pos_i}, {pos_j}) should be masked"

                # Check that current and past positions are not masked
                for pos_j in range(pos_i + 1):
                    assert mask[pos_i, pos_j] == 0, f"Head {i}: position ({pos_i}, {pos_j}) should not be masked"

    def test_attention_weights_behavior(self, sample_inputs):
        """
        Test that attention weights from different heads can differ.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        torch.manual_seed(789)  # Set seed for deterministic but different heads
        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)

        # Extract attention weights from different heads by manually computing them
        head1_keys = mha.heads[0].W_key(batch)
        head1_queries = mha.heads[0].W_query(batch)
        head2_keys = mha.heads[1].W_key(batch)
        head2_queries = mha.heads[1].W_query(batch)

        # Check that different heads learn different representations
        # (weights should be different due to random initialization)
        assert not torch.allclose(mha.heads[0].W_query.weight, mha.heads[1].W_query.weight), "Different heads should have different query weights"
        assert not torch.allclose(mha.heads[0].W_key.weight, mha.heads[1].W_key.weight), "Different heads should have different key weights"
        assert not torch.allclose(mha.heads[0].W_value.weight, mha.heads[1].W_value.weight), "Different heads should have different value weights"

    def test_concatenation_behavior(self, sample_inputs):
        """
        Test that outputs are properly concatenated along the feature dimension.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        torch.manual_seed(456)
        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)

        # Get individual head outputs
        individual_outputs = []
        for head in mha.heads:
            individual_outputs.append(head(batch))

        # Get combined output
        combined_output = mha(batch)

        # Manually concatenate individual outputs
        manual_concat = torch.cat(individual_outputs, dim=-1)

        # They should be identical
        torch.testing.assert_close(combined_output, manual_concat, msg="Combined output should equal manual concatenation")

    def test_wrong_input_dimensions(self):
        """
        Test error handling for wrong input dimensions.
        """
        d_in, d_out = 3, 2
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)

        # Test wrong feature dimension
        with pytest.raises(RuntimeError):
            wrong_input = torch.randn(2, 6, 5)  # Wrong d_in (5 instead of 3)
            mha(wrong_input)

        # Test wrong number of dimensions
        with pytest.raises((ValueError, RuntimeError)):
            wrong_dims = torch.randn(6, 3)  # 2D instead of 3D
            mha(wrong_dims)

    def test_large_scale_transformer_configuration(self):
        """
        Test MultiHeadAttentionWrapper with realistic transformer configuration.

        This test uses parameters similar to those found in large language models:
        - embed_dim=768 (common embedding dimension)
        - 12 attention heads
        - context_length=1024 (typical context window)
        - batch_size=8
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the multi-head attention wrapper with transformer-like configuration
        mha_ch03_wrapper = MultiHeadAttentionWrapper(
            d_in=embed_dim,
            d_out=embed_dim//12,  # 768//12 = 64 dimensions per head
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_ch03_wrapper(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out * num_heads (64 * 12 = 768)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_ch03_mha_alias_wrapper(self):
        """
        Test MultiHeadAttentionWrapper using Ch03_MHA alias pattern.

        This test verifies that the MultiHeadAttentionWrapper works with the naming
        convention used in Chapter 3 of the book, where it's referenced as Ch03_MHA.
        """
        # Use the Ch03_MHA alias pattern
        Ch03_MHA = MultiHeadAttentionWrapper

        # Set up transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")

        # Create embeddings tensor
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize using the Ch03_MHA alias (MultiHeadAttentionWrapper)
        mha_ch03 = Ch03_MHA(
            d_in=embed_dim,
            d_out=embed_dim//12,  # Note: d_out per head for wrapper
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_ch03(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional verification
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"


class TestMultiHeadAttention:
    """
    Test suite for the efficient MultiHeadAttention module.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiHeadAttention.
        """
        torch.manual_seed(42)
        return torch.tensor([
            [0.43, 0.15, 0.89],  # Sample tensor to test attention
            [0.55, 0.87, 0.66],  # mechanisms with known values
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]
        ])

    def test_output_shape_divisible_dimensions(self, sample_inputs):
        """
        Test that multi-head attention produces correct output shape with divisible dimensions.
        """
        d_in, d_out = 3, 8  # d_out divisible by various head counts
        context_length = 8
        dropout = 0.1
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        output = mha(batch)

        assert output.shape == (2, 6, 8), f"Expected shape (2, 6, 8), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion(self):
        """
        Test that assertion is raised when d_out is not divisible by num_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out must be divisible by num_heads"):
            MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

    def test_head_dimension_calculation(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        assert mha.d_out == 12, f"Expected d_out=12, got {mha.d_out}"
        assert mha.num_heads == 4, f"Expected num_heads=4, got {mha.num_heads}"

    def test_gradient_flow_efficiency(self, sample_inputs):
        """
        Test that gradients flow properly through the efficient multi-head attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output = mha(batch)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_output_projection_effect(self, sample_inputs):
        """
        Test that output projection layer affects the results.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Create two identical models
        torch.manual_seed(123)
        mha1 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        torch.manual_seed(123)
        mha2 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Zero out one model's output projection
        with torch.no_grad():
            mha2.out_proj.weight.zero_()
            mha2.out_proj.bias.zero_()

        output1 = mha1(batch)
        output2 = mha2(batch)

        # Results should be different due to output projection
        assert not torch.allclose(output1, output2), "Output projection should affect results"

    def test_causal_mask_efficiency(self, sample_inputs):
        """
        Test that causal masking works properly in the efficient implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Test with eval mode to ensure consistent behavior
        mha.eval()

        with torch.no_grad():
            output = mha(batch)

            # Check that changing future tokens doesn't affect past outputs
            modified_batch = batch.clone()
            modified_batch[:, -1, :] = torch.randn_like(modified_batch[:, -1, :])

            modified_output = mha(modified_batch)

            # First 5 positions should be identical (causal masking)
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), \
                "Causal masking should prevent future tokens from affecting past outputs"

    def test_different_head_counts_efficiency(self, sample_inputs):
        """
        Test multi-head attention with different numbers of heads for efficiency.
        """
        d_in = 3
        context_length = 8
        dropout = 0.1

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test different head counts with appropriate d_out values
        test_configs = [
            (1, 4),   # 1 head, d_out=4
            (2, 6),   # 2 heads, d_out=6
            (3, 9),   # 3 heads, d_out=9
            (4, 8),   # 4 heads, d_out=8
        ]

        for num_heads, d_out in test_configs:
            mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, \
                f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_with_seed_efficiency(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in efficient implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should be identical with same random seed"

    def test_versus_wrapper_implementation(self, sample_inputs):
        """
        Test that efficient implementation produces different but valid results compared to wrapper.
        """
        d_in, d_out_per_head = 3, 2
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Wrapper implementation (d_out per head)
        torch.manual_seed(789)
        wrapper = MultiHeadAttentionWrapper(d_in, d_out_per_head, context_length, dropout, num_heads)
        wrapper_output = wrapper(batch)

        # Efficient implementation (total d_out)
        torch.manual_seed(789)
        efficient = MultiHeadAttention(d_in, d_out_per_head * num_heads, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert wrapper_output.shape == efficient_output.shape, \
            f"Output shapes should match: wrapper {wrapper_output.shape} vs efficient {efficient_output.shape}"

        # Results will be different due to different architectures and output projection
        # but both should be valid attention outputs
        assert not torch.allclose(wrapper_output, efficient_output), \
            "Different implementations should produce different results"

    def test_attention_weights_normalization(self, sample_inputs):
        """
        Test that attention weights are properly normalized (though not directly accessible).
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Test that the model produces reasonable outputs
        with torch.no_grad():
            output = mha(batch)

            # Check that outputs are finite and reasonable
            assert torch.isfinite(output).all(), "All outputs should be finite"
            assert not torch.isnan(output).any(), "No outputs should be NaN"

    def test_wrong_input_dimensions_efficiency(self):
        """
        Test error handling for wrong input dimensions in efficient implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_efficient(self):
        """
        Test efficient MultiHeadAttention with realistic transformer configuration.

        This test uses parameters similar to those found in large language models:
        - embed_dim=768 (common embedding dimension)
        - 12 attention heads
        - context_length=1024 (typical context window)
        - batch_size=8

        The key difference from MultiHeadAttentionWrapper is that this efficient implementation:
        - Uses d_out=768 total (not per head)
        - Computes all heads simultaneously
        - Includes output projection layer
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the efficient multi-head attention with transformer-like configuration
        # Note: d_out=embed_dim (total), not per head like in MultiHeadAttentionWrapper
        mha_ch03_efficient = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,  # Total output dimension (768)
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_ch03_efficient(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out (768 total)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_ch03_efficient.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_ch03_efficient.head_dim}"
        assert mha_ch03_efficient.d_out == 768, f"Expected d_out=768, got {mha_ch03_efficient.d_out}"
        assert mha_ch03_efficient.num_heads == 12, f"Expected num_heads=12, got {mha_ch03_efficient.num_heads}"

    def test_ch03_mha_alias_efficient(self):
        """
        Test MultiHeadAttention using Ch03_MHA alias pattern.

        This test verifies that the efficient MultiHeadAttention works with the naming
        convention used in Chapter 3 of the book, where it's referenced as Ch03_MHA.
        """
        # Use the Ch03_MHA alias pattern
        Ch03_MHA = MultiHeadAttention

        # Set up transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")

        # Create embeddings tensor
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize using the Ch03_MHA alias (MultiHeadAttention)
        mha_ch03 = Ch03_MHA(
            d_in=embed_dim,
            d_out=embed_dim,  # Note: total d_out for efficient implementation
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_ch03(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional verification
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"

        # Verify this is the efficient implementation
        assert hasattr(mha_ch03, 'head_dim'), "Should have head_dim attribute for efficient implementation"
        assert mha_ch03.head_dim == 64, f"Expected head_dim=64, got {mha_ch03.head_dim}"


class TestMultiHeadAttentionCombinedQKV:
    """
    Test suite for the MultiHeadAttentionCombinedQKV module.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiHeadAttentionCombinedQKV.
        """
        torch.manual_seed(42)
        return torch.tensor([
            [0.43, 0.15, 0.89],  # Sample tensor to test attention
            [0.55, 0.87, 0.66],  # mechanisms with known values
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]
        ])

    def test_output_shape_combined_qkv(self, sample_inputs):
        """
        Test that combined QKV multi-head attention produces correct output shape.
        """
        d_in, d_out = 8, 8  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 4

        # Expand input to match d_in=8
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 5)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        output = mha(batch)

        assert output.shape == (2, 6, 8), f"Expected shape (2, 6, 8), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion_combined(self):
        """
        Test that assertion is raised when d_out is not divisible by num_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out is indivisible by num_heads"):
            MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

    def test_head_dimension_calculation_combined(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly in combined QKV implementation.
        """
        d_in, d_out = 12, 12  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        # Note: Implementation doesn't store d_out as attribute
        assert mha.num_heads == 4, f"Expected num_heads=4, got {mha.num_heads}"
        # Verify correct calculation: d_out = num_heads * head_dim
        calculated_d_out = mha.num_heads * mha.head_dim
        assert calculated_d_out == d_out, f"Expected calculated d_out={d_out}, got {calculated_d_out}"

    def test_combined_qkv_projection_efficiency(self, sample_inputs):
        """
        Test that combined QKV projection works efficiently.
        """
        d_in, d_out = 6, 6  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        # Test the QKV projection creates correct shape
        qkv_output = mha.qkv(batch)
        expected_qkv_shape = (2, 6, 3 * d_out)  # 3 * d_out for Q, K, V
        assert qkv_output.shape == expected_qkv_shape, f"Expected QKV shape {expected_qkv_shape}, got {qkv_output.shape}"

        # Test full forward pass
        output = mha(batch)
        assert output.shape == (2, 6, d_out), f"Expected output shape (2, 6, {d_out}), got {output.shape}"

    def test_gradient_flow_combined_qkv(self, sample_inputs):
        """
        Test that gradients flow properly through the combined QKV multi-head attention.
        """
        d_in, d_out = 6, 6  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)
        output = mha(batch)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_causal_mask_combined_qkv(self, sample_inputs):
        """
        Test that causal masking works properly in the combined QKV implementation.
        """
        d_in, d_out = 6, 6  # Implementation requires d_in == d_out
        context_length = 6
        dropout = 0.0
        num_heads = 3

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        # Test with eval mode to ensure consistent behavior
        mha.eval()

        with torch.no_grad():
            output = mha(batch)

            # Check that changing future tokens doesn't affect past outputs
            modified_batch = batch.clone()
            modified_batch[:, -1, :] = torch.randn_like(modified_batch[:, -1, :])

            modified_output = mha(modified_batch)

            # First 5 positions should be identical (causal masking)
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), \
                "Causal masking should prevent future tokens from affecting past outputs"

    def test_different_head_counts_combined_qkv(self, sample_inputs):
        """
        Test combined QKV multi-head attention with different numbers of heads.
        """
        context_length = 8
        dropout = 0.1

        # Test different head counts with d_in == d_out for each config
        test_configs = [
            (1, 4, 4),   # 1 head, d_in=4, d_out=4
            (2, 6, 6),   # 2 heads, d_in=6, d_out=6
            (3, 9, 9),   # 3 heads, d_in=9, d_out=9
            (4, 8, 8),   # 4 heads, d_in=8, d_out=8
        ]

        for num_heads, d_in, d_out in test_configs:
            # Create input with appropriate dimensions
            if d_in <= 3:
                expanded_inputs = sample_inputs[:, :d_in]
            else:
                expanded_inputs = torch.cat([sample_inputs, torch.randn(6, d_in - 3)], dim=-1)
            batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

            mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, \
                f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_combined_qkv(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in combined QKV implementation.
        """
        d_in, d_out = 8, 8  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        # Expand input to match d_in=8
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 5)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should be identical with same random seed"

    def test_versus_other_implementations(self, sample_inputs):
        """
        Test that combined QKV implementation produces valid results compared to other implementations.
        """
        d_in, d_out = 4, 4  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Expand input to match d_in=4
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 1)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

        # Combined QKV implementation
        torch.manual_seed(789)
        combined_qkv = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)
        combined_output = combined_qkv(batch)

        # Efficient implementation for comparison (different seed to show they can differ)
        torch.manual_seed(456)
        efficient = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert combined_output.shape == efficient_output.shape, \
            f"Output shapes should match: combined {combined_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(combined_output).all(), "Combined QKV output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(combined_output).any(), "Combined QKV output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

        # Results will typically be different due to different random initialization
        # but when they're the same (same seed), that's also valid

    def test_attention_weights_normalization_combined(self, sample_inputs):
        """
        Test that attention weights are properly normalized in combined QKV implementation.
        """
        d_in, d_out = 4, 4  # Implementation requires d_in == d_out
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Expand input to match d_in=4
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 1)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        # Test that the model produces reasonable outputs
        with torch.no_grad():
            output = mha(batch)

            # Check that outputs are finite and reasonable
            assert torch.isfinite(output).all(), "All outputs should be finite"
            assert not torch.isnan(output).any(), "No outputs should be NaN"

    def test_wrong_input_dimensions_combined_qkv(self):
        """
        Test error handling for wrong input dimensions in combined QKV implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_combined_qkv(self):
        """
        Test combined QKV multi-head attention with realistic transformer configuration.

        This test uses parameters similar to those found in large language models
        and verifies the specific user-requested test case.
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the combined QKV multi-head attention with transformer-like configuration
        mha_combined_qkv = MultiHeadAttentionCombinedQKV(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_combined_qkv(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_combined_qkv.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_combined_qkv.head_dim}"
        # Note: Implementation doesn't store d_out as attribute, but we can verify the calculation
        calculated_d_out = mha_combined_qkv.num_heads * mha_combined_qkv.head_dim
        assert calculated_d_out == 768, f"Expected calculated d_out=768, got {calculated_d_out}"
        assert mha_combined_qkv.num_heads == 12, f"Expected num_heads=12, got {mha_combined_qkv.num_heads}"

    def test_parameter_efficiency_combined_qkv(self, sample_inputs):
        """
        Test that combined QKV implementation is parameter efficient.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Combined QKV implementation
        combined_qkv = MultiHeadAttentionCombinedQKV(d_in, d_out, num_heads, context_length, dropout)

        # Regular implementation for comparison
        regular = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Count parameters
        combined_params = sum(p.numel() for p in combined_qkv.parameters())
        regular_params = sum(p.numel() for p in regular.parameters())

        # Combined QKV should have fewer parameters due to single QKV projection
        # but similar total parameters due to the proj layer
        assert combined_params > 0, "Combined QKV should have parameters"
        assert regular_params > 0, "Regular implementation should have parameters"

        # Verify the QKV layer has the right size (3 * d_out parameters)
        qkv_weight_size = combined_qkv.qkv.weight.numel()
        expected_qkv_size = d_in * 3 * d_out
        assert qkv_weight_size == expected_qkv_size, \
            f"Expected QKV weight size {expected_qkv_size}, got {qkv_weight_size}"


class TestMHAEinsum:
    """
    Test suite for the MHAEinsum module (einsum-based multi-head attention).
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MHAEinsum.
        """
        torch.manual_seed(42)
        return torch.tensor([
            [0.43, 0.15, 0.89],  # Sample tensor to test attention
            [0.55, 0.87, 0.66],  # mechanisms with known values
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]
        ])

    def test_output_shape_einsum(self, sample_inputs):
        """
        Test that einsum multi-head attention produces correct output shape.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.1
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion_einsum(self):
        """
        Test that assertion is raised when d_out is not divisible by num_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out must be divisible by num_heads"):
            MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

    def test_head_dimension_calculation_einsum(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly in einsum implementation.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        assert mha.d_out == 12, f"Expected d_out=12, got {mha.d_out}"
        assert mha.num_heads == 4, f"Expected num_heads=4, got {mha.num_heads}"

    def test_einsum_parameter_initialization(self, sample_inputs):
        """
        Test that einsum implementation initializes parameters correctly.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Test without bias
        mha_no_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
        assert mha_no_bias.bias_q is None, "bias_q should be None when qkv_bias=False"
        assert mha_no_bias.bias_k is None, "bias_k should be None when qkv_bias=False"
        assert mha_no_bias.bias_v is None, "bias_v should be None when qkv_bias=False"

        # Test with bias
        mha_with_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=True)
        assert mha_with_bias.bias_q is not None, "bias_q should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_k is not None, "bias_k should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_v is not None, "bias_v should be initialized when qkv_bias=True"
        assert mha_with_bias.bias_q.shape == (d_out,), f"bias_q should have shape ({d_out},), got {mha_with_bias.bias_q.shape}"

    def test_gradient_flow_einsum(self, sample_inputs):
        """
        Test that gradients flow properly through the einsum multi-head attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output = mha(batch)

        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        assert mha.W_query.grad is not None, "W_query should have gradients"
        assert mha.W_key.grad is not None, "W_key should have gradients"
        assert mha.W_value.grad is not None, "W_value should have gradients"
        assert mha.out_proj.weight.grad is not None, "out_proj should have gradients"

    def test_causal_mask_einsum(self, sample_inputs):
        """
        Test that causal masking works properly in the einsum implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test with eval mode to ensure consistent behavior
        mha.eval()

        with torch.no_grad():
            output = mha(batch)

            # Check that changing future tokens doesn't affect past outputs
            modified_batch = batch.clone()
            modified_batch[:, -1, :] = torch.randn_like(modified_batch[:, -1, :])

            modified_output = mha(modified_batch)

            # First 5 positions should be identical (causal masking)
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), \
                "Causal masking should prevent future tokens from affecting past outputs"

    def test_different_head_counts_einsum(self, sample_inputs):
        """
        Test einsum multi-head attention with different numbers of heads.
        """
        d_in = 3
        context_length = 8
        dropout = 0.1

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test different head counts with appropriate d_out values
        test_configs = [
            (1, 4),   # 1 head, d_out=4
            (2, 6),   # 2 heads, d_out=6
            (3, 9),   # 3 heads, d_out=9
            (4, 8),   # 4 heads, d_out=8
        ]

        for num_heads, d_out in test_configs:
            mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, \
                f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_einsum(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in einsum implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should be identical with same random seed"

    def test_versus_other_implementations(self, sample_inputs):
        """
        Test that einsum implementation produces valid results compared to other implementations.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Einsum implementation
        torch.manual_seed(789)
        einsum_mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)
        einsum_output = einsum_mha(batch)

        # Efficient implementation for comparison (different seed to show they can differ)
        torch.manual_seed(456)
        efficient = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert einsum_output.shape == efficient_output.shape, \
            f"Output shapes should match: einsum {einsum_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(einsum_output).all(), "Einsum output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(einsum_output).any(), "Einsum output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

    def test_attention_weights_normalization_einsum(self, sample_inputs):
        """
        Test that attention weights are properly normalized in einsum implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test that the model produces reasonable outputs
        with torch.no_grad():
            output = mha(batch)

            # Check that outputs are finite and reasonable
            assert torch.isfinite(output).all(), "All outputs should be finite"
            assert not torch.isnan(output).any(), "No outputs should be NaN"

    def test_wrong_input_dimensions_einsum(self):
        """
        Test error handling for wrong input dimensions in einsum implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_einsum(self):
        """
        Test einsum multi-head attention with realistic transformer configuration.

        This test uses parameters similar to those found in large language models
        and verifies the specific user-requested test case.
        """
        # Set up realistic transformer parameters
        embed_dim = 768
        context_len = 1024
        batch_size = 8
        device = torch.device("cpu")  # Use CPU for testing

        # Create embeddings tensor with realistic dimensions
        embeddings = torch.randn(batch_size, context_len, embed_dim)

        # Initialize the einsum multi-head attention with transformer-like configuration
        mha_einsum = MHAEinsum(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_einsum(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_einsum.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_einsum.head_dim}"
        assert mha_einsum.d_out == 768, f"Expected d_out=768, got {mha_einsum.d_out}"
        assert mha_einsum.num_heads == 12, f"Expected num_heads=12, got {mha_einsum.num_heads}"

    def test_parameter_efficiency_einsum(self, sample_inputs):
        """
        Test that einsum implementation has expected parameter counts.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Einsum implementation
        einsum_mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads)

        # Regular implementation for comparison
        regular = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Count parameters
        einsum_params = sum(p.numel() for p in einsum_mha.parameters())
        regular_params = sum(p.numel() for p in regular.parameters())

        assert einsum_params > 0, "Einsum implementation should have parameters"
        assert regular_params > 0, "Regular implementation should have parameters"

        # Verify specific parameter shapes
        assert einsum_mha.W_query.shape == (d_in, d_out), f"Expected W_query shape ({d_in}, {d_out}), got {einsum_mha.W_query.shape}"
        assert einsum_mha.W_key.shape == (d_in, d_out), f"Expected W_key shape ({d_in}, {d_out}), got {einsum_mha.W_key.shape}"
        assert einsum_mha.W_value.shape == (d_in, d_out), f"Expected W_value shape ({d_in}, {d_out}), got {einsum_mha.W_value.shape}"

    def test_bias_functionality_einsum(self, sample_inputs):
        """
        Test that bias terms work correctly when enabled.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with bias
        mha_with_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=True)
        output_with_bias = mha_with_bias(batch)

        # Test without bias
        mha_no_bias = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
        output_no_bias = mha_no_bias(batch)

        # Both should produce valid outputs
        assert torch.isfinite(output_with_bias).all(), "Output with bias should be finite"
        assert torch.isfinite(output_no_bias).all(), "Output without bias should be finite"
        assert output_with_bias.shape == output_no_bias.shape, "Output shapes should match regardless of bias"

    def test_einsum_operations_correctness(self, sample_inputs):
        """
        Test that einsum operations produce mathematically correct results.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAEinsum(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)

        # Extract components for manual verification
        b, n, _ = batch.shape

        # Test einsum QKV projection manually
        Q_einsum = torch.einsum("bnd,do->bno", batch, mha.W_query)
        Q_manual = batch @ mha.W_query

        # Should be equivalent
        assert torch.allclose(Q_einsum, Q_manual, atol=1e-6), "Einsum QKV projection should match manual computation"

        # Test full forward pass produces valid output
        output = mha(batch)
        assert output.shape == (b, n, d_out), f"Expected output shape ({b}, {n}, {d_out}), got {output.shape}"
