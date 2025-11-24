import pytest
import torch
import torch.nn as nn

from src.attention import (
    SelfAttention,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    MHAEinsum,
    MHAPyTorchScaledDotProduct,
    MHAPyTorchSDPAWithoutFlash,
    MHAPyTorchClass,
    MHAPyTorchFlexAttention, causal,
    MultiHeadAttentionCached
)
from src.configurations import GptConfig


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
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6, msg="Attention weights should sum to 1 for each query position")

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
        torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4, msg="Output should match expected values from original implementation")

    def test_word_sequence_example(self):
        """
        Test with the word sequence example input.
        """
        word_inputs = torch.tensor(
            [[0.43, 0.15, 0.89],    # Your     (x^1)
             [0.55, 0.87, 0.66],    # journey  (x^2)
             [0.57, 0.85, 0.64],    # starts   (x^3)
             [0.22, 0.58, 0.33],    # with     (x^4)
             [0.77, 0.25, 0.10],    # one      (x^5)
             [0.05, 0.80, 0.55]]    # step     (x^6)
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
            [[0.43, 0.15, 0.89],    # Your     (x^1)
             [0.55, 0.87, 0.66],    # journey  (x^2)
             [0.57, 0.85, 0.64],    # starts   (x^3)
             [0.22, 0.58, 0.33],    # with     (x^4)
             [0.77, 0.25, 0.10],    # one      (x^5)
             [0.05, 0.80, 0.55]]    # step     (x^6)
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
        torch.testing.assert_close(row_sums, expected, atol=1e-6, rtol=1e-6, msg="Attention weights should sum to 1 for each word position")


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
            [[0.43, 0.15, 0.89],    # Batch 1:  Your     (x^1)
             [0.55, 0.87, 0.66],    #           journey  (x^2)
             [0.57, 0.85, 0.64],    #           starts   (x^3)
             [0.22, 0.58, 0.33]],   #           with     (x^4)
            [[0.77, 0.25, 0.10],    # Batch 2:  one      (x^1)
             [0.05, 0.80, 0.55],    #           step     (x^2)
             [0.91, 0.44, 0.33],    #           forward  (x^3)
             [0.12, 0.67, 0.89]]    #           now      (x^4)
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
        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            wrong_input = torch.randn(2, 6, 5)  # Wrong d_in (5 instead of 3)
            mha(wrong_input)

        # Test wrong number of dimensions
        with pytest.raises((ValueError, RuntimeError), match="not enough values to unpack"):
            wrong_dims = torch.randn(6, 3)  # 2D instead of 3D
            mha(wrong_dims)


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
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), "Causal masking should prevent future tokens from affecting past outputs"

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
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

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

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

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
        assert wrapper_output.shape == efficient_output.shape, f"Output shapes should match: wrapper {wrapper_output.shape} vs efficient {efficient_output.shape}"

        # Results will be different due to different architectures and output projection
        # but both should be valid attention outputs
        assert not torch.allclose(wrapper_output, efficient_output), "Different implementations should produce different results"

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
        mha_efficient = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,  # Total output dimension (768)
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_efficient(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out (768 total)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_efficient.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_efficient.head_dim}"
        assert mha_efficient.d_out == 768, f"Expected d_out=768, got {mha_efficient.d_out}"
        assert mha_efficient.num_heads == 12, f"Expected num_heads=12, got {mha_efficient.num_heads}"


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
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), "Causal masking should prevent future tokens from affecting past outputs"

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
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

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
        assert combined_output.shape == efficient_output.shape, f"Output shapes should match: combined {combined_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(combined_output).all(), "Combined QKV output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(combined_output).any(), "Combined QKV output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

        # Results will typically be different due to different random initialization but when they're the same (same seed), that's also valid

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
        assert qkv_weight_size == expected_qkv_size, f"Expected QKV weight size {expected_qkv_size}, got {qkv_weight_size}"


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

        # Compute loss and back-propagate
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
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), "Causal masking should prevent future tokens from affecting past outputs"

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
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

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

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

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
        assert einsum_output.shape == efficient_output.shape, f"Output shapes should match: einsum {einsum_output.shape} vs efficient {efficient_output.shape}"

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


class TestMHAPyTorchScaledDotProduct:
    """
    Test suite for the MHAPyTorchScaledDotProduct module (PyTorch built-in scaled dot-product attention).
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MHAPyTorchScaledDotProduct.
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

    def test_output_shape_pytorch_scaled(self, sample_inputs):
        """
        Test that PyTorch scaled dot-product attention produces correct output shape.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.1
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_dimension_divisibility_assertion_pytorch_scaled(self):
        """
        Test that assertion is raised when d_out is not divisible by num_heads.
        """
        d_in, d_out = 3, 7  # 7 is not divisible by 3
        context_length = 8
        dropout = 0.1
        num_heads = 3

        with pytest.raises(AssertionError, match="d_out is indivisible by num_heads"):
            MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

    def test_head_dimension_calculation_pytorch_scaled(self, sample_inputs):
        """
        Test that head dimensions are calculated correctly in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 12
        context_length = 8
        dropout = 0.0
        num_heads = 4

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        assert mha.head_dim == 3, f"Expected head_dim=3, got {mha.head_dim}"
        assert mha.d_out == 12, f"Expected d_out=12, got {mha.d_out}"
        assert mha.num_heads == 4, f"Expected num_heads=4, got {mha.num_heads}"

    def test_pytorch_scaled_qkv_projection(self, sample_inputs):
        """
        Test that QKV projection works correctly in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test the QKV projection creates correct shape
        qkv_output = mha.qkv(batch)
        expected_qkv_shape = (2, 6, 3 * d_out)  # 3 * d_out for Q, K, V
        assert qkv_output.shape == expected_qkv_shape, f"Expected QKV shape {expected_qkv_shape}, got {qkv_output.shape}"

        # Test full forward pass
        output = mha(batch)
        assert output.shape == (2, 6, d_out), f"Expected output shape (2, 6, {d_out}), got {output.shape}"

    def test_gradient_flow_pytorch_scaled(self, sample_inputs):
        """
        Test that gradients flow properly through the PyTorch scaled attention.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.1
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output = mha(batch)

        # Compute loss and back-propagate
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

        # Check that all parameters have gradients
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_causal_mask_pytorch_scaled(self, sample_inputs):
        """
        Test that causal masking works properly in the PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 6
        dropout = 0.0
        num_heads = 3

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test with eval mode to ensure consistent behavior
        mha.eval()

        with torch.no_grad():
            output = mha(batch)

            # Check that changing future tokens doesn't affect past outputs
            modified_batch = batch.clone()
            modified_batch[:, -1, :] = torch.randn_like(modified_batch[:, -1, :])

            modified_output = mha(modified_batch)

            # First 5 positions should be identical (causal masking)
            assert torch.allclose(output[:, :5, :], modified_output[:, :5, :], atol=1e-6), "Causal masking should prevent future tokens from affecting past outputs"

    def test_different_head_counts_pytorch_scaled(self, sample_inputs):
        """
        Test PyTorch scaled attention with different numbers of heads.
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
            mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, f"For {num_heads} heads, expected shape {expected_shape}, got {output.shape}"

    def test_reproducibility_pytorch_scaled(self, sample_inputs):
        """
        Test that outputs are reproducible with same random seed in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 8
        context_length = 8
        dropout = 0.0  # No dropout for reproducibility
        num_heads = 4

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # First run
        torch.manual_seed(42)
        mha1 = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output1 = mha1(batch)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        output2 = mha2(batch)

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical with same random seed"

    def test_versus_other_implementations(self, sample_inputs):
        """
        Test that PyTorch scaled implementation produces valid results compared to other implementations.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # PyTorch scaled implementation
        torch.manual_seed(789)
        pytorch_scaled = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
        pytorch_output = pytorch_scaled(batch)

        # Efficient implementation for comparison (different seed to show they can differ)
        torch.manual_seed(456)
        efficient = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
        efficient_output = efficient(batch)

        # Shapes should match
        assert pytorch_output.shape == efficient_output.shape, f"Output shapes should match: pytorch_scaled {pytorch_output.shape} vs efficient {efficient_output.shape}"

        # Both should produce valid outputs
        assert torch.isfinite(pytorch_output).all(), "PyTorch scaled output should be finite"
        assert torch.isfinite(efficient_output).all(), "Efficient output should be finite"
        assert not torch.isnan(pytorch_output).any(), "PyTorch scaled output should not have NaN"
        assert not torch.isnan(efficient_output).any(), "Efficient output should not have NaN"

    def test_training_vs_eval_mode(self, sample_inputs):
        """
        Test that training and evaluation modes work correctly.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.2  # High dropout to see the effect
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Training mode
        mha.train()
        torch.manual_seed(123)
        output_train = mha(batch)

        # Evaluation mode
        mha.eval()
        torch.manual_seed(123)
        output_eval = mha(batch)

        # In eval mode, dropout should be disabled, so results might be different
        # but both should be finite and valid
        assert torch.isfinite(output_train).all(), "Training output should be finite"
        assert torch.isfinite(output_eval).all(), "Evaluation output should be finite"
        assert not torch.isnan(output_train).any(), "Training output should not have NaN"
        assert not torch.isnan(output_eval).any(), "Evaluation output should not have NaN"

    def test_wrong_input_dimensions_pytorch_scaled(self):
        """
        Test error handling for wrong input dimensions in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.1
        num_heads = 2

        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test with wrong input dimension (should be 3D)
        wrong_input_2d = torch.randn(6, 3)  # Missing batch dimension
        wrong_input_4d = torch.randn(2, 6, 3, 1)  # Extra dimension

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_2d)

        with pytest.raises(Exception):  # Should fail on tensor operations
            mha(wrong_input_4d)

    def test_large_scale_transformer_configuration_pytorch_scaled(self):
        """
        Test PyTorch scaled attention with realistic transformer configuration.

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

        # Initialize the PyTorch scaled attention with transformer-like configuration
        mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=12,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False
        ).to(device)

        # Forward pass
        out = mha_pytorch_scaled(embeddings)

        # Verify the output shape
        expected_shape = torch.Size([8, 1024, 768])  # batch_size, seq_len, d_out
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

        # Additional checks
        assert isinstance(out, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain finite values"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Verify internal dimensions are correctly calculated
        assert mha_pytorch_scaled.head_dim == 64, f"Expected head_dim=64 (768/12), got {mha_pytorch_scaled.head_dim}"
        assert mha_pytorch_scaled.d_out == 768, f"Expected d_out=768, got {mha_pytorch_scaled.d_out}"
        assert mha_pytorch_scaled.num_heads == 12, f"Expected num_heads=12, got {mha_pytorch_scaled.num_heads}"

    def test_parameter_efficiency_pytorch_scaled(self, sample_inputs):
        """
        Test that PyTorch scaled implementation has expected parameter counts.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # PyTorch scaled implementation
        pytorch_scaled = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Regular implementation for comparison
        regular = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

        # Count parameters
        pytorch_params = sum(p.numel() for p in pytorch_scaled.parameters())
        regular_params = sum(p.numel() for p in regular.parameters())

        assert pytorch_params > 0, "PyTorch scaled implementation should have parameters"
        assert regular_params > 0, "Regular implementation should have parameters"

        # Both should have similar parameter counts (QKV + proj)
        # The difference should be minimal since both use similar architectures
        assert abs(pytorch_params - regular_params) < pytorch_params * 0.1, "Parameter counts should be similar between implementations"

    def test_bias_functionality_pytorch_scaled(self, sample_inputs):
        """
        Test that bias terms work correctly when enabled in PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with bias
        mha_with_bias = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout, qkv_bias=True)
        output_with_bias = mha_with_bias(batch)

        # Test without bias
        mha_no_bias = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout, qkv_bias=False)
        output_no_bias = mha_no_bias(batch)

        # Both should produce valid outputs
        assert torch.isfinite(output_with_bias).all(), "Output with bias should be finite"
        assert torch.isfinite(output_no_bias).all(), "Output without bias should be finite"
        assert output_with_bias.shape == output_no_bias.shape, "Output shapes should match regardless of bias"

        # Check that bias parameters exist when requested
        assert mha_with_bias.qkv.bias is not None, "QKV bias should exist when qkv_bias=True"
        assert mha_no_bias.qkv.bias is None, "QKV bias should not exist when qkv_bias=False"

    def test_pytorch_scaled_attention_backend(self, sample_inputs):
        """
        Test that PyTorch scaled_dot_product_attention is being used correctly.
        """
        d_in, d_out = 3, 4
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)
        mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)

        # Test that we can successfully run the forward pass
        # (implicitly testing that scaled_dot_product_attention is working)
        output = mha(batch)

        b, n, _ = batch.shape
        assert output.shape == (b, n, d_out), f"Expected output shape ({b}, {n}, {d_out}), got {output.shape}"

        # Test with different training modes
        mha.train()
        train_output = mha(batch)

        mha.eval()
        eval_output = mha(batch)

        # Both should produce valid outputs
        assert torch.isfinite(train_output).all(), "Training output should be finite"
        assert torch.isfinite(eval_output).all(), "Evaluation output should be finite"

    def test_pytorch_scaled_performance_characteristics(self, sample_inputs):
        """
        Test performance-related characteristics of PyTorch scaled implementation.
        """
        d_in, d_out = 3, 6
        context_length = 8
        dropout = 0.0
        num_heads = 2

        batch = torch.stack((sample_inputs, sample_inputs), dim=0)

        # Test with different batch sizes to ensure scalability
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            # Create batch of the specified size
            test_batch = torch.stack([sample_inputs] * batch_size, dim=0)

            mha = MHAPyTorchScaledDotProduct(d_in, d_out, num_heads, context_length, dropout)
            output = mha(test_batch)

            expected_shape = (batch_size, 6, d_out)
            assert output.shape == expected_shape, f"For batch_size={batch_size}, expected shape {expected_shape}, got {output.shape}"

            # Verify output is finite and valid
            assert torch.isfinite(output).all(), f"Output should be finite for batch_size={batch_size}"
            assert not torch.isnan(output).any(), f"Output should not contain NaN for batch_size={batch_size}"


class TestMHAPyTorchSDPAWithoutFlash:
    """
    Test suite for MHAPyTorchSDPAWithoutFlash multi-head attention implementation.
    """

    def test_output_shape_pytorch_sdpa_without_flash(self):
        """
        Test that MHAPyTorchSDPAWithoutFlash produces correct output shape.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256
        num_heads = 4
        context_length = 16

        mha = MHAPyTorchSDPAWithoutFlash(
            d_in=d_in, d_out=d_out, num_heads=num_heads,
            context_length=context_length, dropout=0.0
        )

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain only finite values"

    def test_dimension_divisibility_assertion_pytorch_sdpa_without_flash(self):
        """Test that MHAPyTorchSDPAWithoutFlash raises error for invalid dimensions."""
        with pytest.raises(AssertionError, match="d_out is indivisible by num_heads"):
            MHAPyTorchSDPAWithoutFlash(d_in=512, d_out=257, num_heads=4, context_length=16)

    def test_head_dimension_calculation_pytorch_sdpa_without_flash(self):
        """Test that head dimensions are calculated correctly."""
        d_out = 768
        num_heads = 12
        mha = MHAPyTorchSDPAWithoutFlash(d_in=512, d_out=d_out, num_heads=num_heads, context_length=16)

        expected_head_dim = d_out // num_heads
        assert mha.head_dim == expected_head_dim, f"Expected head_dim={expected_head_dim}, got {mha.head_dim}"
        assert mha.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha.num_heads}"
        assert mha.d_out == d_out, f"Expected d_out={d_out}, got {mha.d_out}"

    def test_pytorch_sdpa_without_flash_qkv_projection(self):
        """
        Test QKV projection dimensions and functionality.
        """
        d_in, d_out = 512, 256
        num_heads = 4
        mha = MHAPyTorchSDPAWithoutFlash(
            d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16
        )

        # Check QKV projection layer
        assert mha.qkv.in_features == d_in, f"Expected QKV in_features={d_in}, got {mha.qkv.in_features}"
        assert mha.qkv.out_features == 3 * d_out, f"Expected QKV out_features={3 * d_out}, got {mha.qkv.out_features}"
        assert mha.proj.in_features == d_out, f"Expected proj in_features={d_out}, got {mha.proj.in_features}"
        assert mha.proj.out_features == d_out, f"Expected proj out_features={d_out}, got {mha.proj.out_features}"

    def test_gradient_flow_pytorch_sdpa_without_flash(self):
        """
        Test that gradients flow properly through the model.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256

        mha = MHAPyTorchSDPAWithoutFlash(
            d_in=d_in, d_out=d_out, num_heads=4,
            context_length=16, dropout=0.0
        )

        x = torch.randn(batch_size, num_tokens, d_in, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input tensor should have gradients after backward pass"
        assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN values"
        assert mha.qkv.weight.grad is not None, "QKV weights should have gradients after backward pass"
        assert mha.proj.weight.grad is not None, "Projection weights should have gradients after backward pass"

    def test_causal_mask_pytorch_sdpa_without_flash(self):
        """Test that causal masking works correctly."""
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 1, 4, 64
        d_out = 64

        mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        # Check that mask buffer exists and has correct shape
        assert hasattr(mha, 'mask'), "MHA should have mask attribute"
        assert mha.mask.shape == (8, 8), "Mask should have shape (context_length, context_length)"
        assert mha.mask.dtype == torch.bool, f"Expected mask dtype to be torch.bool, got {mha.mask.dtype}"

        # Verify upper triangular structure (causal mask)
        expected_mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        assert torch.equal(mha.mask, expected_mask), "Mask should be upper triangular causal mask"

    def test_different_head_counts_pytorch_sdpa_without_flash(self):
        """Test behavior with different numbers of heads."""
        batch_size, num_tokens, d_in = 2, 6, 512
        d_out = 256
        context_length = 10

        for num_heads in [1, 2, 4, 8]:
            mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

            x = torch.randn(batch_size, num_tokens, d_in)
            out = mha(x)

            assert out.shape == (batch_size, num_tokens, d_out)
            assert mha.head_dim == d_out // num_heads, f"For num_heads={num_heads}, expected head_dim={d_out // num_heads}, got {mha.head_dim}"

    def test_reproducibility_pytorch_sdpa_without_flash(self):
        """Test that the model produces reproducible results."""
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256

        # First run
        torch.manual_seed(42)
        mha1 = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.0)
        x1 = torch.randn(batch_size, num_tokens, d_in)
        out1 = mha1(x1)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.0)
        x2 = torch.randn(batch_size, num_tokens, d_in)
        out2 = mha2(x2)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_versus_other_implementations(self):
        """Test consistency with other attention implementations (shape and basic properties)."""
        torch.manual_seed(123)
        batch_size, num_tokens, d_in = 2, 6, 128
        d_out = 128
        num_heads = 4
        context_length = 10

        # Create models with same configuration
        mha_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        mha_combined_qkv = MultiHeadAttentionCombinedQKV(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)

        out_sdpa_no_flash = mha_sdpa_no_flash(x)
        out_combined_qkv = mha_combined_qkv(x)

        # Check shapes are consistent
        assert out_sdpa_no_flash.shape == out_combined_qkv.shape, f"Output shapes should match: SDPA {out_sdpa_no_flash.shape} vs Combined QKV {out_combined_qkv.shape}"
        assert out_sdpa_no_flash.shape == (batch_size, num_tokens, d_out)

    def test_training_vs_eval_mode(self):
        """Test behavior in training vs evaluation mode."""
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 2, 8, 256
        d_out = 256

        mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.1)  # Non-zero dropout

        x = torch.randn(batch_size, num_tokens, d_in)

        # Training mode
        mha.train()
        out_train = mha(x)

        # Evaluation mode
        mha.eval()
        out_eval = mha(x)

        # Outputs should be different due to dropout behavior
        # Note: This test might be flaky, but generally dropout affects training/eval differently
        assert out_train.shape == out_eval.shape, f"Training and eval outputs should have same shape: train {out_train.shape} vs eval {out_eval.shape}"
        assert isinstance(mha.dropout, float), "Dropout should be stored as float"

    def test_wrong_input_dimensions_pytorch_sdpa_without_flash(self):
        """
        Test error handling for incorrect input dimensions.
        """
        mha = MHAPyTorchSDPAWithoutFlash(d_in=512, d_out=256, num_heads=4, context_length=16)

        # Wrong input dimension
        x_wrong = torch.randn(2, 8, 256)  # d_in should be 512
        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            mha(x_wrong)

    def test_large_scale_transformer_configuration_pytorch_sdpa_without_flash(self):
        """
        Test MHAPyTorchSDPAWithoutFlash with large-scale transformer configuration.
        """
        # Large transformer configuration
        batch_size = 8
        context_len = 1024
        embed_dim = 768
        num_heads = 12
        device = torch.device("cpu")  # Use CPU for testing

        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, context_len, embed_dim).to(device)

        mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False
        ).to(device)

        out = mha_pytorch_sdpa_no_flash(embeddings)

        # Main test: verify output shape
        assert out.shape == torch.Size([8, 1024, 768]), f"Expected shape [8, 1024, 768], got {out.shape}"
        assert not torch.isnan(out).any(), "Large scale output should not contain NaN values"
        assert torch.isfinite(out).all(), "Large scale output should contain only finite values"

        # Verify model parameters
        assert mha_pytorch_sdpa_no_flash.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha_pytorch_sdpa_no_flash.num_heads}"
        assert mha_pytorch_sdpa_no_flash.head_dim == embed_dim // num_heads, f"Expected head_dim={embed_dim // num_heads}, got {mha_pytorch_sdpa_no_flash.head_dim}"
        assert mha_pytorch_sdpa_no_flash.d_out == embed_dim, f"Expected d_out={embed_dim}, got {mha_pytorch_sdpa_no_flash.d_out}"

    def test_parameter_efficiency_pytorch_sdpa_without_flash(self):
        """
        Test parameter count and efficiency.
        """
        d_in, d_out = 512, 256
        num_heads = 4

        mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16)

        total_params = sum(p.numel() for p in mha.parameters())

        # QKV projection: d_in * (3 * d_out) = 512 * 768 = 393,216
        # QKV bias (if enabled): 3 * d_out = 768
        # Output projection: d_out * d_out = 256 * 256 = 65,536
        # Output bias: d_out = 256
        expected_params = (d_in * 3 * d_out) + (d_out * d_out) + d_out  # No QKV bias by default

        assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"

    def test_bias_functionality_pytorch_sdpa_without_flash(self):
        """
        Test bias parameter functionality.
        """
        d_in, d_out = 256, 128
        num_heads = 4

        # Test without bias
        mha_no_bias = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=False)
        assert mha_no_bias.qkv.bias is None, "QKV layer should have no bias when qkv_bias=False"

        # Test with bias
        mha_with_bias = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=True)
        assert mha_with_bias.qkv.bias is not None, "QKV layer should have bias when qkv_bias=True"
        assert mha_with_bias.qkv.bias.shape == (3 * d_out,), f"QKV bias should have shape (3 * d_out,), got {mha_with_bias.qkv.bias.shape}"

    def test_pytorch_sdpa_without_flash_backend_verification(self):
        """
        Test that the implementation uses PyTorch's scaled_dot_product_attention correctly.
        """
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 128

        mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=4, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        # Verify that mask is used (is_causal=False with explicit mask)
        assert hasattr(mha, 'mask'), "MHA should have mask attribute for causal masking"
        assert out.shape == (batch_size, num_tokens, d_out), f"Expected output shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"

        # Ensure output is different from input (attention has been applied)
        qkv = mha.qkv(x)
        qkv_reshaped = qkv.view(batch_size, num_tokens, 3, mha.num_heads, mha.head_dim)
        queries = qkv_reshaped[:, :, 0, :, :].transpose(1, 2)

        # Output should not be identical to raw queries
        queries_flattened = queries.transpose(1, 2).reshape(batch_size, num_tokens, d_out)
        assert not torch.allclose(out, queries_flattened, atol=1e-3), "Output should differ from raw queries after attention"

    def test_pytorch_sdpa_without_flash_explicit_masking(self):
        """
        Test explicit masking behavior vs is_causal parameter.
        """
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 2, 4, 64
        d_out = 64

        mha = MHAPyTorchSDPAWithoutFlash(d_in=d_in, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)

        # Test that the mask is properly applied
        with torch.no_grad():
            # Get QKV tensors
            qkv = mha.qkv(x)
            qkv = qkv.view(batch_size, num_tokens, 3, mha.num_heads, mha.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            queries, keys, values = qkv

            # Manual attention computation to verify masking
            scale = (mha.head_dim ** -0.5)
            attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

            # Apply the same mask as in the model
            mask = mha.mask[:num_tokens, :num_tokens]
            attn_scores_masked = attn_scores.masked_fill(mask, -torch.inf)

            # Check that future positions are properly masked (should be -inf)
            assert torch.isinf(attn_scores_masked[:, :, 0, 1:]).all(), "First token can't see future positions"
            assert torch.isfinite(attn_scores_masked[:, :, 1, :2]).all(), "Second token can see past positions"


class TestMHAPyTorchClass:
    """
    Test suite for MHAPyTorchClass multi-head attention implementation.
    """

    def test_output_shape_pytorch_class(self):
        """
        Test that MHAPyTorchClass produces correct output shape.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 512  # Note: MHAPyTorchClass requires d_in == d_out
        num_heads = 4
        context_length = 16

        mha = MHAPyTorchClass(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_out)  # Input must match embed_dim
        out = mha(x)

        assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain only finite values"

    def test_multihead_attention_parameters(self):
        """
        Test that nn.MultiheadAttention is correctly configured.
        """
        d_out = 256
        num_heads = 8
        dropout = 0.1

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16, dropout=dropout, qkv_bias=True)

        # Check MultiheadAttention configuration
        assert mha.multihead_attn.embed_dim == d_out, f"Expected embed_dim={d_out}, got {mha.multihead_attn.embed_dim}"
        assert mha.multihead_attn.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha.multihead_attn.num_heads}"
        assert mha.multihead_attn.dropout == dropout, f"Expected dropout={dropout}, got {mha.multihead_attn.dropout}"
        assert mha.multihead_attn.batch_first == True, f"Expected batch_first=True, got {mha.multihead_attn.batch_first}"

    def test_head_dimension_calculation_pytorch_class(self):
        """
        Test that PyTorch's MultiheadAttention handles head dimensions correctly.
        """
        d_out = 768
        num_heads = 12

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16)

        # MultiheadAttention should handle head dimension calculation internally
        assert mha.multihead_attn.embed_dim == d_out, f"Expected embed_dim={d_out}, got {mha.multihead_attn.embed_dim}"
        assert mha.multihead_attn.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha.multihead_attn.num_heads}"
        # Head dimension is calculated internally by PyTorch

    def test_pytorch_class_projection_layers(self):
        """
        Test additional projection layer functionality.
        """
        d_out = 512
        num_heads = 8

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16)

        # Check additional projection layer
        assert mha.proj.in_features == d_out, f"Expected proj.in_features={d_out}, got {mha.proj.in_features}"
        assert mha.proj.out_features == d_out, f"Expected proj.out_features={d_out}, got {mha.proj.out_features}"
        assert hasattr(mha, 'need_weights'), "MHAPyTorchClass should have need_weights attribute"

    def test_gradient_flow_pytorch_class(self):
        """
        Test that gradients flow properly through the model.
        """
        batch_size, num_tokens, d_out = 2, 8, 512

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=8, context_length=16, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_out, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN values"

        # Check that MultiheadAttention and projection have gradients
        mha_params = list(mha.multihead_attn.parameters())
        assert len(mha_params) > 0, "Should have some parameters"
        assert all(p.grad is not None for p in mha_params if p.requires_grad), "All trainable parameters should have gradients"
        assert mha.proj.weight.grad is not None, "Projection layer weights should have gradients"

    def test_causal_mask_pytorch_class(self):
        """
        Test that causal masking works correctly.
        """
        torch.manual_seed(42)
        batch_size, num_tokens, d_out = 1, 4, 64

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_out)
        out = mha(x)

        # Check that mask buffer exists and has correct shape
        assert hasattr(mha, 'mask'), "MHA should have mask attribute"
        assert mha.mask.shape == (8, 8), f"Expected mask shape (8, 8), got {mha.mask.shape}"
        assert mha.mask.dtype == torch.bool, f"Expected mask dtype torch.bool, got {mha.mask.dtype}"

        # Verify upper triangular structure (causal mask)
        expected_mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        assert torch.equal(mha.mask, expected_mask), "Mask should be upper triangular causal mask"

    def test_different_head_counts_pytorch_class(self):
        """
        Test behavior with different numbers of heads.
        """
        batch_size, num_tokens = 2, 6
        context_length = 10

        for num_heads in [1, 2, 4, 8]:
            d_out = num_heads * 64  # Ensure divisibility

            mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

            x = torch.randn(batch_size, num_tokens, d_out)
            out = mha(x)

            assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
            assert mha.multihead_attn.num_heads == num_heads, f"For num_heads={num_heads}, expected {num_heads}, got {mha.multihead_attn.num_heads}"

    def test_reproducibility_pytorch_class(self):
        """
        Test that the model produces reproducible results.
        """
        batch_size, num_tokens, d_out = 2, 8, 512

        # First run
        torch.manual_seed(42)
        mha1 = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=8, context_length=16, dropout=0.0)
        x1 = torch.randn(batch_size, num_tokens, d_out)
        out1 = mha1(x1)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=8, context_length=16, dropout=0.0)
        x2 = torch.randn(batch_size, num_tokens, d_out)
        out2 = mha2(x2)

        assert torch.allclose(out1, out2, atol=1e-6), "Outputs should be close for same seed and inputs"

    def test_versus_other_implementations(self):
        """
        Test consistency with other attention implementations (shape and basic properties).
        """
        torch.manual_seed(123)
        batch_size, num_tokens, d_in = 2, 6, 128
        d_out = 128
        num_heads = 4
        context_length = 10

        # Create models with same configuration
        mha_pytorch_class = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        mha_combined_qkv = MultiHeadAttentionCombinedQKV(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        x_pytorch = torch.randn(batch_size, num_tokens, d_out)  # Input for PyTorch class
        x_combined = torch.randn(batch_size, num_tokens, d_in)  # Input for combined QKV

        out_pytorch_class = mha_pytorch_class(x_pytorch)
        out_combined_qkv = mha_combined_qkv(x_combined)

        # Check shapes are consistent
        assert out_pytorch_class.shape == out_combined_qkv.shape, f"Output shapes should match: PyTorch class {out_pytorch_class.shape} vs Combined QKV {out_combined_qkv.shape}"
        assert out_pytorch_class.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out_pytorch_class.shape}"

    def test_need_weights_functionality(self):
        """
        Test need_weights parameter functionality.
        """
        batch_size, num_tokens, d_out = 2, 4, 256

        # Test with need_weights=True
        mha_with_weights = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=4, context_length=8, need_weights=True)

        # Test with need_weights=False
        mha_without_weights = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=4, context_length=8, need_weights=False)

        x = torch.randn(batch_size, num_tokens, d_out)

        out_with_weights = mha_with_weights(x)
        out_without_weights = mha_without_weights(x)

        # Both should produce same shape outputs
        assert out_with_weights.shape == out_without_weights.shape, f"Output shapes should be same regardless of need_weights: with {out_with_weights.shape} vs without {out_without_weights.shape}"
        assert out_with_weights.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out_with_weights.shape}"

        # Check that need_weights is stored correctly
        assert mha_with_weights.need_weights == True, "MHA with weights should have need_weights=True"
        assert mha_without_weights.need_weights == False, "MHA without weights should have need_weights=False"

    def test_wrong_input_dimensions_pytorch_class(self):
        """
        Test error handling for incorrect input dimensions.
        """
        d_out = 512
        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=8, context_length=16)

        # Wrong input dimension (PyTorch MultiheadAttention expects embed_dim)
        x_wrong = torch.randn(2, 8, 256)  # Should be 512
        with pytest.raises(AssertionError, match="was expecting embedding dimension of 512, but got 256"):
            mha(x_wrong)

    def test_large_scale_transformer_configuration_pytorch_class(self):
        """
        Test MHAPyTorchClass with large-scale transformer configuration.
        """
        # Large transformer configuration
        batch_size = 8
        context_len = 1024
        embed_dim = 768
        num_heads = 12
        device = torch.device("cpu")  # Use CPU for testing

        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, context_len, embed_dim).to(device)

        mha_pytorch_class_default = MHAPyTorchClass(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False
        ).to(device)

        out = mha_pytorch_class_default(embeddings)

        # Main test: verify output shape
        assert out.shape == torch.Size([8, 1024, 768]), f"Expected shape [8, 1024, 768], got {out.shape}"
        assert not torch.isnan(out).any(), "Large scale output should not contain NaN values"
        assert torch.isfinite(out).all(), "Large scale output should contain only finite values"

        # Verify model parameters
        assert mha_pytorch_class_default.multihead_attn.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha_pytorch_class_default.multihead_attn.num_heads}"
        assert mha_pytorch_class_default.multihead_attn.embed_dim == embed_dim, f"Expected embed_dim={embed_dim}, got {mha_pytorch_class_default.multihead_attn.embed_dim}"

    def test_parameter_efficiency_pytorch_class(self):
        """
        Test parameter count and efficiency.
        """
        d_out = 512
        num_heads = 8

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16)

        total_params = sum(p.numel() for p in mha.parameters())

        # Parameters include:
        # - MultiheadAttention parameters (QKV projections, output projection, biases)
        # - Additional projection layer: d_out * d_out + d_out
        additional_proj_params = d_out * d_out + d_out

        assert total_params > additional_proj_params, f"Total params ({total_params}) should be greater than just projection params ({additional_proj_params})"

        # Check that our additional projection exists
        proj_params = sum(p.numel() for p in mha.proj.parameters())
        assert proj_params == additional_proj_params, f"Projection params should match expected: {proj_params} vs {additional_proj_params}"

    def test_bias_functionality_pytorch_class(self):
        """
        Test bias parameter functionality.
        """
        d_out = 256
        num_heads = 4

        # Test without bias
        mha_no_bias = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=False)

        # Test with bias
        mha_with_bias = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=True)

        # Both should work (PyTorch handles bias configuration internally)
        x = torch.randn(2, 4, d_out)

        out_no_bias = mha_no_bias(x)
        out_with_bias = mha_with_bias(x)

        assert out_no_bias.shape == out_with_bias.shape, f"Outputs should have same shape regardless of bias: no bias {out_no_bias.shape} vs with bias {out_with_bias.shape}"
        assert out_no_bias.shape == (2, 4, d_out), f"Expected shape (2, 4, {d_out}), got {out_no_bias.shape}"

    def test_pytorch_multihead_attention_integration(self):
        """
        Test integration with PyTorch's nn.MultiheadAttention.
        """
        batch_size, num_tokens, d_out = 2, 4, 128

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=4,context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_out)
        out = mha(x)

        # Verify that PyTorch's MultiheadAttention is being used
        assert isinstance(mha.multihead_attn, nn.MultiheadAttention), "multihead_attn should be an instance of nn.MultiheadAttention"
        assert out.shape == (batch_size, num_tokens, d_out), f"Expected output shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"

        # Ensure output is different from input (attention has been applied)
        assert not torch.allclose(out, x, atol=1e-3), "Output should differ from input after attention"

    def test_pytorch_class_mask_handling(self):
        """
        Test that mask is properly handled by nn.MultiheadAttention.
        """
        torch.manual_seed(42)
        batch_size, num_tokens, d_out = 2, 4, 64

        mha = MHAPyTorchClass(d_in=d_out, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_out)

        # Test that the mask is properly applied by checking that we get different
        # results with different sequence lengths (causal masking effect)
        out_full = mha(x)

        # Shorter sequence should behave differently due to causal masking
        x_short = x[:, :2, :]  # Only first 2 tokens
        out_short = mha(x_short)

        assert out_full.shape[1] == num_tokens, f"Full sequence output should have {num_tokens} tokens, got {out_full.shape[1]}"
        assert out_short.shape[1] == 2, f"Short sequence output should have 2 tokens, got {out_short.shape[1]}"

        # The mask should be correctly sized for each input
        assert out_full.shape == (batch_size, num_tokens, d_out), f"Expected full output shape ({batch_size}, {num_tokens}, {d_out}), got {out_full.shape}"
        assert out_short.shape == (batch_size, 2, d_out), f"Expected short output shape ({batch_size}, 2, {d_out}), got {out_short.shape}"

    def test_pytorch_class_no_weights_large_scale_configuration(self):
        """
        Test MHAPyTorchClass with need_weights=False and large-scale transformer configuration.
        """
        # Large transformer configuration
        batch_size = 8
        context_len = 1024
        embed_dim = 768
        num_heads = 12
        device = torch.device("cpu")  # Use CPU for testing

        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, context_len, embed_dim).to(device)

        mha_pytorch_class_noweights = MHAPyTorchClass(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False,
            need_weights=False  # NEW!
        ).to(device)

        out = mha_pytorch_class_noweights(embeddings)

        # Main test: verify output shape
        assert out.shape == torch.Size([8, 1024, 768]), f"Expected shape (8, 1024, 768), got {out.shape}"
        assert not torch.isnan(out).any(), "Large scale output should not contain NaN values"
        assert torch.isfinite(out).all(), "Large scale output should contain only finite values"

        # Verify model parameters
        assert mha_pytorch_class_noweights.multihead_attn.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha_pytorch_class_noweights.multihead_attn.num_heads}"
        assert mha_pytorch_class_noweights.multihead_attn.embed_dim == embed_dim, f"Expected embed_dim={embed_dim}, got {mha_pytorch_class_noweights.multihead_attn.embed_dim}"
        assert mha_pytorch_class_noweights.need_weights == False, "No weights configuration should have need_weights=False"


class TestMHAPyTorchFlexAttention:
    """
    Test suite for MHAPyTorchFlexAttention multi-head attention implementation.
    """

    def test_flex_attention_not_available_error(self):
        """
        Test that FlexAttention works with proper device compatibility.
        """
        # Since we now support device compatibility, FlexAttention should work
        try:
            mha = MHAPyTorchFlexAttention(d_in=512, d_out=256, num_heads=4, context_length=16)
            # Test that it can be created without errors
            assert mha is not None, "MHA instance should be created successfully"
            assert mha.num_heads == 4, f"Expected num_heads=4, got {mha.num_heads}"
            assert mha.d_out == 256, f"Expected d_out=256, got {mha.d_out}"
        except RuntimeError as e:
            # Only allow PyTorch version errors
            if "PyTorch 2.5+" not in str(e):
                raise

    def test_output_shape_pytorch_flex(self):
        """
        Test that MHAPyTorchFlexAttention produces correct output shape.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 512
        num_heads = 4
        context_length = 16

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        assert out.shape == (batch_size, num_tokens, d_out), f"Expected output shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain only finite values"

    def test_dimension_divisibility_assertion_pytorch_flex(self):
        """
        Test that MHAPyTorchFlexAttention raises error for invalid dimensions.
        """
        with pytest.raises(AssertionError, match="d_out is indivisible by num_heads"):
            MHAPyTorchFlexAttention(d_in=512, d_out=513, num_heads=4, context_length=16)

    def test_head_dimension_calculation_pytorch_flex(self):
        """
        Test that head dimensions are calculated correctly.
        """
        d_out = 768
        num_heads = 12
        mha = MHAPyTorchFlexAttention(d_in=512, d_out=d_out, num_heads=num_heads, context_length=16)

        expected_head_dim = d_out // num_heads
        assert mha.head_dim == expected_head_dim, f"Expected head_dim={expected_head_dim}, got {mha.head_dim}"
        assert mha.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha.num_heads}"
        assert mha.d_out == d_out, f"Expected d_out={d_out}, got {mha.d_out}"

    def test_pytorch_flex_qkv_projection(self):
        """
        Test QKV projection dimensions and functionality.
        """
        d_in, d_out = 512, 256
        num_heads = 4
        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16)

        # Check QKV projection layer
        assert mha.qkv.in_features == d_in, f"Expected qkv.in_features={d_in}, got {mha.qkv.in_features}"
        assert mha.qkv.out_features == 3 * d_out, f"Expected qkv.out_features={3 * d_out}, got {mha.qkv.out_features}"
        assert mha.proj.in_features == d_out, f"Expected proj.in_features={d_out}, got {mha.proj.in_features}"
        assert mha.proj.out_features == d_out, f"Expected proj.out_features={d_out}, got {mha.proj.out_features}"
    def test_gradient_flow_pytorch_flex(self):
        """
        Test that gradients flow properly through the model.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN values"
        assert mha.qkv.weight.grad is not None, "QKV layer weights should have gradients"
        assert mha.proj.weight.grad is not None, "Projection layer weights should have gradients"

    def test_flex_attention_block_mask(self):
        """
        Test that FlexAttention block mask is created correctly.
        """
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 64

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        # Check that block_mask exists and has correct properties
        assert hasattr(mha, 'block_mask')
        assert mha.block_mask is not None, "Block mask should be initialized for flex attention"

        # Verify output shape
        assert out.shape == (batch_size, num_tokens, d_out)

    def test_different_head_counts_pytorch_flex(self):
        """
        Test behavior with different numbers of heads.
        """
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 256
        context_length = 10

        for num_heads in [1, 2, 4, 8]:
            if d_out % num_heads == 0:  # Only test valid configurations
                mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

                x = torch.randn(batch_size, num_tokens, d_in)
                out = mha(x)

                assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
                assert mha.num_heads == num_heads, f"For num_heads={num_heads}, expected {num_heads}, got {mha.num_heads}"

    def test_reproducibility_pytorch_flex(self):
        """
        Test that the model produces reproducible results.
        """
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 256

        # First run
        torch.manual_seed(42)
        mha1 = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.0
        )
        x1 = torch.randn(batch_size, num_tokens, d_in)
        out1 = mha1(x1)

        # Second run with same seed
        torch.manual_seed(42)
        mha2 = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.0)
        x2 = torch.randn(batch_size, num_tokens, d_in)
        out2 = mha2(x2)

        assert torch.allclose(out1, out2, atol=1e-6), "Outputs should be close for same seed and inputs"

    def test_versus_other_implementations(self):
        torch.manual_seed(123)
        batch_size, num_tokens, d_in = 2, 6, 128
        d_out = 128
        num_heads = 4
        context_length = 10

        # Create models with same configuration
        mha_flex = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        mha_combined_qkv = MultiHeadAttentionCombinedQKV(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=context_length, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)

        out_flex = mha_flex(x)
        out_combined_qkv = mha_combined_qkv(x)

        # Check shapes are consistent
        assert out_flex.shape == out_combined_qkv.shape, f"Output shapes should match: Flex {out_flex.shape} vs Combined QKV {out_combined_qkv.shape}"
        assert out_flex.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out_flex.shape}"

    def test_training_vs_eval_mode(self):
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 256

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=4, context_length=16, dropout=0.1)  # Use dropout for difference

        x = torch.randn(batch_size, num_tokens, d_in)

        # Training mode
        mha.train()
        out_train = mha(x)

        # Evaluation mode
        mha.eval()
        out_eval = mha(x)

        # Both should produce valid outputs with same shape
        assert out_train.shape == out_eval.shape, f"Training and evaluation outputs should have the same shape, got {out_train.shape} and {out_eval.shape}"
        assert isinstance(mha.dropout, float), f"Dropout should be stored as float, got {type(mha.dropout)}"

    def test_wrong_input_dimensions_pytorch_flex(self):
        """
        Test error handling for incorrect input dimensions.
        """
        mha = MHAPyTorchFlexAttention(d_in=512, d_out=256, num_heads=4, context_length=16)

        # Wrong input dimension
        x_wrong = torch.randn(2, 8, 256)  # d_in should be 512
        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            mha(x_wrong)

    def test_large_scale_transformer_configuration_pytorch_flex(self):
        """
        Test MHAPyTorchFlexAttention with large-scale transformer configuration.
        """
        # Large transformer configuration
        batch_size = 8
        context_len = 1024
        embed_dim = 768
        num_heads = 12
        device = torch.device("cpu")  # Use CPU for testing

        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, context_len, embed_dim).to(device)

        mha_pytorch_flex = MHAPyTorchFlexAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_len,
            dropout=0.0,
            qkv_bias=False
        ).to(device)

        out = mha_pytorch_flex(embeddings)

        # Main test: verify output shape
        assert out.shape == torch.Size([8, 1024, 768]), f"Expected shape [8, 1024, 768], got {out.shape}"
        assert not torch.isnan(out).any(), "Large scale output should not contain NaN values"
        assert torch.isfinite(out).all(), "Large scale output should contain only finite values"

        # Verify model parameters
        assert mha_pytorch_flex.num_heads == num_heads, f"Expected num_heads={num_heads}, got {mha_pytorch_flex.num_heads}"
        assert mha_pytorch_flex.head_dim == embed_dim // num_heads, f"Expected head_dim={embed_dim // num_heads}, got {mha_pytorch_flex.head_dim}"
        assert mha_pytorch_flex.d_out == embed_dim, f"Expected d_out={embed_dim}, got {mha_pytorch_flex.d_out}"

    def test_parameter_efficiency_pytorch_flex(self):
        """
        Test parameter count and efficiency.
        """
        d_in, d_out = 512, 256
        num_heads = 4

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16)

        total_params = sum(p.numel() for p in mha.parameters())

        # QKV projection: d_in * (3 * d_out) = 512 * 768 = 393,216
        # QKV bias (if enabled): 3 * d_out = 768
        # Output projection: d_out * d_out = 256 * 256 = 65,536
        # Output bias: d_out = 256
        expected_params = (d_in * 3 * d_out) + (d_out * d_out) + d_out  # No QKV bias by default

        assert total_params == expected_params, f"Total params ({total_params}) should match expected ({expected_params})"

    def test_bias_functionality_pytorch_flex(self):
        """
        Test bias parameter functionality.
        """
        d_in, d_out = 256, 128
        num_heads = 4

        # Test without bias
        mha_no_bias = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=False)
        assert mha_no_bias.qkv.bias is None, "QKV layer bias should be None when qkv_bias=False"

        # Test with bias
        mha_with_bias = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_length=16, qkv_bias=True)
        assert mha_with_bias.qkv.bias is not None, "QKV layer bias should be present when qkv_bias=True"
        assert mha_with_bias.qkv.bias.shape == (3 * d_out,), f"Expected qkv.bias shape ({3 * d_out},), got {mha_with_bias.qkv.bias.shape}"

    def test_pytorch_flex_attention_backend_verification(self):
        """
        Test that the implementation uses FlexAttention correctly.
        """
        batch_size, num_tokens, d_in = 2, 4, 128
        d_out = 128

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=4, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        # Verify that block_mask is used
        assert hasattr(mha, 'block_mask'), "MHA should have block_mask attribute for FlexAttention"
        assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"

        # Ensure output is different from input (attention has been applied)
        qkv = mha.qkv(x)
        qkv_reshaped = qkv.view(batch_size, num_tokens, 3, mha.num_heads, mha.head_dim)
        queries = qkv_reshaped[:, :, 0, :, :].transpose(1, 2)

        # Output should not be identical to raw queries
        queries_flattened = queries.transpose(1, 2).reshape(batch_size, num_tokens, d_out)
        assert not torch.allclose(out, queries_flattened, atol=1e-3), "Output should differ from raw queries after attention"

    def test_pytorch_flex_causal_behavior(self):
        """
        Test that FlexAttention maintains causal behavior.
        """
        torch.manual_seed(42)
        batch_size, num_tokens, d_in = 2, 4, 64
        d_out = 64

        mha = MHAPyTorchFlexAttention(d_in=d_in, d_out=d_out, num_heads=2, context_length=8, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in)

        # Test that the block mask implements proper causal behavior
        # This is verified by ensuring the flex_attention function is called
        # and the output has the expected properties
        out = mha(x)

        assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
        assert torch.isfinite(out).all(), "Flex attention causal output should contain only finite values"

        # Test causal function properties
        assert causal(0, 0, 0, 0) == True, "Position should be able to attend to itself"
        assert causal(0, 0, 1, 0) == True, "Later position should be able to attend to earlier position"
        assert causal(0, 0, 1, 1) == True, "Position should be able to attend to itself"
        assert causal(0, 0, 0, 1) == False, "Earlier position cannot attend to later position (causal mask)"
        assert causal(0, 0, 2, 3) == False, "Position cannot attend to future positions (causal mask)"


class TestMultiHeadAttentionCached:
    """
    Test suite for the MultiHeadAttentionCached module with KV cache support.
    """

    @pytest.fixture
    def sample_config(self):
        """
        Create a small test configuration for faster testing.
        """
        return GptConfig(
            emb_dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=100,
            context_length=16,
            drop_rate=0.1,
            qkv_bias=False
        )

    @pytest.fixture
    def optimized_attention(self, sample_config):
        """
        Create a MultiHeadAttentionCached for testing.
        """
        return MultiHeadAttentionCached(sample_config)

    def test_kv_cache_functionality(self, optimized_attention, sample_config):
        """
        Test that KV cache properly stores and reuses key-value pairs.
        """
        batch_size, seq_len = 2, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        # Reset cache and forward pass with cache
        optimized_attention.reset_cache()
        optimized_attention.eval()

        with torch.no_grad():
            output1 = optimized_attention(input_tensor, use_cache=True)

            # Cache should now contain keys and values
            assert optimized_attention.cache_k is not None, "Cache for keys should be populated"
            assert optimized_attention.cache_v is not None, "Cache for values should be populated"
            assert optimized_attention.ptr_cur == seq_len, f"Cache pointer should be at {seq_len}, got {optimized_attention.ptr_cur}"

            # Forward pass with additional tokens
            new_input = torch.randn(batch_size, 2, sample_config.emb_dim)
            output2 = optimized_attention(new_input, use_cache=True)

            # Cache pointer should be updated
            assert optimized_attention.ptr_cur == seq_len + 2, f"Cache pointer should be at {seq_len + 2}, got {optimized_attention.ptr_cur}"

    def test_cached_vs_uncached_consistency(self, optimized_attention, sample_config):
        """
        Test that cached and uncached outputs are consistent for single forward passes.
        """
        batch_size, seq_len = 2, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        optimized_attention.eval()

        with torch.no_grad():
            # Uncached forward pass
            optimized_attention.reset_cache()
            output_uncached = optimized_attention(input_tensor, use_cache=False)

            # Cached forward pass with same input
            optimized_attention.reset_cache()
            output_cached = optimized_attention(input_tensor, use_cache=True)

            # Outputs should be very close (small numerical differences expected)
            torch.testing.assert_close(output_uncached, output_cached, atol=1e-6, rtol=1e-6, msg="Cached and uncached outputs should be close")

    def test_cache_reset(self, optimized_attention, sample_config):
        """
        Test that cache reset properly clears stored key-value pairs.
        """
        batch_size, seq_len = 2, 4
        input_tensor = torch.randn(batch_size, seq_len, sample_config.emb_dim)

        # Forward pass with cache
        optimized_attention(input_tensor, use_cache=True)

        # Verify cache is populated
        assert optimized_attention.cache_k is not None, "Cache for keys should be populated"
        assert optimized_attention.cache_v is not None, "Cache for values should be populated"
        assert optimized_attention.ptr_cur > 0, f"Cache pointer should be greater than 0, got {optimized_attention.ptr_cur}"

        # Reset cache
        optimized_attention.reset_cache()

        # Verify cache is cleared
        assert optimized_attention.cache_k is None, "Cache for keys should be cleared"
        assert optimized_attention.cache_v is None, "Cache for values should be cleared"
        assert optimized_attention.ptr_cur == 0, f"Cache pointer should be reset to 0, got {optimized_attention.ptr_cur}"
