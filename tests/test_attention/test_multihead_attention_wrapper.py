import pytest
import torch

from attention import MultiHeadAttentionWrapper


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
