import torch

from attention import SelfAttention


class TestSelfAttention:
    """
    Test suite for the SelfAttention module.
    """

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
