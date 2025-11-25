import pytest
import torch

from src.attention import MultiheadAttentionCached


class TestMultiHeadAttentionCached:
    """
    Test suite for MultiHeadAttentionCached implementation.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiHeadAttentionCached.
        """
        torch.manual_seed(42)
        return torch.tensor([
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]
        ])

    def test_output_shape_cached_attention(self, sample_inputs):
        """
        Test that MultiHeadAttentionCached produces correct output shape.
        """
        d_in = 12
        d_out = 12
        context_length = 8
        dropout = 0.1
        n_heads = 4

        # Expand input to match d_in=12
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 9)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_cache_functionality(self, sample_inputs):
        """
        Test that caching functionality works correctly.
        """
        d_in = 6
        d_out = 6
        context_length = 8
        dropout = 0.0
        n_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)

        # Test incremental generation simulation
        for use_cache in [False, True]:  # Test both cached and non-cached modes
            output = mha(batch, use_cache=use_cache)

            # Check output shape
            expected_shape = (batch.shape[0], batch.shape[1], d_out)
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

            # Verify output is finite
            assert torch.isfinite(output).all(), f"Output should be finite"
            assert not torch.isnan(output).any(), f"Output should not contain NaN"

    def test_gradient_flow_cached_attention(self, sample_inputs):
        """
        Test that gradients flow properly through cached attention.
        """
        d_in = 6
        d_out = 6
        context_length = 8
        dropout = 0.1
        n_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)
        output = mha(batch)

        loss = output.sum()
        loss.backward()

        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

    def test_cache_consistency(self, sample_inputs):
        """
        Test that cached and non-cached outputs are consistent.
        """
        d_in = 6
        d_out = 6
        context_length = 8
        dropout = 0.0
        n_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

        # Create two identical models
        torch.manual_seed(123)
        mha_cached = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)

        torch.manual_seed(123)
        mha_regular = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)

        # Get outputs from both models
        output_cached = mha_cached(batch, use_cache=True)
        output_regular = mha_regular(batch, use_cache=False)

        # Outputs should be very similar (allowing for small numerical differences)
        assert torch.allclose(output_cached, output_regular, atol=1e-5), "Cached and regular outputs should be nearly identical"

    def test_different_head_counts_cached(self, sample_inputs):
        """
        Test cached attention with different numbers of heads.
        """
        context_length = 8
        dropout = 0.1

        # Test different head counts with appropriate d_in/d_out values
        test_configs = [
            (1, 4),   # 1 head, d_in=d_out=4
            (2, 6),   # 2 heads, d_in=d_out=6
            (4, 8),   # 4 heads, d_in=d_out=8
        ]

        for n_heads, d_in in test_configs:
            d_out = d_in

            # Create input with appropriate dimensions
            if d_in <= 3:
                expanded_inputs = sample_inputs[:, :d_in]
            else:
                expanded_inputs = torch.cat([sample_inputs, torch.randn(6, d_in - 3)], dim=-1)
            batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

            mha = MultiheadAttentionCached(d_in, d_out, context_length, dropout, n_heads)
            output = mha(batch)

            expected_shape = (2, 6, d_out)
            assert output.shape == expected_shape, f"For {n_heads} heads, expected shape {expected_shape}, got {output.shape}"
            assert torch.isfinite(output).all(), f"Output for {n_heads} heads should be finite"
            assert not torch.isnan(output).any(), f"Output for {n_heads} heads should not contain NaN"
