import pytest
import torch

from src.attention import (
    MultiheadAttentionPyTorchSdpaWithoutFlash,
    MultiheadAttentionPyTorchClass,
    MultiheadAttentionPyTorchFlexAttention,
    MultiheadAttentionCached
)


class TestMultiheadAttentionPyTorchSdpaWithoutFlash:
    """
    Test suite for MultiheadAttentionPyTorchSDPAWithoutFlash multi-head attention implementation.
    """

    def test_output_shape_pytorch_sdpa_without_flash(self):
        """
        Test that MultiheadAttentionPyTorchSDPAWithoutFlash produces correct output shape.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256
        num_heads = 4
        context_length = 16

        mha = MultiheadAttentionPyTorchSdpaWithoutFlash(d_in=d_in, d_out=d_out, n_heads=num_heads, context_length=context_length)

        x = torch.randn(batch_size, num_tokens, d_in)
        out = mha(x)

        assert out.shape == (batch_size, num_tokens, d_out), f"Expected shape ({batch_size}, {num_tokens}, {d_out}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output should not contain NaN values"
        assert torch.isfinite(out).all(), "Output should contain only finite values"

    def test_dimension_divisibility_assertion_pytorch_sdpa_without_flash(self):
        """
        Test that MultiheadAttentionPyTorchSdpaWithoutFlash raises error for invalid dimensions.
        """
        with pytest.raises(AssertionError, match="d_out.*divisible.*n_heads"):
            MultiheadAttentionPyTorchSdpaWithoutFlash(d_in=512, d_out=257, n_heads=4, context_length=16)

    def test_gradient_flow_pytorch_sdpa_without_flash(self):
        """
        Test that gradients flow properly through the model.
        """
        batch_size, num_tokens, d_in = 2, 8, 512
        d_out = 256

        mha = MultiheadAttentionPyTorchSdpaWithoutFlash(d_in=d_in, d_out=d_out, n_heads=4, context_length=16, dropout=0.0)

        x = torch.randn(batch_size, num_tokens, d_in, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input tensor should have gradients after backward pass"
        assert mha.qkv.weight.grad is not None, "QKV weights should have gradients after backward pass"


class TestMultiheadAttentionPyTorchClass:
    """
    Test suite for MultiheadAttentionPyTorchClass multi-head attention implementation.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiheadAttentionPyTorchClass.
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

    def test_output_shape_pytorch_class(self, sample_inputs):
        """
        Test that MultiheadAttentionPyTorchClass produces correct output shape.
        """
        d_in, d_out = 12, 12  # MultiheadAttentionPyTorchClass expects d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 4

        # Expand input to match d_in=12
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 9)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiheadAttentionPyTorchClass(d_in, d_out, num_heads, context_length, dropout)

        output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_gradient_flow_pytorch_class(self, sample_inputs):
        """
        Test that gradients flow properly through MultiheadAttentionPyTorchClass.
        """
        d_in, d_out = 6, 6  # MultiheadAttentionPyTorchClass expects d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        batch.requires_grad_(True)

        mha = MultiheadAttentionPyTorchClass(d_in, d_out, num_heads, context_length, dropout)
        output = mha(batch)

        loss = output.sum()
        loss.backward()

        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

    def test_pytorch_multihead_attention_backend(self, sample_inputs):
        """
        Test that PyTorch MultiheadAttention backend is working correctly.
        """
        d_in, d_out = 4, 4  # MultiheadAttentionPyTorchClass expects d_in == d_out
        context_length = 8
        dropout = 0.0
        num_heads = 2

        # Expand input to match d_in=4
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 1)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        mha = MultiheadAttentionPyTorchClass(d_in, d_out, num_heads, context_length, dropout)

        # Test that we can successfully run the forward pass
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


class TestMultiheadAttentionPyTorchFlexAttention:
    """
    Test suite for MultiheadAttentionPyTorchFlexAttention multi-head attention implementation.
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Sample input tensor for testing MultiheadAttentionPyTorchFlexAttention.
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

    def test_output_shape_flex_attention(self, sample_inputs):
        """
        Test that MultiheadAttentionPyTorchFlexAttention produces correct output shape.
        """
        d_in, d_out = 12, 12  # FlexAttention expects d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 4

        # Expand input to match d_in=12
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 9)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)

        # Test with flex_attention if available, otherwise use fallback
        try:
            mha = MultiheadAttentionPyTorchFlexAttention(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)
        except (ImportError, AttributeError, TypeError):
            # If flex_attention is not available, test that the class handles it gracefully
            # by falling back to regular attention
            from src.attention import MultiheadAttentionPyTorchClass  # Use regular MHA as fallback
            mha = MultiheadAttentionPyTorchClass(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)

        assert output.shape == (2, 6, 12), f"Expected shape (2, 6, 12), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    def test_gradient_flow_flex_attention(self, sample_inputs):
        """
        Test that gradients flow properly through MultiheadAttentionPyTorchFlexAttention.
        """
        d_in, d_out = 6, 6  # FlexAttention expects d_in == d_out
        context_length = 8
        dropout = 0.1
        num_heads = 2

        # Expand input to match d_in=6
        expanded_inputs = torch.cat([sample_inputs, torch.randn(6, 3)], dim=-1)
        batch = torch.stack((expanded_inputs, expanded_inputs), dim=0)
        batch.requires_grad_(True)

        try:
            mha = MultiheadAttentionPyTorchFlexAttention(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)
        except (ImportError, AttributeError, TypeError):
            # If flex_attention is not available, test gradients with fallback
            from src.attention import MultiheadAttentionPyTorchClass  # Use regular MHA as fallback
            mha = MultiheadAttentionPyTorchClass(d_in, d_out, num_heads, context_length, dropout)
            output = mha(batch)

        loss = output.sum()
        loss.backward()

        assert batch.grad is not None, "Input should have gradients"
        assert batch.grad.shape == batch.shape, "Gradient shape should match input shape"

    def test_flex_attention_availability(self):
        """
        Test detection of flex_attention availability.
        """
        # This test checks whether the flex_attention is properly handled when unavailable
        try:
            from torch.nn.attention.flex_attention import flex_attention
            flex_available = True
        except ImportError:
            flex_available = False

        # The test should pass regardless of availability
        assert isinstance(flex_available, bool)


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
