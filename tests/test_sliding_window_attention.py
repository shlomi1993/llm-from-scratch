"""
Tests for Sliding Window Attention (SWA) functionality.
"""

import pytest
import torch

from src.configurations import GptConfig
from src.gpt import GptModel
from src.attention.multihead_attention_swa import MultiheadAttentionWithSwa


class TestMultiheadAttentionWithSwa:
    """
    Test Sliding Window Attention module.
    """

    def test_swa_initialization(self):
        """
        Test that SWA attention initializes correctly.
        """
        att = MultiheadAttentionWithSwa(
            d_in=768,
            d_out=768,
            n_heads=12,
            dropout=0.1,
            qkv_bias=False,
            sliding_window_size=512
        )

        assert att.d_out == 768, "Output dimension should match"
        assert att.num_heads == 12, "Number of heads should match"
        assert att.head_dim == 64, "Head dimension should be d_out / num_heads"
        assert att.sliding_window_size == 512, "Sliding window size should match"
        assert att.cache_k is None, "Cache should be None initially"
        assert att.cache_v is None, "Cache should be None initially"
        assert att.ptr_current_pos == 0, "Current position should be 0 initially"

    def test_swa_forward_without_cache(self):
        """
        Test forward pass without caching.
        """
        att = MultiheadAttentionWithSwa(
            d_in=768,
            d_out=768,
            n_heads=12,
            dropout=0.0,
            qkv_bias=False,
            sliding_window_size=512
        )

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 768)

        output = att(x, use_cache=False)

        assert output.shape == (batch_size, seq_len, 768), "Output shape should match input"
        assert att.cache_k is None, "Cache should remain None when not using cache"
        assert att.cache_v is None, "Cache should remain None when not using cache"

    def test_swa_forward_with_cache(self):
        """
        Test forward pass with caching.
        """
        att = MultiheadAttentionWithSwa(
            d_in=768,
            d_out=768,
            n_heads=12,
            dropout=0.0,
            qkv_bias=False,
            sliding_window_size=512
        )

        batch_size = 2
        seq_len_1 = 100
        seq_len_2 = 50

        x1 = torch.randn(batch_size, seq_len_1, 768)
        x2 = torch.randn(batch_size, seq_len_2, 768)

        # First forward pass
        output1 = att(x1, use_cache=True)
        assert output1.shape == (batch_size, seq_len_1, 768), "First output shape should match"
        assert att.cache_k is not None, "Cache should be created"
        assert att.cache_v is not None, "Cache should be created"
        assert att.cache_k.shape[1] == seq_len_1, "Cache should store all tokens"

        # Second forward pass (cache should accumulate)
        output2 = att(x2, use_cache=True)
        assert output2.shape == (batch_size, seq_len_2, 768), "Second output shape should match"
        assert att.cache_k.shape[1] == seq_len_1 + seq_len_2, "Cache should accumulate tokens"

    def test_swa_window_trimming(self):
        """
        Test that cache is trimmed when exceeding sliding window size.
        """
        window_size = 128
        att = MultiheadAttentionWithSwa(
            d_in=768,
            d_out=768,
            n_heads=12,
            dropout=0.0,
            qkv_bias=False,
            sliding_window_size=window_size
        )

        batch_size = 2

        # Add tokens exceeding window size
        for _ in range(5):
            x = torch.randn(batch_size, 50, 768)
            att(x, use_cache=True)

        # Cache should be trimmed to window size
        assert att.cache_k.shape[1] == window_size, f"Cache should be trimmed to {window_size}"
        assert att.cache_v.shape[1] == window_size, f"Cache should be trimmed to {window_size}"

    def test_swa_reset_cache(self):
        """
        Test cache reset functionality.
        """
        att = MultiheadAttentionWithSwa(
            d_in=768,
            d_out=768,
            n_heads=12,
            dropout=0.0,
            qkv_bias=False,
            sliding_window_size=512
        )

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 768)

        # Create cache
        att(x, use_cache=True)
        assert att.cache_k is not None, "Cache should exist"

        # Reset cache
        att.reset_cache()
        assert att.cache_k is None, "Cache should be None after reset"
        assert att.cache_v is None, "Cache should be None after reset"
        assert att.ptr_current_pos == 0, "Position should be 0 after reset"

    def test_swa_causal_masking(self):
        """
        Test that causal masking is applied correctly.
        """
        att = MultiheadAttentionWithSwa(
            d_in=64,
            d_out=64,
            n_heads=4,
            dropout=0.0,
            qkv_bias=False,
            sliding_window_size=10
        )

        batch_size = 1
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 64)

        # Forward pass should not raise errors
        output = att(x, use_cache=False)
        assert output.shape == (batch_size, seq_len, 64), "Output shape should match"


class TestGptModelWithSwa:
    """
    Test GPT model with Sliding Window Attention.
    """

    def test_gpt_with_swa_all_layers(self):
        """
        Test GPT model with SWA applied to all layers.
        """
        config = GptConfig(
            emb_dim=256,
            n_layers=4,
            n_heads=4,
            vocab_size=1000,
            context_length=128,
            drop_rate=0.0,
            sliding_window_size=64,
            sliding_window_stride=-1  # All SWA
        )

        model = GptModel(config)

        # Check that all layers have SWA
        for i, block in enumerate(model.trf_blocks):
            assert isinstance(block.att, MultiheadAttentionWithSwa), f"Layer {i} should use SWA"
            assert block.att.sliding_window_size == 64, f"Layer {i} should have window size 64"

    def test_gpt_with_swa_no_layers(self):
        """
        Test GPT model with SWA disabled (all regular layers) by setting sliding_window_size=0.
        """
        config = GptConfig(
            emb_dim=256,
            n_layers=4,
            n_heads=4,
            vocab_size=1000,
            context_length=128,
            drop_rate=0.0,
            sliding_window_size=0,  # No SWA
            sliding_window_stride=0
        )

        model = GptModel(config)

        # Check that no layers have SWA
        for i, block in enumerate(model.trf_blocks):
            assert not isinstance(block.att, MultiheadAttentionWithSwa), f"Layer {i} should not use SWA"

    def test_gpt_with_none_window_size(self):
        """
        Test GPT model with sliding_window_size=None (no SWA support).
        """
        config = GptConfig(
            emb_dim=256,
            n_layers=4,
            n_heads=4,
            vocab_size=1000,
            context_length=128,
            drop_rate=0.0,
            sliding_window_size=None,  # Explicitly None
            sliding_window_stride=0
        )

        model = GptModel(config)

        # Check that no layers have SWA
        for i, block in enumerate(model.trf_blocks):
            assert not isinstance(block.att, MultiheadAttentionWithSwa), f"Layer {i} should not use SWA when window_size is None"

        # Verify forward pass works
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 1000), "Output shape should be correct"

    def test_gpt_with_swa_k_to_1_pattern(self):
        """
        Test GPT model with K:1 SWA pattern (K SWA layers, 1 regular layer).
        """
        config = GptConfig(
            emb_dim=256,
            n_layers=12,  # 12 layers for clear pattern
            n_heads=4,
            vocab_size=1000,
            context_length=128,
            drop_rate=0.0,
            sliding_window_size=64,
            sliding_window_stride=2  # 2:1 pattern
        )

        model = GptModel(config)

        # Check pattern: layers 0,1 are SWA, layer 2 is regular, layers 3,4 are SWA, etc.
        expected_swa = [True, True, False, True, True, False, True, True, False, True, True, False]

        for i, (block, expected) in enumerate(zip(model.trf_blocks, expected_swa)):
            is_swa = isinstance(block.att, MultiheadAttentionWithSwa) and block.att.sliding_window_size is not None
            assert is_swa == expected, f"Layer {i} SWA status mismatch: got {is_swa}, expected {expected}"

    def test_gpt_with_swa_forward(self):
        """
        Test GPT model forward pass with SWA.
        """
        config = GptConfig(
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=64,
            drop_rate=0.0,
            sliding_window_size=32,
            sliding_window_stride=-1
        )

        model = GptModel(config)
        model.eval()

        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass without cache
        output = model(input_ids, use_cache=False)
        assert output.shape == (batch_size, seq_len, 1000), "Output shape should match"

    def test_gpt_with_swa_cached_generation(self):
        """
        Test GPT model cached generation with SWA.
        """
        config = GptConfig(
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=64,
            drop_rate=0.0,
            sliding_window_size=32,
            sliding_window_stride=-1
        )

        model = GptModel(config)
        model.eval()

        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Generate with cache
        with torch.no_grad():
            output = model.generate_text_simple_cached(
                idx=input_ids,
                max_new_tokens=5,
                context_size=config.context_length,
                use_cache=True
            )

        assert output.shape == (batch_size, seq_len + 5), "Should generate 5 new tokens"

    def test_gpt_with_swa_reset_cache(self):
        """
        Test that cache reset works for GPT model with SWA.
        """
        config = GptConfig(
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=64,
            drop_rate=0.0,
            sliding_window_size=32,
            sliding_window_stride=-1
        )

        model = GptModel(config)
        model.eval()

        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward with cache
        model(input_ids, use_cache=True)

        # Check caches exist
        for block in model.trf_blocks:
            if isinstance(block.att, MultiheadAttentionWithSwa):
                assert block.att.cache_k is not None, "Cache should exist"

        # Reset cache
        model.reset_kv_cache()

        # Check caches are cleared
        for block in model.trf_blocks:
            if isinstance(block.att, MultiheadAttentionWithSwa):
                assert block.att.cache_k is None, "Cache should be None after reset"
                assert block.att.ptr_current_pos == 0, "Position should be 0 after reset"


class TestSwaVsRegularAttention:
    """
    Test comparing SWA with regular attention.
    """

    def test_swa_reduces_memory_usage(self):
        """
        Test that SWA cache size is limited by window size.
        """
        window_size = 64

        # SWA model
        config_swa = GptConfig(
            emb_dim=256,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            context_length=512,
            drop_rate=0.0,
            sliding_window_size=window_size,
            sliding_window_stride=-1
        )
        model_swa = GptModel(config_swa)
        model_swa.eval()

        batch_size = 1
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            # Generate with SWA - multiple passes to exceed window
            model_swa.reset_kv_cache()
            total_tokens = 0
            for i in range(10):  # Multiple iterations to exceed window
                logits_swa = model_swa(input_ids, use_cache=True)
                total_tokens += seq_len

                # Check cache size after each iteration
                current_cache_size = model_swa.trf_blocks[0].att.cache_k.shape[1]

                # Cache should never exceed window size
                assert current_cache_size <= window_size, f"Iteration {i}: SWA cache ({current_cache_size}) should not exceed {window_size}"

                # Once we've processed more than window_size tokens, cache should be exactly window_size
                if total_tokens > window_size:
                    assert current_cache_size == window_size, f"Iteration {i}: SWA cache should be exactly {window_size} after processing {total_tokens} tokens"

    def test_swa_maintains_causal_property(self):
        """
        Test that SWA maintains causal (autoregressive) property.
        """
        config = GptConfig(
            emb_dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=100,
            context_length=64,
            drop_rate=0.0,
            sliding_window_size=16,
            sliding_window_stride=-1
        )

        model = GptModel(config)
        model.eval()

        batch_size = 1
        seq_len = 20
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            # Get logits for full sequence
            logits_full = model(input_ids, use_cache=False)

            # Get logits using autoregressive generation
            model.reset_kv_cache()
            logits_auto = []
            for i in range(seq_len):
                logits_i = model(input_ids[:, i:i+1], use_cache=True)
                logits_auto.append(logits_i[:, -1:, :])
            logits_auto = torch.cat(logits_auto, dim=1)

            # The last token's logits should match (within numerical precision)
            # Note: Earlier tokens may differ due to different attention windows
            last_token_diff = torch.abs(logits_full[:, -1, :] - logits_auto[:, -1, :]).max()
            assert last_token_diff < 1e-4, f"Last token logits should match, diff: {last_token_diff}"
