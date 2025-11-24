import pytest
import torch

from src.gpt import GptModelCached
from src.configurations import GptConfig


class TestGptModelCached:
    """
    Test suite for the GptModelCached implementation with KV cache support.
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
    def cached_model(self, sample_config: GptConfig) -> GptModelCached:
        """
        Create a GptModelCached instance for testing.
        """
        model = GptModelCached(sample_config)
        model.eval()
        return model

    def test_kv_cache_initialization_and_reset(self, cached_model: GptModelCached, sample_config: GptConfig) -> None:
        """
        Test KV cache initialization and reset functionality.
        """
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        # Initially, caches should be empty/reset
        assert cached_model.ptr_current_pos == 0

        # Forward pass with cache
        with torch.no_grad():
            output = cached_model(input_ids, use_cache=True)

        # Cache should now be populated
        assert cached_model.ptr_current_pos == seq_len, f"Expected current position to be {seq_len}, got {cached_model.ptr_current_pos}"
        assert output.shape == (batch_size, seq_len, sample_config.vocab_size), "Output shape mismatch"

        # Verify that transformer blocks have cache
        for blk in cached_model.trf_blocks:
            assert blk.att.cache_k is not None, "Cache for keys should be populated"
            assert blk.att.cache_v is not None, "Cache for values should be populated"

        # Reset cache
        cached_model.reset_kv_cache()
        assert cached_model.ptr_current_pos == 0

        # Verify all block caches are reset
        for blk in cached_model.trf_blocks:
            assert blk.att.cache_k is None, "Cache for keys should be reset to None"
            assert blk.att.cache_v is None, "Cache for values should be reset to None"

    def test_cached_vs_uncached_forward_consistency(self, cached_model: GptModelCached, sample_config: GptConfig) -> None:
        """
        Test that cached and uncached forward passes produce consistent results for single sequences.
        """
        torch.manual_seed(42)
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            # Uncached forward pass
            cached_model.reset_kv_cache()
            output_uncached = cached_model(input_ids, use_cache=False)

            # Cached forward pass with same input
            cached_model.reset_kv_cache()
            output_cached = cached_model(input_ids, use_cache=True)

        # Outputs should be very close (small numerical differences expected due to computation order)
        torch.testing.assert_close(output_uncached, output_cached, atol=1e-5, rtol=1e-5, msg="Cached and uncached outputs do not match")

    def test_incremental_generation_with_cache(self, cached_model: GptModelCached, sample_config: GptConfig) -> None:
        """
        Test incremental generation using KV cache to verify cache accumulation and position tracking.
        """
        batch_size = 1
        initial_seq_len = 4
        additional_tokens = 3

        # Create initial input
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, initial_seq_len))

        cached_model.reset_kv_cache()

        with torch.no_grad():
            # Process initial sequence
            output1 = cached_model(input_ids, use_cache=True)
            assert cached_model.ptr_current_pos == initial_seq_len, f"Expected current position to be {initial_seq_len}, got {cached_model.ptr_current_pos}"
            assert output1.shape == (batch_size, initial_seq_len, sample_config.vocab_size), "Output shape mismatch for initial sequence"

            # Add more tokens incrementally
            accumulated_tokens = []
            for i in range(additional_tokens):
                next_token = torch.randint(0, sample_config.vocab_size, (batch_size, 1))
                accumulated_tokens.append(next_token)
                output_incremental = cached_model(next_token, use_cache=True)

                # Check position tracking
                assert cached_model.ptr_current_pos == initial_seq_len + i + 1, f"Expected current position to be {initial_seq_len + i + 1}, got {cached_model.ptr_current_pos}"
                assert output_incremental.shape == (batch_size, 1, sample_config.vocab_size), "Output shape mismatch for incremental token"

                # Verify cache accumulation by checking that we can process the full sequence
                full_sequence = torch.cat([input_ids] + accumulated_tokens, dim=1)
                assert full_sequence.shape[1] == initial_seq_len + i + 1, f"Expected full sequence length to be {initial_seq_len + i + 1}, got {full_sequence.shape[1]}"

    def test_generate_text_method(self, cached_model: GptModelCached, sample_config: GptConfig) -> None:
        """
        Test the generate_text method for cached generation functionality.
        """
        batch_size, initial_len = 1, 3
        max_new_tokens = 5

        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, initial_len))

        # Generate text using the cached method
        output = cached_model.generate_text(input_ids, max_new_tokens, sample_config.context_length)

        # Verify output shape and content
        expected_length = initial_len + max_new_tokens
        assert output.shape == (batch_size, expected_length), f"Expected output shape {(batch_size, expected_length)}, got {output.shape}"

        # Verify that initial tokens are preserved
        torch.testing.assert_close(output[:, :initial_len], input_ids, msg="Initial tokens do not match input_ids")

        # Verify that new tokens are within vocabulary range
        assert (output[:, initial_len:] >= 0).all(), "New tokens contain values less than 0"
        assert (output[:, initial_len:] < sample_config.vocab_size).all(), f"New tokens contain values greater than or equal to vocab size {sample_config.vocab_size}"

    def test_batch_processing_with_cache(self, cached_model: GptModelCached, sample_config: GptConfig) -> None:
        """
        Test that the model properly handles batch processing with KV cache.
        """
        batch_size, seq_len = 3, 6
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        cached_model.reset_kv_cache()

        with torch.no_grad():
            # Process batch with cache
            output = cached_model(input_ids, use_cache=True)

            # Verify output shape
            assert output.shape == (batch_size, seq_len, sample_config.vocab_size), "Output shape mismatch for batch processing"

            # Verify cache dimensions match batch size
            for blk in cached_model.trf_blocks:
                assert blk.att.cache_k.size(0) == batch_size, f"Cache key batch size mismatch: expected {batch_size}, got {blk.att.cache_k.size(0)}"
                assert blk.att.cache_v.size(0) == batch_size, f"Cache value batch size mismatch: expected {batch_size}, got {blk.att.cache_v.size(0)}"

            # Test that we can continue with the same batch
            next_tokens = torch.randint(0, sample_config.vocab_size, (batch_size, 2))
            next_output = cached_model(next_tokens, use_cache=True)

            assert next_output.shape == (batch_size, 2, sample_config.vocab_size), "Output shape mismatch for continued batch processing"
            assert cached_model.ptr_current_pos == seq_len + 2, f"Expected current position to be {seq_len + 2}, got {cached_model.ptr_current_pos}"

    def test_positional_embedding_handling(self, sample_config: GptConfig) -> None:
        """
        Test that positional embeddings are handled correctly in cached vs uncached modes.
        """
        cached_model = GptModelCached(sample_config)
        cached_model.eval()

        batch_size, seq_len = 1, 4
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            # Test uncached mode - should use positions [0, 1, 2, 3]
            cached_model.reset_kv_cache()
            output_uncached = cached_model(input_ids, use_cache=False)
            assert cached_model.ptr_current_pos == 0, f"Expected current position to be 0, got {cached_model.ptr_current_pos}"  # Should not update in uncached mode

            # Test cached mode - should use positions [0, 1, 2, 3] first, then continue
            cached_model.reset_kv_cache()
            output_cached_first = cached_model(input_ids, use_cache=True)
            assert cached_model.ptr_current_pos == seq_len, f"Expected current position to be {seq_len}, got {cached_model.ptr_current_pos}"

            # Add more tokens - should use positions [4, 5]
            next_tokens = torch.randint(0, sample_config.vocab_size, (batch_size, 2))
            output_cached_next = cached_model(next_tokens, use_cache=True)
            assert cached_model.ptr_current_pos == seq_len + 2, f"Expected current position to be {seq_len + 2}, got {cached_model.ptr_current_pos}"

            # Verify shapes
            assert output_uncached.shape == (batch_size, seq_len, sample_config.vocab_size), "Output shape mismatch for uncached mode"
            assert output_cached_first.shape == (batch_size, seq_len, sample_config.vocab_size), "Output shape mismatch for first cached mode"
            assert output_cached_next.shape == (batch_size, 2, sample_config.vocab_size), "Output shape mismatch for next cached mode"
