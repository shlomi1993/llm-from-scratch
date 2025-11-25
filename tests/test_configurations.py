"""
Test Suite for GPT Configuration Module

This module contains comprehensive tests for the GptConfig dataclass and predefined model configurations, testing
parameter validation, configuration instantiation, and preset model configurations.
"""

import pytest
import pickle
import json
import copy

from dataclasses import FrozenInstanceError

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M


class TestGptConfig:
    """
    Test suite for the GptConfig dataclass.
    """

    def test_default_config_creation(self):
        """
        Test creating a configuration with standard values.
        """
        config = GPT_CONFIG_124M

        # Check standard GPT-2 124M values
        assert config.emb_dim == 768, "GPT-124M embedding dimension should be 768"
        assert config.n_layers == 12, "GPT-124M number of layers should be 12"
        assert config.n_heads == 12, "GPT-124M number of heads should be 12"
        assert config.vocab_size == 50257, "GPT-124M vocabulary size should be 50257"
        assert config.context_length == 1024, "GPT-124M context length should be 1024"
        assert config.drop_rate == 0.1, "GPT-124M dropout rate should be 0.1"
        assert config.qkv_bias is False, "GPT-124M qkv_bias should be False"

    def test_custom_config_creation(self):
        """
        Test creating a configuration with custom values.
        """
        config = GptConfig(
            emb_dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=30000,
            context_length=2048,
            drop_rate=0.2,
            qkv_bias=True
        )

        assert config.emb_dim == 512, "Custom embedding dimension should be 512"
        assert config.n_layers == 8, "Custom number of layers should be 8"
        assert config.n_heads == 8, "Custom number of heads should be 8"
        assert config.vocab_size == 30000, "Custom vocabulary size should be 30000"
        assert config.context_length == 2048, "Custom context length should be 2048"
        assert config.drop_rate == 0.2, "Custom dropout rate should be 0.2"
        assert config.qkv_bias is True, "Custom qkv_bias should be True"

    def test_config_immutability(self):
        """
        Test that configuration is frozen (immutable).
        """
        config = GptConfig(emb_dim=768, n_layers=12, n_heads=12)

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            config.emb_dim = 1024

        with pytest.raises(FrozenInstanceError):
            config.n_layers = 24

        with pytest.raises(FrozenInstanceError):
            config.vocab_size = 60000

    def test_config_equality(self):
        """
        Test configuration equality comparison.
        """
        config1 = GptConfig(emb_dim=512, n_layers=8, n_heads=8)
        config2 = GptConfig(emb_dim=512, n_layers=8, n_heads=8)
        config3 = GptConfig(emb_dim=768, n_layers=8, n_heads=8)

        assert config1 == config2, "Configurations with same parameters should be equal"
        assert config1 != config3, "Configurations with different parameters should not be equal"

    def test_config_hash(self):
        """
        Test that configuration can be hashed (useful for caching).
        """
        config1 = GptConfig(emb_dim=512, n_layers=8, n_heads=8)
        config2 = GptConfig(emb_dim=512, n_layers=8, n_heads=8)
        config3 = GptConfig(emb_dim=768, n_layers=8, n_heads=8)

        # Same configurations should have same hash
        assert hash(config1) == hash(config2), "Same configurations should have same hash"

        # Different configurations should have different hash (usually)
        assert hash(config1) != hash(config3), "Different configurations should have different hash"

        # Should be usable in sets and as dict keys
        config_set = {config1, config2, config3}
        assert len(config_set) == 2, "Set should contain only unique configurations"

        config_dict = {config1: "model1", config3: "model2"}
        assert len(config_dict) == 2, "Dictionary should accept configurations as keys"

    def test_attention_head_dimension_consistency(self):
        """
        Test that embedding dimension is divisible by number of attention heads.
        """
        # Valid configurations
        valid_configs = [
            GptConfig(emb_dim=768, n_layers=12, n_heads=12),  # 768 / 12 = 64
            GptConfig(emb_dim=512, n_layers=8, n_heads=8),   # 512 / 8 = 64
            GptConfig(emb_dim=1024, n_layers=16, n_heads=16), # 1024 / 16 = 64
        ]

        for config in valid_configs:
            head_dim = config.emb_dim // config.n_heads
            assert config.emb_dim % config.n_heads == 0, f"Embedding dim {config.emb_dim} should be divisible by n_heads {config.n_heads}"
            assert head_dim > 0, f"Head dimension should be positive, got {head_dim}"


class TestPredefinedConfigurations:
    """
    Test suite for predefined GPT model configurations.
    """

    def test_gpt_config_124m(self):
        """
        Test GPT-2 124M parameter configuration.
        """
        config = GPT_CONFIG_124M

        assert config.emb_dim == 768, "GPT-124M should have 768 embedding dimensions"
        assert config.n_layers == 12, "GPT-124M should have 12 layers"
        assert config.n_heads == 12, "GPT-124M should have 12 attention heads"
        assert config.vocab_size == 50257, "GPT-124M should have 50257 vocabulary size"
        assert config.context_length == 1024, "GPT-124M should have 1024 context length"
        assert config.drop_rate == 0.1, "GPT-124M should have 0.1 dropout rate"
        assert config.qkv_bias is False, "GPT-124M should have no QKV bias"

    def test_gpt_config_355m(self):
        """
        Test GPT-2 355M parameter configuration.
        """
        config = GPT_CONFIG_355M

        assert config.emb_dim == 1024, "GPT-355M should have 1024 embedding dimensions"
        assert config.n_layers == 24, "GPT-355M should have 24 layers"
        assert config.n_heads == 16, "GPT-355M should have 16 attention heads"
        assert config.vocab_size == 50257, "GPT-355M should have 50257 vocabulary size"
        assert config.context_length == 1024, "GPT-355M should have 1024 context length"
        assert config.drop_rate == 0.1, "GPT-355M should have 0.1 dropout rate"
        assert config.qkv_bias is False, "GPT-355M should have no QKV bias"

    def test_gpt_config_774m(self):
        """
        Test GPT-2 774M parameter configuration.
        """
        config = GPT_CONFIG_774M

        assert config.emb_dim == 1280, "GPT-774M should have 1280 embedding dimensions"
        assert config.n_layers == 36, "GPT-774M should have 36 layers"
        assert config.n_heads == 20, "GPT-774M should have 20 attention heads"
        assert config.vocab_size == 50257, "GPT-774M should have 50257 vocabulary size"
        assert config.context_length == 1024, "GPT-774M should have 1024 context length"
        assert config.drop_rate == 0.1, "GPT-774M should have 0.1 dropout rate"
        assert config.qkv_bias is False, "GPT-774M should have no QKV bias"

    def test_gpt_config_1558m(self):
        """
        Test GPT-2 1558M parameter configuration.
        """
        config = GPT_CONFIG_1558M

        assert config.emb_dim == 1600, "GPT-1558M should have 1600 embedding dimensions"
        assert config.n_layers == 48, "GPT-1558M should have 48 layers"
        assert config.n_heads == 25, "GPT-1558M should have 25 attention heads"
        assert config.vocab_size == 50257, "GPT-1558M should have 50257 vocabulary size"
        assert config.context_length == 1024, "GPT-1558M should have 1024 context length"
        assert config.drop_rate == 0.1, "GPT-1558M should have 0.1 dropout rate"
        assert config.qkv_bias is False, "GPT-1558M should have no QKV bias"

    def test_all_configs_immutable(self):
        """
        Test that all predefined configurations are immutable.
        """
        configs = [GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M]

        for i, config in enumerate(configs):
            with pytest.raises(FrozenInstanceError):
                config.emb_dim = 999
            with pytest.raises(FrozenInstanceError):
                config.n_layers = 999

    def test_configs_head_dimension_consistency(self):
        """
        Test that all predefined configurations have consistent head dimensions.
        """
        configs = [
            ("GPT_CONFIG_124M", GPT_CONFIG_124M),
            ("GPT_CONFIG_355M", GPT_CONFIG_355M),
            ("GPT_CONFIG_774M", GPT_CONFIG_774M),
            ("GPT_CONFIG_1558M", GPT_CONFIG_1558M),
        ]

        for name, config in configs:
            head_dim = config.emb_dim // config.n_heads
            assert config.emb_dim % config.n_heads == 0, f"{name}: emb_dim {config.emb_dim} should be divisible by n_heads {config.n_heads}"
            assert head_dim >= 32, f"{name}: head dimension {head_dim} should be at least 32 for good performance"

    def test_configs_scale_progression(self):
        """
        Test that configurations scale appropriately from small to large.
        """
        configs = [GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M]

        # Check that embedding dimensions increase
        emb_dims = [config.emb_dim for config in configs]
        assert emb_dims == sorted(emb_dims), "Embedding dimensions should increase with model size"

        # Check that number of layers increase
        n_layers = [config.n_layers for config in configs]
        assert n_layers == sorted(n_layers), "Number of layers should increase with model size"

        # Check that number of heads increase
        n_heads = [config.n_heads for config in configs]
        assert n_heads == sorted(n_heads), "Number of heads should increase with model size"

        # All should have same vocab_size and context_length (GPT-2 standard)
        vocab_sizes = [config.vocab_size for config in configs]
        context_lengths = [config.context_length for config in configs]
        drop_rates = [config.drop_rate for config in configs]
        qkv_biases = [config.qkv_bias for config in configs]

        assert len(set(vocab_sizes)) == 1, "All configs should have same vocab_size"
        assert len(set(context_lengths)) == 1, "All configs should have same context_length"
        assert len(set(drop_rates)) == 1, "All configs should have same drop_rate"
        assert len(set(qkv_biases)) == 1, "All configs should have same qkv_bias setting"

    def test_config_parameter_estimation(self):
        """
        Test approximate parameter count estimation for configurations.
        """
        def estimate_params(config):
            """Rough parameter estimation for GPT model."""
            # Token embedding: vocab_size * emb_dim
            tok_emb_params = config.vocab_size * config.emb_dim

            # Position embedding: context_length * emb_dim
            pos_emb_params = config.context_length * config.emb_dim

            # Transformer blocks: rough approximation
            # Each block has attention (4 * emb_dim^2) + FFN (8 * emb_dim^2) + norms
            block_params = config.n_layers * (12 * config.emb_dim * config.emb_dim + 4 * config.emb_dim)

            # Output head: emb_dim * vocab_size
            out_head_params = config.emb_dim * config.vocab_size

            return tok_emb_params + pos_emb_params + block_params + out_head_params

        # Test that estimated parameters are in reasonable ranges
        config_sizes = [
            (GPT_CONFIG_124M, 160_000_000, 170_000_000),        # ~163M actual
            (GPT_CONFIG_355M, 400_000_000, 410_000_000),        # ~406M actual
            (GPT_CONFIG_774M, 835_000_000, 845_000_000),        # ~838M actual
            (GPT_CONFIG_1558M, 1_630_000_000, 1_645_000_000),   # ~1637M actual
        ]

        for config, min_params, max_params in config_sizes:
            estimated = estimate_params(config)
            assert min_params <= estimated <= max_params, f"Estimated params {estimated:,} should be between {min_params:,} and {max_params:,}"

    def test_config_serialization_compatibility(self):
        """
        Test that configurations can be serialized and deserialized.
        """

        config = GPT_CONFIG_124M

        # Test pickle serialization
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        assert unpickled == config, "Pickled config should be identical"

        # Test JSON serialization
        config_dict = vars(config)
        json_str = json.dumps(config_dict)
        loaded_dict = json.loads(json_str)
        reconstructed = GptConfig(**loaded_dict)
        assert reconstructed == config, "JSON-reconstructed config should be identical"

    def test_config_copy_behavior(self):
        """
        Test configuration copying behavior.
        """

        # Shallow copy
        shallow_copy = copy.copy(GPT_CONFIG_124M)
        assert shallow_copy == GPT_CONFIG_124M, "Shallow copy should be equal"
        assert shallow_copy is not GPT_CONFIG_124M, "Shallow copy should be different object"

        # Deep copy
        deep_copy = copy.deepcopy(GPT_CONFIG_124M)
        assert deep_copy == GPT_CONFIG_124M, "Deep copy should be equal"
        assert deep_copy is not GPT_CONFIG_124M, "Deep copy should be different object"
