import pytest
import torch
import torch.nn as nn

from src.configurations import GptConfig, GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from src.transformer import TransformerBlock


def _att_param_count(emb_dim: int, qkv_bias: bool) -> int:
    # W_query/W_key/W_value: (emb_dim x emb_dim) each + optional bias
    # out_proj: (emb_dim x emb_dim) + bias
    w = emb_dim * emb_dim
    qkv_bias_params = emb_dim if qkv_bias else 0
    return 3 * (w + qkv_bias_params) + (w + emb_dim)


def _ff_param_count(emb_dim: int) -> int:
    # Linear(emb -> 4emb): weights (4emb x emb) + bias (4emb)
    # Linear(4emb -> emb): weights (emb x 4emb) + bias (emb)
    return (4 * emb_dim * emb_dim + 4 * emb_dim) + (4 * emb_dim * emb_dim + emb_dim)


def _gpt2_total_params(cfg: GptConfig) -> int:
    vocab, ctx, emb = cfg.vocab_size, cfg.context_length, cfg.emb_dim
    tok_emb = vocab * emb
    pos_emb = ctx * emb

    # norm1 + norm2, each has weight + bias => 2 * emb; so 2 norms => 4 * emb
    per_block = _att_param_count(emb, cfg.qkv_bias) + _ff_param_count(emb) + 4 * emb
    final_ln = 2 * emb
    lm_head = vocab * emb

    return tok_emb + pos_emb + cfg.n_layers * per_block + final_ln + lm_head


def _gpt2_trainable_params_weight_tied(cfg: GptConfig) -> int:
    # weight tying removes the separate lm_head weight tensor (vocab * emb)
    return _gpt2_total_params(cfg) - (cfg.vocab_size * cfg.emb_dim)


def _model_size_mb_from_params(total_params: int) -> float:
    # float32 => 4 bytes per parameter
    return (total_params * 4) / (1024 ** 2)


@pytest.mark.parametrize("cfg", [GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M], ids=["124M", "355M", "774M", "1558M"])
def test_transformer_block_forward_shape(cfg: GptConfig) -> None:
    torch.manual_seed(123)
    x = torch.rand(2, 4, cfg.emb_dim)
    block = TransformerBlock(cfg)
    y = block(x)
    assert y.shape == x.shape, f"Output shape mismatch for emb_dim={cfg.emb_dim}: expected {x.shape}, got {y.shape}"


def test_transformer_block_structure_and_param_counts_124m() -> None:
    block = TransformerBlock(GPT_CONFIG_124M)

    for attr in ("att", "ff", "norm1", "norm2", "drop_shortcut"):
        assert hasattr(block, attr), f"TransformerBlock missing attribute '{attr}'"

    assert isinstance(block.drop_shortcut, nn.Dropout), f"drop_shortcut expected nn.Dropout, got {type(block.drop_shortcut)}"
    assert block.drop_shortcut.p == pytest.approx(0.1), f"drop_shortcut.p expected 0.1, got {block.drop_shortcut.p}"

    # Attention inner layers (names per your expected structure)
    for name in ("W_query", "W_key", "W_value", "out_proj"):
        assert hasattr(block.att, name), f"Attention module missing '{name}'"

    Wq, Wk, Wv, Wo = (
        block.att.W_query,
        block.att.W_key,
        block.att.W_value,
        block.att.out_proj,
    )

    for layer, lname in zip((Wq, Wk, Wv, Wo), ("W_query", "W_key", "W_value", "out_proj")):
        assert isinstance(layer, nn.Linear), f"{lname} expected nn.Linear, got {type(layer)}"
        assert (layer.in_features, layer.out_features) == (768, 768), f"{lname} expected (768→768), got ({layer.in_features} → {layer.out_features})"

    assert Wq.bias is None, "W_query should not have bias (qkv_bias=False)"
    assert Wk.bias is None, "W_key should not have bias (qkv_bias=False)"
    assert Wv.bias is None, "W_value should not have bias (qkv_bias=False)"
    assert Wo.bias is not None, "out_proj should have bias=True"

    ff_params = sum(p.numel() for p in block.ff.parameters())
    att_params = sum(p.numel() for p in block.att.parameters())

    assert ff_params == 4_722_432, f"FeedForward param count mismatch: expected 4,722,432, got {ff_params}"
    assert att_params == 2_360_064, f"Attention param count mismatch: expected 2,360,064, got {att_params}"

    # Helpful context if repr differs slightly (but still sanity-check it contains key parts)
    s = str(block)
    assert "TransformerBlock" in s, f"repr missing 'TransformerBlock':\n{s}"
    assert "FeedForward" in s, f"repr missing 'FeedForward':\n{s}"
    assert "W_query" in s and "W_key" in s and "W_value" in s and "out_proj" in s, f"repr missing attention layer names:\n{s}"


@pytest.mark.parametrize("cfg", [GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M], ids=["124M", "355M", "774M", "1558M"],)
def test_transformer_block_param_counts_match_formulas(cfg: GptConfig) -> None:
    block = TransformerBlock(cfg)

    att = sum(p.numel() for p in block.att.parameters())
    ff = sum(p.numel() for p in block.ff.parameters())

    expected_att = _att_param_count(cfg.emb_dim, cfg.qkv_bias)
    expected_ff = _ff_param_count(cfg.emb_dim)

    assert att == expected_att, f"Attention params mismatch for emb_dim={cfg.emb_dim}: expected {expected_att}, got {att}"
    assert ff == expected_ff, f"FeedForward params mismatch for emb_dim={cfg.emb_dim}: expected {expected_ff}, got {ff}"
