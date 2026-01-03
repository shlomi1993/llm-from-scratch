import torch
import pytest

from torch import Tensor

from src.model.attention import MultiheadAttentionWrapper, MultiheadAttention, MultiheadAttentionCached


@pytest.fixture
def input_tensor():
    return torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )


@pytest.fixture
def input_batch(input_tensor: Tensor):
    return torch.stack((input_tensor, input_tensor), dim=0)


def test_multihead_attention_wrapper(input_batch: Tensor):
    torch.manual_seed(123)
    mha = MultiheadAttentionWrapper(d_in=input_batch.shape[2], d_out=2, context_length=input_batch.shape[1], dropout=0.0, n_heads=2)
    out = mha(input_batch)
    expected = torch.tensor(
        [[[-0.4519,  0.2216,  0.4772,  0.1063],
          [-0.5874,  0.0058,  0.5891,  0.3257],
          [-0.6300, -0.0632,  0.6202,  0.3860],
          [-0.5675, -0.0843,  0.5478,  0.3589],
          [-0.5526, -0.0981,  0.5321,  0.3428],
          [-0.5299, -0.1081,  0.5077,  0.3493]],
         [[-0.4519,  0.2216,  0.4772,  0.1063],
          [-0.5874,  0.0058,  0.5891,  0.3257],
          [-0.6300, -0.0632,  0.6202,  0.3860],
          [-0.5675, -0.0843,  0.5478,  0.3589],
          [-0.5526, -0.0981,  0.5321,  0.3428],
          [-0.5299, -0.1081,  0.5077,  0.3493]]]
    )
    assert torch.allclose(out, expected, atol=1e-4), f"MultiheadAttentionWrapper output mismatch: {out} vs {expected}"
    assert out.shape == (2, 6, 4), f"MultiheadAttentionWrapper output shape mismatch: got {out.shape}, expected (2, 6, 4)"


def test_multihead_attention_wrapper_shape():
    batch = torch.randn(8, 4, 256)
    n_heads = 2
    d_in = 256
    d_out = d_in // n_heads
    mha = MultiheadAttentionWrapper(d_in, d_out, context_length=4, dropout=0.0, n_heads=n_heads)
    out = mha(batch)
    assert out.shape == (8, 4, 256), f"MultiheadAttention output shape mismatch: got {out.shape}, expected (8, 4, 256)"


def test_multihead_attention_param_count():
    mha = MultiheadAttention(d_in=768, d_out=768, context_length=1004, dropout=0.0, n_heads=12)
    param_count = sum(p.numel() for p in mha.parameters() if p.requires_grad)
    assert param_count == 2360064, f"Parameter count mismatch: {param_count} != 2360064"


def test_multihead_attention_cached_shape():
    batch = torch.randn(4, 8, 256)  # (batch_size, context_length, d_in)
    mha_cached = MultiheadAttentionCached(d_in=256, d_out=256, context_length=8, dropout=0.0, n_heads=8)
    out = mha_cached(batch)
    assert out.shape == (4, 8, 256), f"MultiheadAttentionCached output shape mismatch: got {out.shape}, expected (4, 8, 256)"


def test_multihead_attention_cached_param_count():
    mha_cached = MultiheadAttentionCached(d_in=768, d_out=768, context_length=1024, dropout=0.0, n_heads=12)
    param_count = sum(p.numel() for p in mha_cached.parameters() if p.requires_grad)
    assert param_count == 2360064, f"MultiheadAttentionCached parameter count mismatch: {param_count} != 2360064"
