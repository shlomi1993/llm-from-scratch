import torch
import pytest

from torch import Tensor

from src.model.attention import SelfAttention, CausalAttention


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


@pytest.mark.parametrize("seed, expected", [
    (123, torch.tensor([
        [-0.5337, -0.1051],
        [-0.5323, -0.1080],
        [-0.5323, -0.1079],
        [-0.5297, -0.1076],
        [-0.5311, -0.1066],
        [-0.5299, -0.1081]
    ])),
    (789, torch.tensor([
        [-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]
    ]))
])
def test_self_attention(input_tensor: Tensor, seed: int, expected: Tensor):
    d_in = input_tensor.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    torch.manual_seed(seed)
    sa = SelfAttention(d_in, d_out)
    out = sa(input_tensor)
    assert torch.allclose(out, expected, atol=1e-4), f"SelfAttention output mismatch with seed {seed}: {out} vs {expected}"


def test_causal_attention(input_batch: Tensor):
    torch.manual_seed(123)
    ca = CausalAttention(d_in=input_batch.shape[2], d_out=2, context_length=input_batch.shape[1], dropout=0.0)
    out = ca(input_batch)
    expected = torch.tensor(
        [[[-0.4519,  0.2216],
          [-0.5874,  0.0058],
          [-0.6300, -0.0632],
          [-0.5675, -0.0843],
          [-0.5526, -0.0981],
          [-0.5299, -0.1081]],
         [[-0.4519,  0.2216],
          [-0.5874,  0.0058],
          [-0.6300, -0.0632],
          [-0.5675, -0.0843],
          [-0.5526, -0.0981],
          [-0.5299, -0.1081]]]
    )
    assert torch.allclose(out, expected, atol=1e-4), f"CausalAttention output mismatch: {out} vs {expected}"
    assert out.shape == (2, 6, 2), f"CausalAttention output shape mismatch: got {out.shape}, expected (2, 6, 2)"
