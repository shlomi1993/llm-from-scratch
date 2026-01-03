import torch

from src.model.activation import GELU


def test_gelu_forward_basic():
    gelu = GELU()
    x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    out = gelu(x)
    expected = torch.nn.functional.gelu(x)
    assert torch.allclose(out, expected, atol=2e-4), f"GELU output mismatch: {out} vs {expected}"

def test_gelu_shape_preservation():
    gelu = GELU()
    x = torch.randn(3, 4, 5)
    out = gelu(x)
    assert out.shape == x.shape, f"GELU should preserve input shape, got {out.shape} vs {x.shape}"

def test_gelu_backward():
    gelu = GELU()
    x = torch.randn(5, requires_grad=True)
    out = gelu(x)
    grad_output = torch.ones_like(x)
    out.backward(grad_output)
    assert x.grad is not None, "GELU should produce gradients for input"
    assert torch.all(torch.isfinite(x.grad)), "Gradients should be finite"
