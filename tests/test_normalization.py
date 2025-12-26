import pytest
import torch

from src.normalization import LayerNorm


def test_layernorm_output_mean_and_var_are_zero_and_one() -> None:
    torch.manual_seed(123)
    batch_example = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 0.0, -10.0, 5.0, 2.0]], dtype=torch.float32)

    ln = LayerNorm(emb_dim=5)

    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6, rtol=0.0), f"LayerNorm mean should be ~0 per row. Got:\n{mean}"
    assert torch.allclose(var, torch.ones_like(var), atol=2e-5, rtol=0.0), f"LayerNorm variance should be close to 1 per row (eps={ln.eps}). Got:\n{var}"


def test_layernorm_has_trainable_scale_and_shift() -> None:
    ln = LayerNorm(emb_dim=5)

    assert hasattr(ln, "scale"), "LayerNorm missing 'scale' parameter"
    assert hasattr(ln, "shift"), "LayerNorm missing 'shift' parameter"

    assert ln.scale.requires_grad, "LayerNorm.scale should require gradients"
    assert ln.shift.requires_grad, "LayerNorm.shift should require gradients"

    assert ln.scale.shape == torch.Size([5]), f"scale shape expected [5], got {ln.scale.shape}"
    assert ln.shift.shape == torch.Size([5]), f"shift shape expected [5], got {ln.shift.shape}"

    assert torch.allclose(ln.scale.detach(), torch.ones(5)), f"scale init expected all-ones, got:\n{ln.scale.detach()}"
    assert torch.allclose(ln.shift.detach(), torch.zeros(5)), f"shift init expected all-zeros, got:\n{ln.shift.detach()}"


@pytest.mark.parametrize("shape", [(2, 5), (2, 4, 5), (1, 3, 7, 5)])
def test_layernorm_preserves_input_shape(shape) -> None:
    torch.manual_seed(123)
    x = torch.randn(*shape)
    ln = LayerNorm(emb_dim=shape[-1])
    y = ln(x)
    assert y.shape == x.shape, f"LayerNorm should preserve shape. Expected {x.shape}, got {y.shape}"
