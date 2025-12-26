import torch

from src.configurations import GPT_CONFIG_124M
from src.feed_forward import FeedForward


def test_feedforward_output_shape() -> None:
    torch.manual_seed(123)
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)  # [batch_size, num_tokens, emb_dim]
    out = ffn(x)
    assert out.shape == x.shape, f"FeedForward output shape mismatch: expected {x.shape}, got {out.shape}"
