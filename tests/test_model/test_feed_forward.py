import torch

from src.model.config import GPT_CONFIG_124M
from src.model.feed_forward import FeedForward


def test_feedforward_output_shape() -> None:
    torch.manual_seed(123)
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)  # [batch_size, num_tokens, emb_dim]
    out = ffn(x)
    assert out.shape == x.shape, f"FeedForward output shape mismatch: expected {x.shape}, got {out.shape}"


"""
For tests:

Run:
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)  # input shape: [batch_size, num_token, emb_size]
out = ffn(x)

Expect:
out.shape == torch.Size([2, 3, 768])

"""