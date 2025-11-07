import torch

from importlib.metadata import version

from dataloader import create_dataloader_v1


def test_torch_version():
    """
    Test that torch version is >= 2.2.2
    """
    torch_version = version("torch")
    torch_major, torch_minor, torch_patch = map(int, torch_version.split('.')[:3])
    assert (torch_major > 2 or
            (torch_major == 2 and torch_minor > 2) or
            (torch_major == 2 and torch_minor == 2 and torch_patch >= 2)), \
        f"torch version {torch_version} is less than required 2.2.2"


def test_tiktoken_version():
    """
    Test that tiktoken version is >= 0.5.1
    """
    tiktoken_version = version("tiktoken")
    tiktoken_major, tiktoken_minor, tiktoken_patch = map(int, tiktoken_version.split('.')[:3])
    assert (tiktoken_major > 0 or
            (tiktoken_major == 0 and tiktoken_minor > 5) or
            (tiktoken_major == 0 and tiktoken_minor == 5 and tiktoken_patch >= 1)), \
        f"tiktoken version {tiktoken_version} is less than required 0.5.1"


def test_dataloader_functionality():
    """
    Test that the dataloader creates embeddings with expected shape torch.Size([8, 4, 256])
    """
    with open("datasets/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text: str = f.read()

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size, max_length, max_length)

    for batch in dataloader:
        x, y = batch
        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))
        input_embeddings = token_embeddings + pos_embeddings
        break

    expected_shape = torch.Size([8, 4, 256])
    assert input_embeddings.shape == expected_shape, f"Expected shape {expected_shape}, but got {input_embeddings.shape}"
