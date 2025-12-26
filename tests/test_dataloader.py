import torch
import torch.nn as nn
import tiktoken

from importlib.metadata import version

from src.dataloader import create_dataloader_v1


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


def test_gpt_dataset_and_dataloader_embedding_shape() -> None:
    raw_text = (
        "Hello world. This is a small test text for GPT dataset. "
        "We want enough tokens to form several sequences. "
        "Adding more text so we can form enough batches for batch_size=8. "
        "Hello world. This is a small test text for GPT dataset. "
        "We want enough tokens to form several sequences. "
        "Adding more text so we can form enough batches for batch_size=8. "
        "Hello world. This is a small test text for GPT dataset. "
        "We want enough tokens to form several sequences. "
        "Adding more text so we can form enough batches for batch_size=8. "
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    max_length = 4
    batch_size = 8
    stride = max_length
    num_samples = len(range(0, len(encoded_text) - max_length, stride))

    assert num_samples >= batch_size, (
        "Not enough samples to yield one full batch with drop_last=True.\n"
        f"Need >= {batch_size} samples but got {num_samples}.\n"
        f"Token count={len(encoded_text)}, max_length={max_length}, stride={stride}.\n"
        "Fix by increasing raw_text length, reducing batch_size, or setting drop_last=False."
    )

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(max_length, output_dim)

    dataloader = create_dataloader_v1(raw_text, batch_size, max_length, stride, shuffle=False, drop_last=True)

    it = iter(dataloader)
    x, y = next(it)

    assert x.shape == torch.Size([batch_size, max_length]), f"Input batch shape mismatch: expected {torch.Size([batch_size, max_length])}, got {x.shape}"
    assert y.shape == torch.Size([batch_size, max_length]), f"Target batch shape mismatch: expected {torch.Size([batch_size, max_length])}, got {y.shape}"

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings

    assert input_embeddings.shape == torch.Size([batch_size, max_length, output_dim]), \
        f"Input embeddings shape mismatch: expected {torch.Size([batch_size, max_length, output_dim])}, got {input_embeddings.shape}"
