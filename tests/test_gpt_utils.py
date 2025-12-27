import os
import pytest
import shutil
import tempfile
import tiktoken
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.config import GPT_CONFIG_124M, GptConfig
from src.gpt import GptModel
from src.gpt_utils import (
    FILES_TO_DOWNLOAD,
    download_gpt2,
    load_weights_into_gpt,
    calc_loss_batch,
    calc_loss_loader,
    train_test_split,
    evaluate_model,
    generate_and_print_sample,
    train_model
)
from src.utils import text_to_token_ids, token_ids_to_text


@pytest.mark.skip(reason="Downloading files from the internet takes time; run manually when needed")
def test_download_and_load_gpt2_124m(tokenizer: tiktoken.Encoding):
    model_size = "124M"
    temp_dir = tempfile.mkdtemp()
    try:
        model_dir = download_gpt2(model_size, temp_dir)

        # Check that required files are downloaded
        for fname in FILES_TO_DOWNLOAD:
            fpath = os.path.join(model_dir, fname)
            assert os.path.exists(fpath), f"File missing: {fpath}"

        config = GptConfig(emb_dim=768, n_layers=12, n_heads=12, qkv_bias=True)

        gpt = load_weights_into_gpt(model_size, temp_dir, config)
        gpt.eval()

        input_ids = text_to_token_ids("Every effort moves you", tokenizer)

        deterministic_output_ids = gpt.generate_cached(input_ids, max_new_tokens=10, context_size=config.context_length)
        generated_text = token_ids_to_text(deterministic_output_ids, tokenizer)
        assert generated_text == "Every effort moves you forward.\n\nThe first step is to understand", f"Generated text mismatch: got '{generated_text}'"

        prod_output_ids = gpt.generate(input_ids, max_new_tokens=10, context_size=config.context_length, temperature=1.0, top_k=50, eos_id=tokenizer.eot_token)
        assert prod_output_ids.shape[1] == 14, f"Generated token count should be 4 + 10 = 14, but got {prod_output_ids.shape[1]}"
        generated_text = token_ids_to_text(prod_output_ids, tokenizer)
        assert generated_text.startswith("Every effort moves you"), f"Generated text mismatch: got '{generated_text}'"

    finally:
        shutil.rmtree(temp_dir)


def test_download_and_load_gpt2_invalid_size_raises():
    temp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError):
            download_gpt2("invalid_size", temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def test_text_to_token_ids_and_token_ids_to_text(tokenizer):
    text = "Every effort moves you"
    token_ids = text_to_token_ids(text, tokenizer)
    roundtrip = token_ids_to_text(token_ids, tokenizer)
    assert roundtrip == text, f"Roundtrip text mismatch: got '{roundtrip}'"


def test_token_ids_to_text_with_model(tokenizer: tiktoken.Encoding):
    torch.manual_seed(123)
    config = GptConfig(vocab_size=50257, context_length=256, emb_dim=8, n_heads=2, n_layers=1, drop_rate=0.0)
    model = GptModel(config)
    text = "Every effort moves you"
    token_ids = model.generate_naive(
        idx=text_to_token_ids(text, tokenizer),
        max_new_tokens=10,
        context_size=config.context_length
    )
    out = token_ids_to_text(token_ids, tokenizer)
    assert out == "Every effort moves youprettyvis Slow shoulders JA xmlsavf Meridian stereotyp", f"Generated text mismatch: got '{out}'"


def test_calc_loss_batch(device: torch.device):
    torch.manual_seed(123)
    model = GptModel(GPT_CONFIG_124M)
    model.to(device)
    model.eval()

    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        loss = calc_loss_batch(inputs, targets, model, device)

    assert torch.isclose(loss, torch.tensor(10.8765), atol=1e-5), f"Loss mismatch: got {loss.item()}"


def test_calc_loss_loader(device: torch.device):
    torch.manual_seed(123)
    config = GptConfig(vocab_size=10, context_length=3, emb_dim=4, n_heads=2, n_layers=1, drop_rate=0.0, qkv_bias=False)
    model = GptModel(config)
    model.to(device)

    # Random input and target tokens
    batch_size = 2
    n_batches = 3
    inputs = torch.randint(0, 10, (n_batches * batch_size, 3)).to(device)
    targets = inputs.clone().to(device)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size)

    avg_loss = calc_loss_loader(loader, model, device)
    assert isinstance(avg_loss, float) and avg_loss > 0, f"Average loss should be positive float, got {avg_loss}"

    # For empty loader, device doesn't matter
    avg_loss_empty = calc_loss_loader(DataLoader([]), model, device)
    assert avg_loss_empty != avg_loss_empty, "Average loss on empty loader should be NaN"


@pytest.mark.parametrize("max_length, batch_size, stride", [(10, 4, 5)])
@pytest.mark.parametrize("train_ratio", [0.7, 0.8, 0.9])
def test_train_test_split(the_verdict_dataset: str, tokenizer: tiktoken.Encoding, max_length: int, batch_size: int,
                          stride: int, train_ratio: float):
    train_loader, val_loader = train_test_split(the_verdict_dataset, max_length, batch_size, stride, train_ratio)

    for loader in [train_loader, val_loader]:
        for i, (inputs, targets) in enumerate(loader):
            if i == len(loader) - 1:
                continue
            assert inputs.shape == (batch_size, max_length)
            assert targets.shape == (batch_size, max_length)

    split_idx = int(train_ratio * len(the_verdict_dataset))
    train_text = the_verdict_dataset[:split_idx]
    val_text = the_verdict_dataset[split_idx:]

    def calculate_expected_batches(text: str, drop_last: bool) -> int:
        token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        if len(token_ids) < max_length:
            return 0
        n_samples = ((len(token_ids) - max_length) // stride) + 1
        return n_samples // batch_size if drop_last else (n_samples + batch_size - 1) // batch_size

    expected_train_batches = calculate_expected_batches(train_text, drop_last=True)
    expected_val_batches = calculate_expected_batches(val_text, drop_last=False)

    assert len(train_loader) == expected_train_batches, \
        f"Train loader size mismatch. Expected {expected_train_batches}, got {len(train_loader)}"

    assert len(val_loader) == expected_val_batches, \
        f"Val loader size mismatch. Expected {expected_val_batches}, got {len(val_loader)}"


def test_evaluate_model_runs(sample_model: GptModel, dummy_loader: DataLoader, device: torch.device):
    loader = dummy_loader
    sample_model.to(device)
    train_loss, val_loss = evaluate_model(sample_model, loader, loader, device, eval_iter=1)
    assert isinstance(train_loss, float), f"Train loss should be float, got {type(train_loss)}"
    assert isinstance(val_loss, float), f"Val loss should be float, got {type(val_loss)}"


@pytest.mark.parametrize("start_context", ["Hello world", "Test"])
def test_generate_and_print_sample_runs(sample_model: GptModel, tokenizer: tiktoken.Encoding, device: torch.device, capsys, start_context: str):
    sample_model.to(device)
    generate_and_print_sample(sample_model, tokenizer, device, start_context=start_context)
    out = capsys.readouterr().out
    assert isinstance(out, str) and len(out) > 0, "Generated output should be non-empty string"


def test_train_model_runs(sample_model: GptModel, tokenizer: tiktoken.Encoding, dummy_loader: DataLoader, device: torch.device):
    loader = dummy_loader
    sample_model.to(device)
    optimizer = torch.optim.AdamW(sample_model.parameters(), lr=1e-3)
    train_losses, val_losses, tokens_seen = train_model(
        sample_model, loader, loader, optimizer, device,
        n_epochs=1, eval_freq=1, eval_iter=1, start_context="Hello world", tokenizer=tokenizer
    )
    assert isinstance(train_losses, list), f"train_losses should be list, got {type(train_losses)}"
    assert isinstance(val_losses, list), f"val_losses should be list, got {type(val_losses)}"
    assert isinstance(tokens_seen, list), f"tokens_seen should be list, got {type(tokens_seen)}"
