import pytest
import tiktoken
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.gpt import GptModel
from src.config import GPT_CONFIG_124M, GptConfig
from src.gpt_train import text_to_token_ids, token_ids_to_text, calc_loss_batch, calc_loss_loader, train_test_split


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


# TODO: add tests for evaluate_model, generate_and_print_sample and train_model_simple