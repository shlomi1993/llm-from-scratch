import pytest
import tiktoken
import torch


from src.gpt import GptModel
from src.configurations import GPT_CONFIG_124M, GptConfig
from src.gpt_train import text_to_token_ids, token_ids_to_text, calc_loss_batch


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


def test_text_to_token_ids_and_token_ids_to_text(tokenizer):
    text = "Every effort moves you"
    token_ids = text_to_token_ids(text, tokenizer)
    roundtrip = token_ids_to_text(token_ids, tokenizer)
    assert roundtrip == text, f"Roundtrip text mismatch: got '{roundtrip}'"


def test_token_ids_to_text_with_model(tokenizer: tiktoken.Encoding):
    torch.manual_seed(123)
    config = GptConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        qkv_bias=False
    )
    model = GptModel(config)
    text = "Every effort moves you"
    token_ids = model.generate(
        idx=text_to_token_ids(text, tokenizer),
        max_new_tokens=10,
        context_size=config.context_length
    )
    out = token_ids_to_text(token_ids, tokenizer)
    assert out == "Every effort moves youprettyvis Slow shoulders JA xmlsavf Meridian stereotyp", f"Generated text mismatch: got '{out}'"


def test_calc_loss_batch():
    torch.manual_seed(123)
    model = GptModel(GPT_CONFIG_124M)
    model.eval()

    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        loss = calc_loss_batch(inputs, targets, model, device)

    assert torch.isclose(loss, torch.tensor(10.8765), atol=1e-5), f"Loss mismatch: got {loss.item()}"
