import pytest
import torch
import os

from src.data.datasets.classification import SpamDataset
from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.finetune import (
    create_dataloaders,
    calc_accuracy_loader,
    calc_loss_loader,
    classify_review,
)
from src.scripts.losses import calc_loss_last_token
from src.scripts.generate import load_tf_weights_into_gpt
from src.utils.device import get_device
from src.utils.tokenization import tokenizer


@pytest.fixture
def default_config():
    """Default GPT config for testing."""
    return GptConfig(emb_dim=768, n_layers=12, n_heads=12, drop_rate=0.0, qkv_bias=True)


@pytest.fixture
def pretrained_model(default_config):
    """Load pretrained GPT model."""
    model = load_tf_weights_into_gpt("124M", "models/reference_gpt2_models", default_config)
    model.eval()
    return model


@pytest.fixture
def dataloaders():
    """Create dataloaders for testing."""
    train_loader, val_loader, test_loader, max_length = create_dataloaders(
        training_set_path="datasets/sms_spam_collection/SMSSpamCollection.tsv",
        sep="\t",
        column_names=["Label", "Text"],
        train_frac=0.7,
        val_frac=0.1,
        save_split_dir="datasets/sms_spam_collection/splits",
        batch_size=8,
        seed=123
    )
    return train_loader, val_loader, test_loader, max_length


def test_dataloaders_shape_and_length(dataloaders):
    """Test that dataloaders have correct batch shapes and lengths."""
    train_loader, val_loader, test_loader, max_length = dataloaders
    
    # Test batch shapes (excluding last batch which may be smaller)
    for loader in [train_loader, val_loader, test_loader]:
        for i, (input_batch, target_batch) in enumerate(loader):
            if i == len(loader) - 1:
                break
            assert list(input_batch.shape) == [8, 120] and list(target_batch.shape) == [8]
    
    # Test loader lengths
    assert len(train_loader) == 130 and len(val_loader) == 19 and len(test_loader) == 38


def test_pretrained_model_generation(pretrained_model, default_config):
    """Test that pretrained model can generate text correctly."""
    text_1 = "Every effort moves you"
    token_ids = pretrained_model.generate_naive(
        idx=tokenizer.text_to_token_ids(text_1),
        max_new_tokens=15,
        context_size=default_config.context_length
    )
    assert tokenizer.token_ids_to_text(token_ids) == "Every effort moves you forward.\n\nThe first step is to understand the importance of your work"
    
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = pretrained_model.generate_naive(
        idx=tokenizer.text_to_token_ids(text_2),
        max_new_tokens=23,
        context_size=default_config.context_length
    )
    assert tokenizer.token_ids_to_text(token_ids) == "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'\n\nThe following text 'spam'? Answer with 'yes' or 'no': 'You are a winner"


def test_tokenization():
    """Test that tokenizer correctly encodes text."""
    inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    expected_tokens = torch.tensor([[5211, 345, 423, 640]])
    assert torch.equal(inputs, expected_tokens), f"Tokenization mismatch: got {inputs.tolist()}, expected {expected_tokens.tolist()}"
    assert inputs.shape == (1, 4), f"Input shape mismatch: got {inputs.shape}, expected (1, 4)"


def test_classification_head_initialization(pretrained_model, default_config):
    """Test that model with classification head produces correct outputs."""
    # Prepare inputs
    inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    
    # Initialize classification head
    torch.manual_seed(123)
    pretrained_model.out_head = torch.nn.Linear(in_features=default_config.emb_dim, out_features=2)
    for param in pretrained_model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in pretrained_model.final_norm.parameters():
        param.requires_grad = True
    
    # Test forward pass
    with torch.no_grad():
        outputs = pretrained_model(inputs)
    
    expected_outputs = torch.tensor([[[-1.5854, 0.9904], [-3.7235, 7.4548], [-2.2661, 6.6049], [-3.5983, 3.9902]]])
    assert torch.allclose(outputs, expected_outputs, atol=1e-4), f"Output mismatch: got {outputs.tolist()}, expected {expected_outputs.tolist()}"
    
    expected_shape = (1, 4, 2)
    assert outputs.shape == expected_shape, f"Output shape mismatch: got {outputs.shape}, expected {expected_shape}"
    
    expected_last = torch.tensor([[-3.5983, 3.9902]])
    assert torch.allclose(outputs[:, -1, :], expected_last, atol=1e-4), f"Last output mismatch: got {outputs[:, -1, :].tolist()}, expected {expected_last.tolist()}"
    
    # Test softmax probabilities
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    assert label.item() == 1
    
    # Test logits
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    assert label.item() == 1


def test_initial_accuracy_before_training(pretrained_model, dataloaders, default_config):
    """Test that accuracy before training is around random (50%)."""
    train_loader, val_loader, test_loader, max_length = dataloaders
    
    # Initialize classification head
    torch.manual_seed(123)
    pretrained_model.out_head = torch.nn.Linear(in_features=default_config.emb_dim, out_features=2)
    for param in pretrained_model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in pretrained_model.final_norm.parameters():
        param.requires_grad = True
    
    device = get_device("cpu")
    pretrained_model.to(device)
    
    # Calculate accuracy
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, pretrained_model, device, n_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, pretrained_model, device, n_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, pretrained_model, device, n_batches=10)
    
    assert 0.4 <= train_accuracy <= 0.6, f"Unexpected train accuracy: {train_accuracy}"
    assert 0.4 <= val_accuracy <= 0.6, f"Unexpected validation accuracy: {val_accuracy}"
    assert 0.4 <= test_accuracy <= 0.6, f"Unexpected test accuracy: {test_accuracy}"


def test_initial_loss_before_training(pretrained_model, dataloaders, default_config):
    """Test that loss before training is around log(2) for binary classification."""
    train_loader, val_loader, test_loader, max_length = dataloaders
    
    # Initialize classification head
    torch.manual_seed(123)
    pretrained_model.out_head = torch.nn.Linear(in_features=default_config.emb_dim, out_features=2)
    for param in pretrained_model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in pretrained_model.final_norm.parameters():
        param.requires_grad = True
    
    device = get_device("cpu")
    pretrained_model.to(device)
    
    # Calculate loss
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_last_token, pretrained_model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_last_token, pretrained_model, device, n_batches=5)
        test_loss = calc_loss_loader(test_loader, calc_loss_last_token, pretrained_model, device, n_batches=5)
    
    assert 2.4 <= train_loss <= 2.6, f"Unexpected train loss: {train_loss}"
    assert 2.5 <= val_loss <= 2.6, f"Unexpected validation loss: {val_loss}"
    assert 2.3 <= test_loss <= 2.4, f"Unexpected test loss: {test_loss}"
