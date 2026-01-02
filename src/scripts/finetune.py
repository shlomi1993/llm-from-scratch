import argparse
import os
import pandas as pd
import tiktoken
import time
import torch

from dataclasses import dataclass
from logging import getLogger as get_logger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.datasets.classification import SpamDataset
from src.model.config import GptConfig, add_arguments as add_gpt_config_arguments
from src.model.gpt import GptModel
from src.scripts.losses import calc_loss_last_token, calc_loss_loader, evaluate_model
from src.scripts.generate import load_weights_into_gpt
from src.utils.device import Device, get_device
from src.utils.tokenization import tokenizer
from src.utils.visualization import plot_metrics


_logger = get_logger(__name__)


@dataclass
class FineTuningResults:
    model: GptModel
    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]
    n_examples_seen: int = None

    def breakdown(self):
        return self.model, self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies, self.n_examples_seen



def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    # Map labels to numerical values
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    return balanced_df


def random_split(df: pd.DataFrame, train_frac: float, validation_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def calc_accuracy_loader(loader: DataLoader, model: GptModel, device: Device, n_batches: int = None) -> float:
    model.eval()
    correct_predictions = 0
    n_examples = 0
    n_batches = len(loader) if n_batches is None else min(n_batches, len(loader))
    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= n_batches:
            break

        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch: Tensor
        target_batch: Tensor
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # Logits of last output token

        predicted_labels = torch.argmax(logits, dim=-1)

        # Update metrics
        n_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / n_examples


def classify_review(text: str, model: GptModel, tokenizer: tiktoken.Encoding, device: Device, max_length: int,
                    pad_token_id: int = 50256):
    model.eval()

    # Verify that the input length does not exceed model context length
    supported_context = model.pos_emb.weight.shape[0]
    if max_length > supported_context:
        raise ValueError(f"max_length ({max_length}) exceeds model context ({supported_context}).")

    # Tokenize and truncate
    input_ids = tokenizer.encode(text)[:max_length]

    # Pad
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits: Tensor = model(input_tensor)[:, -1]
        label = logits.argmax(dim=-1).item()

    # Decode label
    return "spam" if label == 1 else "not spam"


def create_dataloaders(training_set_path: str, sep: str, column_names: list[str], train_frac: float, val_frac: float,
                       save_split_dir: str, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, int]:

    # Load and preprocess dataset
    df = pd.read_csv(training_set_path, sep=sep, header=None, names=column_names)
    balanced_df = create_balanced_dataset(df)
    train_df, val_df, test_df = random_split(balanced_df, train_frac, val_frac)  # 70% train, 10% validation, 20% test

    # Save splits to CSV files
    os.makedirs(save_split_dir, exist_ok=True)
    train_csv_path = os.path.join(save_split_dir, "train.csv")
    val_csv_path = os.path.join(save_split_dir, "validation.csv")
    test_csv_path = os.path.join(save_split_dir, "test.csv")
    train_df.to_csv(train_csv_path, index=None)
    val_df.to_csv(val_csv_path, index=None)
    test_df.to_csv(test_csv_path, index=None)

    # Create datasets
    train_dataset = SpamDataset(csv_file=train_csv_path, max_length=None)
    val_dataset = SpamDataset(csv_file=val_csv_path, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(csv_file=test_csv_path, max_length=train_dataset.max_length)

    # Create dataloaders
    torch.manual_seed(seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    ### TEST
    for loader in [train_loader, val_loader, test_loader]:
        for i, (input_batch, target_batch) in enumerate(loader):
            if i == len(loader) - 1:
                break
            assert list(input_batch.shape) == [8, 120] and list(target_batch.shape) == [8]
    assert len(train_loader) == 130 and len(val_loader) == 19 and len(test_loader) == 38
    ### END TEST

    return train_loader, val_loader, test_loader, train_dataset.max_length


def finetune_classifier(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        device: Device, n_epochs: int, eval_freq, eval_iter) -> FineTuningResults:

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode

        _logger.info(f"Epoch {epoch} / {n_epochs}")
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_last_token(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, calc_loss_last_token)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                _logger.info(f"  Step {global_step:06d}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, n_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, n_batches=eval_iter)
        _logger.info(f"  Training accuracy: {train_accuracy * 100:.2f}% | Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return FineTuningResults(model, train_losses, val_losses, train_accs, val_accs, examples_seen)


def run_classification_finetuning_flow(config: GptConfig, models_dir: str, model_size: str, training_set_path: str,
                                       sep="\t", column_names=["Label", "Text"], train_frac: float = 0.7,
                                       validation_frac: float = 0.1, save_split_dir: str = ".", batch_size: int = 8,
                                       seed: int = 123, device: str = "auto", lr: float = 5e-4, n_epochs: int = 5,
                                       weight_decay: float = 0.1, eval_freq: int = 50, eval_iter: int = 5) -> FineTuningResults:

    _logger.info("Running classification fine-tuning flow...")

    _logger.info("Preparing dataset")
    train_loader, val_loader, test_loader, max_length = create_dataloaders(
        training_set_path, sep, column_names, train_frac, validation_frac, save_split_dir, batch_size, seed
    )
    assert max_length <= config.context_length, "Dataset sequences are longer than the model's context length."

    _logger.info("Loading pre-trained model")
    model = load_weights_into_gpt(model_size, models_dir, config)
    model.eval()

    ### TEST
    text_1 = "Every effort moves you"
    token_ids = model.generate_naive(
        idx=tokenizer.text_to_token_ids(text_1),
        max_new_tokens=15,
        context_size=config.context_length
    )
    assert tokenizer.token_ids_to_text(token_ids) == "Every effort moves you forward.\n\nThe first step is to understand the importance of your work"
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = model.generate_naive(
        idx=tokenizer.text_to_token_ids(text_2),
        max_new_tokens=23,
        context_size=config.context_length
    )
    assert tokenizer.token_ids_to_text(token_ids) == "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'\n\nThe following text 'spam'? Answer with 'yes' or 'no': 'You are a winner"
    ### END TEST

    ### TEST
    inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
    expected_tokens = torch.tensor([[5211,  345,  423,  640]])
    assert torch.equal(inputs, expected_tokens), f"Tokenization mismatch: got {inputs.tolist()}, expected {expected_tokens.tolist()}"
    assert inputs.shape == (1, 4), f"Input shape mismatch: got {inputs.shape}, expected (1, 4)"
    ### END TEST

    _logger.info("Preparing model for classification fine-tuning")
    torch.manual_seed(seed)
    model.out_head = torch.nn.Linear(in_features=config.emb_dim, out_features=2)
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    ### TEST
    with torch.no_grad():
        outputs = model(inputs)
    expected_outputs = torch.tensor([[[-1.5854,  0.9904], [-3.7235,  7.4548], [-2.2661,  6.6049], [-3.5983,  3.9902]]])
    assert torch.allclose(outputs, expected_outputs, atol=1e-4), f"Output mismatch: got {outputs.tolist()}, expected {expected_outputs.tolist()}"
    expected_shape = (1, 4, 2)
    assert outputs.shape == expected_shape, f"Output shape mismatch: got {outputs.shape}, expected {expected_shape}"
    expected_last = torch.tensor([[-3.5983,  3.9902]])
    assert torch.allclose(outputs[:, -1, :], expected_last, atol=1e-4), f"Last output mismatch: got {outputs[:, -1, :].tolist()}, expected {expected_last.tolist()}"
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    assert label.item() == 1
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    assert label.item() == 1
    ### END TEST

    _logger.info(f"Load model to {device}")
    device = get_device(device)
    model.to(device)

    ### TEST
    torch.manual_seed(seed)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, n_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, n_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, n_batches=10)
    assert 0.4 <= train_accuracy <= 0.6, f"Unexpected train accuracy: {train_accuracy}"
    assert 0.4 <= val_accuracy <= 0.6, f"Unexpected validation accuracy: {val_accuracy}"
    assert 0.4 <= test_accuracy <= 0.6, f"Unexpected test accuracy: {test_accuracy}"
    ### END TEST

    ### TEST
    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, calc_loss_last_token, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_last_token, model, device, n_batches=5)
        test_loss = calc_loss_loader(test_loader, calc_loss_last_token, model, device, n_batches=5)
    assert 2.4 <= train_loss <= 2.6, f"Unexpected train loss: {train_loss}"
    assert 2.5 <= val_loss <= 2.6, f"Unexpected validation loss: {val_loss}"
    assert 2.3 <= test_loss <= 2.4, f"Unexpected test loss: {test_loss}"
    ### END TEST

    _logger.info("Starting fine-tuning...")
    start_time = time.time()
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_results = finetune_classifier(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    _logger.info(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_losses))
    examples_seen_tensor = torch.linspace(0, training_results.n_examples_seen, len(training_results.train_losses))
    plot_metrics(epochs_tensor, examples_seen_tensor, training_results.train_losses, training_results.val_losses, savefig_path="finetune_loss.pdf", label="loss")

    epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_accuracies))
    examples_seen_tensor = torch.linspace(0, training_results.n_examples_seen, len(training_results.train_accuracies))
    import ipdb; ipdb.set_trace(context=11)
    plot_metrics(epochs_tensor, examples_seen_tensor, training_results.train_accuracies, training_results.val_accuracies, savefig_path="finetune_accuracy.pdf", label="accuracy")


    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    assert 0.97 <= train_accuracy <= 0.98, f"Unexpected train accuracy: {train_accuracy}"
    assert 0.97 <= val_accuracy <= 0.98, f"Unexpected validation accuracy: {val_accuracy}"
    assert 0.95 <= test_accuracy <= 0.96, f"Unexpected test accuracy: {test_accuracy}"
    text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
    assert classify_review(text_1, model, tokenizer, device, max_length=max_length) == "spam", f"Classification mismatch for text_1."
    text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    assert classify_review(text_2, model, tokenizer, device, max_length=max_length) == "not spam", f"Classification mismatch for text_2."
    torch.save(model.state_dict(), "review_classifier.pth")
    model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)



run_classification_finetuning_flow(
    config=GptConfig(emb_dim=768, n_layers=12, n_heads=12, drop_rate=0.0, qkv_bias=True),
    models_dir="models/reference_gpt2_models",
    model_size="124M",
    training_set_path="datasets/sms_spam_collection/SMSSpamCollection.tsv",
    sep="\t",
    column_names=["Label", "Text"],
    train_frac=0.7,
    validation_frac=0.1,
    save_split_dir="datasets/sms_spam_collection/splits",
    batch_size=8,
    seed=123
)




# def add_arguments(parser: argparse.ArgumentParser) -> None:
#     parser.add_argument("--training-set-path", type=str, required=True, help="Path to the training text file.")
#     parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer.")
#     parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs.")
#     parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training.")
#     parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
#     parser.add_argument("--dataset-encoding", type=str, default="utf-8", help="Encoding of the training text file.")
#     parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, auto).")
#     parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
#     parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence length for training samples.")





# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Fine-tune a GPT model for SMS spam classification.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     add_gpt_config_arguments(parser)
#     add_arguments(parser)
#     args = parser.parse_args()
    
#     config = GptConfig(emb_dim=768, n_layers=12, n_heads=12, drop_rate=0.0, qkv_bias=True)

#     if not os.path.exists("models/reference_gpt2_models/124M"):
#         download_gpt2("124M", "models/reference_gpt2_models")
#     model = load_weights_into_gpt("124M", "models/reference_gpt2_models", config)

# device should be 'mps'