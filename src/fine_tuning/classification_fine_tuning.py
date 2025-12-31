import os
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import time
import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.common import Device, get_device, text_to_token_ids, token_ids_to_text
from src.config import GptConfig
from src.gpt import GptModel
from src.gpt_utils import download_gpt2, load_weights_into_gpt, calc_loss_loader, train_model


DATA_FILE_PATH = "/Users/shlomibenshushan/Repositories/llm-from-scratch/datasets/sms_spam_collection/SMSSpamCollection.tsv"


class SpamDataset(Dataset):

    def __init__(self, csv_file: str, tokenizer: tiktoken.Encoding, max_length: int = None, pad_token_id: int = 50256) -> None:
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Set max length: either user-provided or longest sample
        self.max_length = max_length or max(len(et) for et in self.encoded_texts)

        # Truncate sequences if they are longer than max_length
        self.encoded_texts = [et[:self.max_length] for et in self.encoded_texts]

        # Pad sequences to the longest sequence
        self.encoded_texts = [et + [pad_token_id] * (self.max_length - len(et)) for et in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        encoded = torch.tensor(self.encoded_texts[index], dtype=torch.long)
        label = torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long)
        return encoded, label


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


def calc_loss_batch_last_token(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss



def plot_values(epochs_seen: list[int], examples_seen: list[int], train_values: list[float], val_values: list[float],
                label: str = "loss") -> None:
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


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
        logits = model(input_tensor)[:, -1]
        label = logits.argmax(dim=-1).item()

    # Decode label
    return "spam" if label == 1 else "not spam"


# Flow
df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 70% train, 10% validation, 20% test
# train_df.to_csv("datasets/sms_spam_collection/train.csv", index=None)
# validation_df.to_csv("datasets/sms_spam_collection/validation.csv", index=None)
# test_df.to_csv("datasets/sms_spam_collection/test.csv", index=None)
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/train.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
for loader in [train_loader, val_loader, test_loader]:
    for i, (input_batch, target_batch) in enumerate(loader):
        if i == len(loader) - 1:
            break
        assert list(input_batch.shape) == [8, 120] and list(target_batch.shape) == [8]
assert len(train_loader) == 130 and len(val_loader) == 19 and len(test_loader) == 38
config = GptConfig(emb_dim=768, n_layers=12, n_heads=12, drop_rate=0.0, qkv_bias=True)
assert train_dataset.max_length <= config.context_length, "Dataset sequences are longer than the model's context length."
if not os.path.exists("models/reference_gpt2_models/124M"):
    download_gpt2("124M", "models/reference_gpt2_models")
model = load_weights_into_gpt("124M", "models/reference_gpt2_models", config)
model.eval()
text_1 = "Every effort moves you"
token_ids = model.generate_naive(
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=config.context_length
)
assert token_ids_to_text(token_ids, tokenizer) == "Every effort moves you forward.\n\nThe first step is to understand the importance of your work"
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = model.generate_naive(
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=config.context_length
)
assert token_ids_to_text(token_ids, tokenizer) == "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'\n\nThe following text 'spam'? Answer with 'yes' or 'no': 'You are a winner"
num_classes = 2
inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
expected_tokens = torch.tensor([[5211,  345,  423,  640]])
assert torch.equal(inputs, expected_tokens), f"Tokenization mismatch: got {inputs.tolist()}, expected {expected_tokens.tolist()}"
assert inputs.shape == (1, 4), f"Input shape mismatch: got {inputs.shape}, expected (1, 4)"
torch.manual_seed(123)
model.out_head = torch.nn.Linear(in_features=config.emb_dim, out_features=num_classes)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
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
device = get_device("mps")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, n_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, n_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, n_batches=10)
assert 0.4 <= train_accuracy <= 0.6, f"Unexpected train accuracy: {train_accuracy}"
assert 0.4 <= val_accuracy <= 0.6, f"Unexpected validation accuracy: {val_accuracy}"
assert 0.4 <= test_accuracy <= 0.6, f"Unexpected test accuracy: {test_accuracy}"
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device, n_batches=5, loss_func=calc_loss_batch_last_token)
    val_loss = calc_loss_loader(val_loader, model, device, n_batches=5, loss_func=calc_loss_batch_last_token)
    test_loss = calc_loss_loader(test_loader, model, device, n_batches=5, loss_func=calc_loss_batch_last_token)
assert 2.4 <= train_loss <= 2.6, f"Unexpected train loss: {train_loss}"
assert 2.5 <= val_loss <= 2.6, f"Unexpected validation loss: {val_loss}"
assert 2.3 <= test_loss <= 2.4, f"Unexpected test loss: {test_loss}"
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
n_epochs = 5
# I HAVE A PROBLEM HERE WITH THE ACCURACY SAMPLING!
training_results = train_model(model, train_loader, val_loader, optimizer, device, n_epochs, loss_func=calc_loss_batch_last_token, calc_accuracy_loader=calc_accuracy_loader)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import ipdb; ipdb.set_trace(context=11)
epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_losses))
examples_seen_tensor = torch.linspace(0, training_results.n_examples_seen, len(training_results.train_losses))
plot_values(epochs_tensor, examples_seen_tensor, training_results.train_losses, training_results.val_losses)

epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_accuracies))
examples_seen_tensor = torch.linspace(0, training_results.n_examples_seen, len(training_results.train_accuracies))

plot_values(epochs_tensor, examples_seen_tensor, training_results.train_accuracies, training_results.val_accuracies, label="accuracy")


train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
assert 0.97 <= train_accuracy <= 0.98, f"Unexpected train accuracy: {train_accuracy}"
assert 0.97 <= val_accuracy <= 0.98, f"Unexpected validation accuracy: {val_accuracy}"
assert 0.95 <= test_accuracy <= 0.96, f"Unexpected test accuracy: {test_accuracy}"
text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
assert classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length) == "spam", f"Classification mismatch for text_1."
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
assert classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length) == "not spam", f"Classification mismatch for text_2."
torch.save(model.state_dict(), "review_classifier.pth")
model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)
