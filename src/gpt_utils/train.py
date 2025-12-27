import matplotlib.pyplot as plt
import torch
import tiktoken

from torch import Tensor
from torch.utils.data import DataLoader

from src.dataloader import GptDataloaderV1
from src.gpt import GptModel
from src.utils import text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: torch.device) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss  # Negative average log probability


def calc_loss_loader(data_loader: DataLoader, model: GptModel, device: torch.device, num_batches: int = None):
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def train_test_split(text: str, max_length: int, batch_size: int, stride: int = None, train_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    stride = stride or max_length
    train_loader = GptDataloaderV1(train_text, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0)
    val_loader = GptDataloaderV1(val_text, batch_size, max_length, stride, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader


def evaluate_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, eval_iter: int) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: GptModel, tokenizer: tiktoken.Encoding, device: torch.device, start_context: str) -> None:
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate_naive(idx=encoded, max_new_tokens=50, context_size=model.pos_emb.weight.shape[0])
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, n_epochs: int, eval_freq: int, eval_iter: int, start_context: str,
                tokenizer: tiktoken.Encoding) -> tuple[list[float], list[float], list[int]]:

    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(1, n_epochs + 1):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            input_batch: Tensor
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen: list[int], tokens_seen: list[int], train_losses: list[float], val_losses: list[float]) -> None:
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()
