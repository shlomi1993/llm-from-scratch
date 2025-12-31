import matplotlib.pyplot as plt
import torch
import tiktoken

from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import DataLoader

from src.common import Device, get_device, text_to_token_ids, token_ids_to_text
from src.config import GptConfig
from src.dataloader import GptDataloaderV1
from src.gpt import GptModel


@dataclass
class TrainingResults:
    model: GptModel
    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float] = None
    val_accuracies: list[float] = None
    n_tokens_seen: list[int] = None
    n_examples_seen: int = None



def calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss  # Negative average log probability


def calc_loss_loader(data_loader: DataLoader, model: GptModel, device: Device, n_batches: int = None,
                     loss_func: callable = calc_loss_batch) -> float:
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.
    n_batches = min(n_batches, len(data_loader)) if n_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= n_batches:
            break

        loss: Tensor = loss_func(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / n_batches


def train_test_split(text: str, max_length: int, batch_size: int, stride: int = None, train_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    stride = stride or max_length
    train_loader = GptDataloaderV1(train_text, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0)
    val_loader = GptDataloaderV1(val_text, batch_size, max_length, stride, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader


def evaluate_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: Device, eval_iter: int,
                   loss_func: callable = calc_loss_batch) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, n_batches=eval_iter, loss_func=loss_func)
        val_loss = calc_loss_loader(val_loader, model, device, n_batches=eval_iter, loss_func=loss_func)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: GptModel, tokenizer: tiktoken.Encoding, device: Device, start_context: str) -> None:
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate_naive(idx=encoded, max_new_tokens=50, context_size=model.pos_emb.weight.shape[0])
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: Device, n_epochs: int, eval_freq: int = 50, eval_iter: int = 5, start_context: str = None,
                tokenizer: tiktoken.Encoding = None, loss_func: callable = calc_loss_batch,
                track_seen_tokens: bool = False, calc_accuracy_loader: callable = None) -> TrainingResults:

    if bool(tokenizer) != bool(start_context):
        raise ValueError(f"Both '{tokenizer.__name__}' and '{start_context.__name__}' must be provided for sample generation.")

    # Initialize lists to track losses and tokens/examples seen
    train_losses, val_losses, train_accuracies, val_accuracies, tokens_seen = [], [], [], [], []
    n_tokens_seen = 0
    n_examples_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(1, n_epochs + 1):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            input_batch: Tensor
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss: Tensor = loss_func(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            n_tokens_seen += input_batch.numel()
            n_examples_seen += input_batch.shape[0]
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, loss_func)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                if track_seen_tokens:
                    tokens_seen.append(n_tokens_seen)
                print(f"Epoch {epoch} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        if tokenizer:
            generate_and_print_sample(model, tokenizer, device, start_context)

        # Calculate accuracy after each epoch if classification mode
        if calc_accuracy_loader is not None:
            train_accuracy = calc_accuracy_loader(train_loader, model, device, n_batches=eval_iter)
            val_accuracy = calc_accuracy_loader(val_loader, model, device, n_batches=eval_iter)
            print(f"Training accuracy: {train_accuracy * 100:.2f}% | Validation accuracy: {val_accuracy * 100:.2f}%")
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

    return TrainingResults(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies if calc_accuracy_loader else None,
        val_accuracies=val_accuracies if calc_accuracy_loader else None,
        n_tokens_seen=tokens_seen if track_seen_tokens else None,
        n_examples_seen=n_examples_seen
    )


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


def run_model_training_flow(config: GptConfig, training_set_path: str, tokenizer: tiktoken.Encoding, lr: float = 5e-4, n_epochs: int = 10,
                            batch_size: int = 2, weight_decay: float = 0.1, dataset_encoding: str = "utf-8",
                            device: str = "cpu", seed: int = 123, max_length: int = None, stride: int = None,
                            train_ratio: float = 0.9, eval_freq: int = 5, eval_iter: int = 1,
                            start_context: str = "Every effort moves you", saved_model_path: str = "model.pth",
                            make_plot: bool = True, saved_plot_path: str = "loss.pdf") -> TrainingResults:

    # TODO Change start_context default

    # Initialize dynamic parameters
    max_length = max_length or config.context_length
    stride = stride or max_length

    # General setup
    torch.manual_seed(seed)
    device = get_device(device)

    # Load dataset
    with open(training_set_path, "r", encoding=dataset_encoding) as file:
        text_data = file.read()

    # Load model
    model = GptModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train-test split
    train_loader, val_loader = train_test_split(text_data, max_length, batch_size, stride, train_ratio)

    # Train model
    training_results = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context, tokenizer
    )

    # Save and load model
    torch.save(model.state_dict(), saved_model_path)  # TODO To load: model = GptModel(config); model.load_state_dict(torch.load(saved_model_path, weights_only=True))

    # Plot loss
    if make_plot:
        epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_losses))
        plot_losses(epochs_tensor, training_results.n_tokens_seen, training_results.train_losses, training_results.val_losses)
        plt.savefig(saved_plot_path)

    # Return training results
    return training_results