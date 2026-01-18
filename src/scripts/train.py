import argparse
import torch

from dataclasses import dataclass
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dataset import GptDatasetV1
from src.model.config import GptConfig, add_arguments as add_gpt_config_arguments
from src.model.gpt import GptModel
from src.utils.checkpoint import save_model
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.losses import calc_loss_batch, calc_losses
from src.utils.tokenization.tokenizer import g_tokenizer
from src.utils.visualization import plot_metrics


@dataclass
class TrainingResults:
    """
    Data class to hold the results of the training process.
    """
    model: GptModel
    train_losses: list[float]
    val_losses: list[float]
    tokens_seen: list[int] = None


def train_test_split(text: str, max_length: int, batch_size: int, stride: int = None, train_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
    """
    Split the text data into training and validation sets, create datasets and data loaders.

    Args:
        text (str): The text data to split.
        max_length (int): The maximum sequence length for training samples.
        batch_size (int): The batch size for data loaders.
        stride (int, optional): The stride for the sliding window over the text. Defaults to max_length if None.
        train_ratio (float): The ratio of data to use for training. Default is 0.9.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    stride = stride or max_length
    train_dataset = GptDatasetV1(train_text, max_length, stride)
    val_dataset = GptDatasetV1(val_text, max_length, stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader


def format_training_progress(epoch: int, n_epochs: int, step: int, n_steps: int, train_loss: float = None,
                             val_loss: float = None, train_acc: float = None, val_acc: float = None) -> str:
    """
    Format the training progress message.

    Args:
        epoch (int): Current epoch number.
        n_epochs (int): Total number of epochs.
        step (int): Current step number.
        n_steps (int): Total number of steps.
        train_loss (float, optional): Training loss. Defaults to None.
        val_loss (float, optional): Validation loss. Defaults to None.
        train_acc (float, optional): Training accuracy. Defaults to None.
        val_acc (float, optional): Validation accuracy. Defaults to None.

    Returns:
        str: Formatted training progress message.

    Example:
        >>> format_training_progress(3, 10, 150, 1000, train_loss=0.456, val_loss=0.512)
        "Epoch  3/10 | Step  150/1000 | Train-loss: 0.456 | Val-loss: 0.512"
    """
    epoch_pad = len(str(n_epochs))
    step_pad = len(str(n_steps))
    msg_list = [
        f"Epoch {epoch:>{epoch_pad}}/{n_epochs}",
        f"Step  {step:>{step_pad}}/{n_steps}",
    ]
    if train_loss is not None:
        msg_list.append(f"Train-loss: {train_loss:.3f}")
    if val_loss is not None:
        msg_list.append(f"Val-loss: {val_loss:.3f}")
    if train_acc is not None:
        msg_list.append(f"Train-acc: {train_acc:.2f}%")
    if val_acc is not None:
        msg_list.append(f"Val-acc: {val_acc:.2f}%")
    return " | ".join(msg_list)


def generate_and_print_sample(model: GptModel, device: Device, start_context: str, max_new_tokens: int = 50) -> None:
    """
    Generate a sample text from the model given a starting context and print it.

    Note that the model is set to eval mode during generation and then switched back to train mode.

    Args:
        model (GptModel): The trained GPT model.
        device (Device): The device to run the model on.
        start_context (str): The starting context for text generation.
    """
    model.eval()
    encoded_idx = g_tokenizer.text_to_token_ids(start_context).to(device)
    with torch.no_grad():
        token_ids = model.generate_naive(encoded_idx, max_new_tokens, model.pos_emb.weight.shape[0])
        decoded_text = g_tokenizer.token_ids_to_text(token_ids)
        g_logger.info(f"Generated sample:\n{decoded_text}")
    model.train()


def train_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, device: Device,
                n_epochs: int, eval_freq: int = 50, eval_iter: int = 5, start_context: str = None,
                max_new_tokens: int = 50) -> TrainingResults:
    """
    Train the GPT model.

    Logic:
    - For each epoch, iterate over the training data loader.
    - For each batch, perform a learning step: forward pass, loss computation, backward pass, and optimizer step.
    - Track the number of tokens seen and global training step.
    - At specified evaluation frequency, compute and log training and validation losses.
    - Optionally, generate and print a sample text after each epoch.

    Args:
        model (GptModel): The GPT model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (Optimizer): The optimizer to use for training.
        device (Device): The device to run the model on.
        n_epochs (int): The number of training epochs.
        eval_freq (int): The frequency (in steps) to evaluate the model on the validation set. Default is 50.
        eval_iter (int): The number of batches to use for evaluation. Default is 5.
        start_context (str, optional): The starting context for sample generation after each epoch. Defaults to None.
        max_new_tokens (int): The number of new tokens to generate for the sample text. Default is 50.

    Returns:
        TrainingResults: The results of the training process.
    """
    train_losses, val_losses, tokens_seen = [], [], []  # Initialize lists to track losses and tokens/examples seen
    token_count = 0
    global_step = -1
    epoch_batches = len(train_loader)
    total_batches = n_epochs * epoch_batches

    try:
        for epoch in range(1, n_epochs + 1):
            for input_batch, target_batch in train_loader:
                input_batch: Tensor

                # Learning step
                model.train()
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients

                # Tracking progress
                token_count += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = calc_losses(model, train_loader, val_loader, device, eval_iter, calc_loss_batch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    tokens_seen.append(token_count)
                    progress_msg = format_training_progress(epoch, n_epochs, global_step, total_batches, train_loss, val_loss)
                    g_logger.info(progress_msg)

            # Print a sample text after each epoch, if requested
            if start_context is not None:
                generate_and_print_sample(model, device, start_context, max_new_tokens)

    except KeyboardInterrupt:
        g_logger.info("Training interrupted by user. Returning current model state...")

    return TrainingResults(model=model, train_losses=train_losses, val_losses=val_losses, tokens_seen=tokens_seen)


def run_training_flow(config: GptConfig, training_set_path: str, lr: float = 5e-4, n_epochs: int = 10,
                      batch_size: int = 2, weight_decay: float = 0.1, dataset_encoding: str = "utf-8",
                      device_type: str = "auto", seed: int = 123, max_length: int = None, stride: int = None,
                      train_ratio: float = 0.9, eval_freq: int = 5, eval_iter: int = 1,
                      start_context: str = "Every effort moves you", saved_model_path: str = "model.pth",
                      saved_plot_path: str = None) -> TrainingResults:
    """
    Run the full training flow for the GPT model.

    Args:
        config (GptConfig): The configuration for the GPT model.
        training_set_path (str): The path to the training .txt file.
        lr (float): The learning rate for the optimizer. Default is 5e-4.
        n_epochs (int): The number of training epochs. Default is 10.
        batch_size (int): The batch size for training. Default is 2.
        weight_decay (float): The weight decay for the optimizer. Default is 0.1.
        dataset_encoding (str): The encoding of the training .txt file. Default is "utf-8".
        device_type (str): The device to use for training ("cpu", "cuda", "auto"). Default is "auto".
        seed (int): The random seed for reproducibility. Default is 123.
        max_length (int, optional): The maximum sequence length for training samples. Defaults to None, which uses config.context_length.
        stride (int, optional): The stride for sliding window over text. Defaults to None, which uses max_length.
        train_ratio (float): The ratio of data to use for training vs. validation. Default is 0.9.
        eval_freq (int): The frequency (in steps) to evaluate model on validation set. Default is 5.
        eval_iter (int): The number of batches to use for evaluation. Default is 1.
        start_context (str): The starting context for sample generation. Default is "Every effort moves you".
        saved_model_path (str): The path to save the trained model. Default is "model.pth".
        saved_plot_path (str, optional): The path to save the loss plot. Defaults to None, which skips saving the plot.

    Returns:
        TrainingResults: The results of the training process.
    """
    g_logger.info("Running foundation model training flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    max_length = max_length or config.context_length
    stride = stride or max_length
    g_logger.info(f"Using max sequence length of {max_length} and stride of {stride}")

    model = GptModel(config)
    model.to(device)
    g_logger.info(f"Initialized a GPT2 model with config: " + ", ".join(f"{k}={v}" for k, v in vars(config).items()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g_logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")

    with open(training_set_path, "r", encoding=dataset_encoding) as file:
        text_data = file.read()
    g_logger.info(f"Loaded training data from {training_set_path} (size: {len(text_data)} characters)")

    train_loader, val_loader = train_test_split(text_data, max_length, batch_size, stride, train_ratio)
    g_logger.info(f"Created training and validation data loaders with train ratio of {train_ratio}")

    g_logger.info(f"Training for {n_epochs} epochs...")
    training_results = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context
    )
    g_logger.info("Training completed.")

    save_model(model, saved_model_path, optimizer)

    if saved_plot_path:
        epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_losses))
        plot_metrics(epochs_tensor, training_results.tokens_seen, training_results.train_losses,
                     training_results.val_losses, "loss", saved_plot_path, legend_loc="upper right")
        g_logger.info(f"Saved loss plot to {saved_plot_path}")

    g_logger.info("Training flow completed.")
    return training_results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments for training configuration to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """
    parser.add_argument("--training-set-path", type=str, required=True, help="Path to the training .txt file.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--dataset-encoding", type=str, default="utf-8", help="Encoding of the training .txt file.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, auto).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence length for training samples.")
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window over text.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Ratio of data to use for training vs. validation.")
    parser.add_argument("--eval-freq", type=int, default=5, help="Frequency (in steps) to evaluate model on validation set.")
    parser.add_argument("--eval-iter", type=int, default=1, help="Number of batches to use for evaluation.")
    parser.add_argument("--start-context", type=str, default="Every effort moves you", help="Starting context for sample generation.")
    parser.add_argument("--saved-model-path", type=str, default="model.pth", help="Path to save the trained model.")
    parser.add_argument("--saved-plot-path", type=str, default=None, help="Path to save the loss plot.")


def main() -> None:
    """
    Main function to parse command-line arguments and run the training flow. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(
        description="Train a GPT model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(parser)
    add_arguments(parser)
    args = parser.parse_args()

    config = GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.use_qkv_bias,
        kv_window_size=args.kv_window_size
    )

    run_training_flow(
        config=config,
        training_set_path=args.training_set_path,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        dataset_encoding=args.dataset_encoding,
        device_type=args.device,
        seed=args.seed,
        max_length=args.max_length,
        stride=args.stride,
        train_ratio=args.train_ratio,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        saved_model_path=args.saved_model_path,
        make_plot=args.make_plot,
        saved_plot_path=args.saved_plot_path
    )


if __name__ == "__main__":
    main()
