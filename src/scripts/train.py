import argparse
import torch

from dataclasses import dataclass
from logging import getLogger as get_logger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data_sets import GptDatasetV1
from src.model.config import GptConfig, add_arguments as add_gpt_config_arguments
from src.model.gpt import GptModel
from src.scripts.common import calc_loss_batch, evaluate_losses, save_model
from src.utils.device import Device, get_device
from src.utils.tokenization import text_to_token_ids, token_ids_to_text
from src.utils.visualization import plot_metrics


_logger = get_logger(__name__)


@dataclass
class TrainingResults:
    model: GptModel
    train_losses: list[float]
    val_losses: list[float]
    tokens_seen: list[int] = None

    def breakdown(self):
        return self.model, self.train_losses, self.val_losses, self.tokens_seen


def train_test_split(text: str, max_length: int, batch_size: int, stride: int = None, train_ratio: float = 0.9) -> tuple[DataLoader, DataLoader]:
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    stride = stride or max_length
    train_dataset = GptDatasetV1(train_text, max_length, stride)
    val_dataset = GptDatasetV1(val_text, max_length, stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader


def generate_and_print_sample(model: GptModel, device: Device, start_context: str) -> None:
    model.eval()
    encoded = text_to_token_ids(start_context).to(device)
    with torch.no_grad():
        token_ids = model.generate_naive(idx=encoded, max_new_tokens=50, context_size=model.pos_emb.weight.shape[0])
        decoded_text = token_ids_to_text(token_ids)
        _logger.info("Generated sample: " + decoded_text.replace("\n", " "))
    model.train()


def train_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, device: Device,
                n_epochs: int, eval_freq: int = 50, eval_iter: int = 5, start_context: str = None) -> TrainingResults:

    train_losses, val_losses, tokens_seen = [], [], []  # Initialize lists to track losses and tokens/examples seen
    token_count = 0
    global_step = -1

    try:
        for epoch in range(1, n_epochs + 1):
            _logger.info(f"Epoch {epoch}/{n_epochs}:")

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
                    train_loss, val_loss = evaluate_losses(model, train_loader, val_loader, device, eval_iter, calc_loss_batch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    tokens_seen.append(token_count)
                    _logger.info(f"  Step {global_step} loss: Train {train_loss:.3f}, Val {val_loss:.3f}")

            # Print a sample text after each epoch
            if start_context is not None:
                generate_and_print_sample(model, device, start_context)

    except KeyboardInterrupt:
        _logger.info("Training interrupted by user. Returning current model state...")


    return TrainingResults(model=model, train_losses=train_losses, val_losses=val_losses, tokens_seen=tokens_seen)


def run_model_training_flow(config: GptConfig, training_set_path: str, lr: float = 5e-4, n_epochs: int = 10,
                            batch_size: int = 2, weight_decay: float = 0.1, dataset_encoding: str = "utf-8",
                            device_type: str = "auto", seed: int = 123, max_length: int = None, stride: int = None,
                            train_ratio: float = 0.9, eval_freq: int = 5, eval_iter: int = 1,
                            start_context: str = "Every effort moves you", saved_model_path: str = "model.pth",
                            saved_plot_path: str = None) -> TrainingResults:

    _logger.info("Running foundation model training flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}")

    max_length = max_length or config.context_length
    stride = stride or max_length

    _logger.info(f"Loading training data from {training_set_path}")
    with open(training_set_path, "r", encoding=dataset_encoding) as file:
        text_data = file.read()

    _logger.info(f"Initializing GPT model with config: " + ", ".join(f"{k}={v}" for k, v in vars(config).items()))
    model = GptModel(config)
    model.to(device)

    _logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    _logger.info(f"Creating training and validation data loaders with train ratio of {train_ratio}")
    train_loader, val_loader = train_test_split(text_data, max_length, batch_size, stride, train_ratio)

    _logger.info(f"Training for {n_epochs} epochs...")
    training_results = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context
    )

    _logger.info(f"Saving pre-trained model to {saved_model_path}")
    save_model(model, saved_model_path, optimizer)

    if saved_plot_path:
        _logger.info(f"Saving loss plot to {saved_plot_path}")
        epochs_tensor = torch.linspace(0, n_epochs, len(training_results.train_losses))
        plot_metrics(epochs_tensor, training_results.tokens_seen, training_results.train_losses,
                     training_results.val_losses, "loss", saved_plot_path, legend_loc="upper right")

    _logger.info("Training flow completed.")
    return training_results


def add_arguments(parser: argparse.ArgumentParser) -> None:
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

    run_model_training_flow(
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
