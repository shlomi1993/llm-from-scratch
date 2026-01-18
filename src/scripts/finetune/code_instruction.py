import argparse
import json
import time
import torch

from functools import partial
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.dataset import AlpacaCodeDataset
from src.scripts.train import train_model
from src.scripts.finetune.instruction import test_assistant
from src.utils.checkpoint import save_model, load_model
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.losses import calc_loss_loader, calc_loss_batch
from src.utils.tokenization.tokenizer import EOT_TOK, EOT_IDX, IGNORE_IDX, g_tokenizer
from src.utils.visualization import plot_metrics



def coding_collate_fn(batch: list[Tensor], device: Device, pad_token_id: int = EOT_IDX,
                      ignore_index: int = IGNORE_IDX, max_allowed_length: int = None) -> tuple[Tensor, Tensor]:
    """
    Pads sequences and masks the Instruction part so the model only trains on the Code.

    Args:
        batch (list[Tensor]): List of token ID tensors.
        device (Device): Device to place tensors on.
        pad_token_id (int): Token ID used for padding.
        ignore_index (int): Target token ID to ignore in loss computation.
        max_allowed_length (int, optional): Maximum length for truncation.

    Returns:
        tuple[Tensor, Tensor]: Padded input and target tensors.
    """

    # Convert batch to a list of lists
    batch_lists = [item.tolist() for item in batch]
    batch_max_length = max(len(item) + 1 for item in batch_lists)

    # Prepare separator for masking logic
    sep_ids = g_tokenizer.encode(AlpacaCodeDataset.RESPONSE_SEPARATOR, allowed_special={EOT_TOK})
    sep_len = len(sep_ids)
    sep_tensor = torch.tensor(sep_ids, device=device)

    inputs_lst, targets_lst = [], []
    for item in batch_lists:
        # Pad sequence
        padded = item + [pad_token_id] * (batch_max_length - len(item))

        # Create Inputs (0..N-1) and Targets (1..N)
        inputs = torch.tensor(padded[:-1], device=device)
        targets = torch.tensor(padded[1:], device=device)

        # Mask Padding in Targets
        mask_pad = targets == pad_token_id
        if mask_pad.any():
            targets[mask_pad] = ignore_index

        # Mask instruction part in Targets
        windows = inputs.unfold(0, sep_len, 1)
        matches = (windows == sep_tensor).all(dim=1)
        nonzero = matches.nonzero(as_tuple=True)[0]

        # If separator found, mask up to its end, else mask all
        if len(nonzero) > 0:
            sep_start_idx = nonzero[0].item()
            mask_end = sep_start_idx + sep_len - 1  # mask up to end of separator
            mask_end = min(mask_end, len(targets))  # boundary check
            targets[:mask_end] = ignore_index
        else:
            targets[:] = ignore_index  # mask all if separator not found

        # Truncate to context length
        if max_allowed_length:
            inputs = inputs[:max_allowed_length]
            targets = targets[:max_allowed_length]

        # Append to batch lists
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Return stacked tensors
    return torch.stack(inputs_lst), torch.stack(targets_lst)


def create_coding_dataloaders(dataset_path: str, train_frac: float, test_frac: float, device: Device,
                              batch_size: int = None, max_allowed_length: int = 1024, seed: int = 123,
                              max_samples: int = None, n_workers: int = 0) -> tuple[DataLoader, DataLoader, list[dict]]:
    """
    Loads the AlpacaCodeDataset, splits it into Train/Val/Test, and returns DataLoaders configured with the coding_collate_fn.

    Args:
        dataset_path (str): Path to the AlpacaCodeDataset folder.
        train_frac (float): Fraction of data to use for training.
        test_frac (float): Fraction of data to use for testing.
        device (Device): Device to place tensors on.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to None.
        max_allowed_length (int, optional): Maximum sequence length for truncation. Defaults to 1024.
        seed (int, optional): Random seed for reproducibility. Defaults to 123.
        max_samples (int, optional): Maximum number of samples to load. Defaults to None.
        n_workers (int, optional): Number of worker processes for data loading. Defaults to 0.

    Returns:
        tuple[DataLoader, DataLoader, list[dict]]: Train and validation DataLoaders, and raw test data entries.
    """

    # Load the full dataset wrapper
    full_dataset = AlpacaCodeDataset(dataset_path, max_length=max_allowed_length, max_samples=max_samples)
    total_len = len(full_dataset)

    # Calculate split indices
    torch.manual_seed(seed)
    indices = torch.randperm(total_len).tolist()
    train_end = int(total_len * train_frac)
    test_end = train_end + int(total_len * test_frac)
    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    val_indices = indices[test_end:]

    # Create Subsets
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)

    # Prepare test data (raw dicts) for evaluation
    test_data_raw = [full_dataset.dataset[i] for i in test_indices]

    g_logger.info(f"Dataset split: {len(train_data)} training, {len(val_data)} validation, {len(test_data_raw)} testing samples")

    # Bind device to collate function
    collate = partial(coding_collate_fn, device=device, max_allowed_length=max_allowed_length)

    # Create standard DataLoaders with our custom collate
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate, shuffle=False, drop_last=False, num_workers=n_workers)

    # Return the DataLoaders and raw test data
    return train_loader, val_loader, test_data_raw


def run_coding_finetuning_flow(pretrained_model_path: str, tuning_set_path: str, train_frac: float = 0.85,
                               test_frac: float = 0.1, batch_size: int = 8, seed: int = 123,
                               device_type: str = "auto", lr: float = 5e-5, n_epochs: int = 2,
                               weight_decay: float = 0.1, eval_freq: int = 5, eval_iter: int = 5,
                               loss_plot_save_path: str = None, model_save_path: str = "coder.pth",
                               max_new_tokens: int = 256, test_output_path: str = "coder-test-responses.json",
                               evaluate: bool = False, max_samples: int = None) -> None:
    """
    Fine-tunes a GPT-2 model on code instruction-following data.

    Args:
        pretrained_model_path (str): Path to the pre-trained GPT-2 model.
        tuning_set_path (str): Path to the tuning dataset.
        train_frac (float): Fraction of data to use for training.
        test_frac (float): Fraction of data to use for testing.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        device_type (str): Device type to use ('cpu', 'cuda', 'mps', 'auto').
        lr (float): Learning rate for the optimizer.
        n_epochs (int): Number of training epochs.
        weight_decay (float): Weight decay for the optimizer.
        eval_freq (int): Frequency of evaluation during training (in epochs).
        eval_iter (int): Number of batches to use for evaluation.
        loss_plot_save_path (str): Path to save the loss plot image.
        model_save_path (str): Path to save the fine-tuned model.
        max_new_tokens (int): Maximum number of tokens to generate during testing.
        test_output_path (str): Path to save the test set responses.
        evaluate (bool): Whether to evaluate the model after training.
        max_samples (int, optional): Maximum number of samples to use from the dataset for debugging
    """
    g_logger.info("Running code instruction finetuning flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    model = load_model(pretrained_model_path, device)[0]
    model.eval()

    train_loader, val_loader, test_data = create_coding_dataloaders(
        tuning_set_path, train_frac, test_frac, device, batch_size, model.config.context_length, seed, max_samples, n_workers=0
    )
    g_logger.info(f"Created DataLoaders with batch size {batch_size}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_batch, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_batch, model, device, n_batches=5)
    g_logger.info(f"Loss before fine-tuning: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g_logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")

    # Prompt format used in dataset and reset seed
    formatted_input = AlpacaCodeDataset.format_input(test_data[0])
    torch.manual_seed(seed)

    g_logger.info("Starting code instruction fine-tuning...")
    start_time = time.time()
    results = train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter,
                          formatted_input, max_new_tokens)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    g_logger.info(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    save_model(model, model_save_path, optimizer)

    if loss_plot_save_path:
        epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
        plot_metrics(epochs_tensor, results.tokens_seen, results.train_losses, results.val_losses, label="loss",
                     savefig_path=loss_plot_save_path, legend_loc="upper right")

    test_assistant(model, test_data, device, max_new_tokens, test_output_path, AlpacaCodeDataset.format_input, evaluate, seed)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds command-line arguments for the coding finetuning flow.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser instance.
    """
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to base GPT2 model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to Alpaca Arrow dataset folder")
    parser.add_argument("--train-frac", type=float, default=0.85, help="Fraction of data to use for training")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data to use for testing")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, mps, auto)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--eval-iter", type=int, default=5, help="Number of batches to use for evaluation")
    parser.add_argument("--model-save-path", type=str, default="coder.pth", help="Path to save the finetuned coder model")
    parser.add_argument("--loss-plot-save-path", type=str, default=None, help="Path to save loss plot image")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum code tokens to generate")
    parser.add_argument("--test-output-path", type=str, default="coder_results.json", help="Path to save test set responses")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for debugging")


def main() -> None:
    """
    Main function to run the coding finetuning flow. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    run_coding_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        tuning_set_path=args.dataset_path,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        model_save_path=args.model_save_path,
        max_new_tokens=args.max_new_tokens,
        test_output_path=args.test_output_path,
        evaluate=args.evaluate,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
