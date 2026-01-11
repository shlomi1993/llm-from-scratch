import argparse
import json
import time
import torch

from functools import partial
from logging import getLogger as get_logger
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Callable

from src.datasets import AlpacaCodeDataset
from src.model.gpt import GptModel
from src.scripts.common import calc_loss_loader, calc_loss_batch, save_model, load_model
from src.scripts.pretrain import train_foundation_model as finetune_assistant
from src.utils.device import Device, get_device
from src.utils.ollama import OllamaEvaluator
from src.utils.tokenization import tokenizer, EOT, PAD_TOKEN_ID, IGNORE_INDEX, text_to_token_ids, token_ids_to_text
from src.utils.visualization import plot_metrics


_logger = get_logger(__name__)


def coding_collate_fn(batch: list[Tensor], device: Device, pad_token_id: int = PAD_TOKEN_ID,
                      ignore_index: int = IGNORE_INDEX, max_allowed_length: int = None) -> tuple[Tensor, Tensor]:
    """
    Pads sequences and masks the Instruction part so the model only trains on the Code.
    """
    # Convert batch to a list of lists
    batch_lists = [item.tolist() for item in batch]
    batch_max_length = max(len(item) + 1 for item in batch_lists)

    # Prepare separator for masking logic
    sep_ids = tokenizer.encode(AlpacaCodeDataset.RESPONSE_SEPARATOR, allowed_special={EOT})
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
                              batch_size: int = None, max_allowed_length: int = 1024, n_workers: int = 0,
                              seed: int = 123, max_samples: int = None) -> tuple[DataLoader, DataLoader, list[dict]]:
    """
    Loads the AlpacaCodeDataset, splits it into Train/Val/Test, and returns DataLoaders configured with the coding_collate_fn.
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

    _logger.info(f"Dataset split: {len(train_data)} training, {len(val_data)} validation, {len(test_data_raw)} testing samples")

    # Bind device to collate function
    collate = partial(coding_collate_fn, device=device, max_allowed_length=max_allowed_length)

    # Create standard DataLoaders with our custom collate
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate, shuffle=False, drop_last=False, num_workers=n_workers)

    # Return the DataLoaders and raw test data
    return train_loader, val_loader, test_data_raw


def test_coder(model: GptModel, test_data: list[dict], device: Device, max_new_tokens: int,
               coding_format_input: Callable, test_output_path: str) -> None:
    _logger.info("Generating responses for test set...")
    for entry in tqdm(test_data, desc="Testing"):
        prompt = coding_format_input(entry)
        token_ids = model.generate(
            idx=text_to_token_ids(prompt).to(device),
            max_new_tokens=max_new_tokens,
            context_size=model.config.context_length,
            eos_id=PAD_TOKEN_ID
        )
        gen_text = token_ids_to_text(token_ids)
        response = gen_text[len(prompt):].strip()
        entry["model_response"] = response
    with open(test_output_path, "w") as f:
        json.dump(test_data, f, indent=4)
    _logger.info(f"Responses saved as {test_output_path}")


def run_coding_finetuning_flow(pretrained_model_path: str, dataset_path: str, train_frac: float = 0.85,
                               test_frac: float = 0.1, batch_size: int = 8, seed: int = 123,
                               device_type: str = "auto", lr: float = 5e-5, n_epochs: int = 2,
                               weight_decay: float = 0.1, eval_freq: int = 5, eval_iter: int = 5,
                               loss_plot_save_path: str = None, model_save_path: str = "coder.pth",
                               max_new_tokens: int = 256, test_output_path: str = "coder-test-responses.json",
                               evaluate: bool = False, max_samples: int = None) -> None:

    _logger.info("Starting codein finetuning flow")

    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}")

    _logger.info("Loading pre-trained model...")
    model, _ = load_model(pretrained_model_path, device)  # Ignore optimizer as we create a new one
    model.eval()
    model.to(device)

    _logger.info("Preparing coding dataset...")
    train_loader, val_loader, test_data = create_coding_dataloaders(
        dataset_path, train_frac, test_frac, device, batch_size, max_allowed_length=model.config.context_length,
        n_workers=0, seed=seed, max_samples=max_samples
    )

    _logger.info("Calculating initial losses...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_batch, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_batch, model, device, n_batches=5)
    _logger.info(f"   Training loss: {train_loss:.4f}")
    _logger.info(f"   Validation loss: {val_loss:.4f}")

    _logger.info(f"Setting up optimizer (LR={lr})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prompt format used in dataset
    coding_format_input = lambda x: f"### Instruction:\n{x['instruction']}\n\n### Response:\n"

    _logger.info("Starting training loop...")
    start_time = time.time()

    # Reuse the assistant finetuning loop
    results = finetune_assistant(
        model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, coding_format_input(test_data[0])
    )

    end_time = time.time()
    _logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

    save_model(model, model_save_path, optimizer)
    _logger.info(f"Model saved to {model_save_path}")

    if loss_plot_save_path:
        _logger.info("Plotting metrics...")
        epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
        plot_metrics(epochs_tensor, results.tokens_seen, results.train_losses, results.val_losses, label="loss",
                     savefig_path=loss_plot_save_path)

    # Generating responses for test set
    test_coder(model, test_data, device, max_new_tokens, coding_format_input, test_output_path)

    if evaluate:
        _logger.info("Evaluating with Ollama...")
        evaluator = OllamaEvaluator(seed=seed)
        avg_score, _ = evaluator.evaluate(test_output_path)
        _logger.info(f"Avg Score: {avg_score:.2f}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to base GPT2 model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to Alpaca Arrow dataset folder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for debugging")
    parser.add_argument("--model-save-path", type=str, default="coder.pth")
    parser.add_argument("--test-output-path", type=str, default="coder_results.json")
    parser.add_argument("--loss-plot-save-path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    run_coding_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device_type=args.device,
        model_save_path=args.model_save_path,
        test_output_path=args.test_output_path,
        evaluate=args.evaluate,
        max_samples=args.max_samples,
        loss_plot_save_path=args.loss_plot_save_path
    )

if __name__ == "__main__":
    main()
