import argparse
import json
import time
import torch
import tqdm

from functools import partial
from logging import getLogger as get_logger
from torch.utils.data import DataLoader

from src.data.datasets import InstructionDataset
from src.data.formatting import format_input
from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.common import calc_loss_loader, calc_loss_batch, load_model
from src.scripts.pretrain import train_foundation_model as finetune_assistant
from src.utils.device import Device, get_device
from src.utils.tokenization.tokenizer import PAD_TOKEN_ID, IGNORE_INDEX, text_to_token_ids, token_ids_to_text
from src.utils.visualization import plot_metrics


_logger = get_logger(__name__)


def custom_collate_fn(batch: list[int], device: Device, pad_token_id: int = PAD_TOKEN_ID,
                      ignore_index: int = IGNORE_INDEX, max_allowed_length: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and create inputs and targets
    inputs_lst, targets_lst = [], []
    for item in batch:
        item: list[int]
        new_item = item.copy()

        # Add an <|endoftext|> token
        new_item += [pad_token_id]

        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length
        if max_allowed_length:
            inputs = inputs[:max_allowed_length]
            targets = targets[:max_allowed_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def create_dataloaders(tuning_set_path: str, train_frac: float, device: Device, batch_size: int = None,
                       max_allowed_length: int = 1024, n_workers: int = 0, seed: int = 123) -> tuple[DataLoader, DataLoader]:

    # Load the tuning dataset
    with open(tuning_set_path, "r") as f:
        tuning_data = json.load(f)

    # Split the dataset
    train_portion = int(len(tuning_data) * train_frac)  # 85% for training

    # Split the dataset into training and validation sets
    train_data = tuning_data[:train_portion]
    val_data = tuning_data[train_portion:]
    _logger.info(f"Dataset split: {len(train_data)} training, {len(val_data)} validation samples")

    # Partially initialize the collate function with device and max length
    customized_collate_fn = partial(custom_collate_fn, device=device, max_allowed_length=max_allowed_length)

    # Create DataLoaders
    torch.manual_seed(seed)
    train_loader = DataLoader(
        dataset=InstructionDataset(train_data),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers
    )
    val_loader = DataLoader(
        dataset=InstructionDataset(val_data),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers
    )
    return train_loader, val_loader


def load_assistant(model_path: str, config: GptConfig, device: Device) -> GptModel:

    # Create model with the same architecture
    model = GptModel(config)

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    return model


def extract_response(response_text: str, input_text: str) -> str:
    return response_text[len(input_text):].replace("### Response:", "").strip()


def generate_response(model: GptModel, context_length: int, prompt: str, max_new_tokens: int = 35, seed: int = 123) -> str:
    torch.manual_seed(seed)
    token_ids = model.generate(
        idx=text_to_token_ids(prompt),
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        eos_id=PAD_TOKEN_ID
    )
    response = token_ids_to_text(token_ids)
    response = extract_response(response, prompt)
    return response


def run_instruction_finetuning_flow(pretrained_model_path: str, tuning_set_path: str, train_frac: float = 0.85,
                                    batch_size: int = 8, seed: int = 123, device_type: str = "auto", lr: float = 5e-5,
                                    n_epochs: int = 2, weight_decay: float = 0.1, eval_freq: int = 5,
                                    eval_iter: int = 5, loss_plot_save_path: str = None,
                                    model_save_path: str = "assistant.pth") -> None:

    _logger.info("Starting instruction finetuning flow")

    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}")

    _logger.info("Loading pre-trained model on device")
    model = load_model(pretrained_model_path, device)[0]
    model.eval()
    model.to(device)

    _logger.info("Preparing instruction fine-tuning dataset")
    train_loader, val_loader = create_dataloaders(
        tuning_set_path, train_frac, device, batch_size, max_allowed_length=model.config.context_length,
        n_workers=0, seed=seed
    )

    _logger.info("Calculating initial losses before fine-tuning")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_batch, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_batch, model, device, n_batches=5)
    _logger.info(f"   Training loss: {train_loss}")
    _logger.info(f"   Validation loss: {val_loss}")

    _logger.info(f"Setting up optimizer with learning rate {lr} and weight decay {weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    torch.manual_seed(seed)
    _logger.info("Starting instruction fine-tuning...")
    start_time = time.time()
    formatted_input = format_input(val_loader.dataset.data[0])
    results = finetune_assistant(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, formatted_input)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    _logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    _logger.info("Final evaluation on validation set")
    epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
    plot_metrics(epochs_tensor, results.tokens_seen, results.train_losses, results.val_losses, label="loss",
                 savefig_path=loss_plot_save_path, legend_loc="upper right")

    torch.save(model.state_dict(), model_save_path)
    _logger.info(f"Model saved as {model_save_path}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to a pre-trained foundation GPT2 model.")
    parser.add_argument("--tuning-set-path", type=str, required=True, help="Path to the instruction tuning JSON file.")
    parser.add_argument("--train-frac", type=float, default=0.85, help="Fraction of data for training (rest is validation).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, mps, auto).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency (in steps).")
    parser.add_argument("--eval-iter", type=int, default=5, help="Number of batches to evaluate.")
    parser.add_argument("--loss-plot-save-path", type=str, default=None, help="Path to save loss plot (None to skip).")
    parser.add_argument("--model-save-path", type=str, default="assistant.pth", help="Path to save the fine-tuned model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT model for instruction following.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    run_instruction_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        tuning_set_path=args.tuning_set_path,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        model_save_path=args.model_save_path
    )


if __name__ == "__main__":
    main()
