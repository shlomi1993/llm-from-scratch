import argparse
import json
import os
import math
import requests
import time
import torch

from functools import partial
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import InstructionDataset, InstructionDatasetWithMasking, InstructionDatasetPhi
from src.scripts.train import train_model
from src.utils.checkpoint import load_model, save_model
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.losses import calc_loss_batch, calc_loss_loader
from src.utils.ollama import format_input
from src.utils.tokenization import tokenizer as tok
from src.utils.visualization import plot_metrics


class LoraLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLora(nn.Module):

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLora(module, rank, alpha))  # Replace the Linear layer with LinearWithLoRA
        else:
            replace_linear_with_lora(module, rank, alpha)  # Recursively apply the same function to child modules


def custom_collate_fn(batch: list, device: Device, max_allowed_length: int = None) -> tuple[torch.Tensor, torch.Tensor]:

    # Detect if batch contains instruction lengths (tuples) or just sequences (lists)
    has_instruction_lengths = isinstance(batch[0], tuple)

    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for _, item in batch) if has_instruction_lengths else max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for batch_item in batch:

        # Unpack instruction length if present
        if has_instruction_lengths:
            instruction_length, item = batch_item
        else:
            instruction_length = None
            item = batch_item

        item: list[int]
        new_item = item.copy()
        new_item += [tok.PAD_IDX]  # Add an <|endoftext|> token

        # Pad sequences to max_length
        padded = new_item + [tok.PAD_IDX] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == tok.PAD_IDX
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = tok.IGNORE_IDX

        # Mask instruction tokens if instruction length is provided
        if instruction_length is not None:
            targets[:instruction_length - 1] = tok.IGNORE_IDX

        # Optionally truncate to maximum sequence length
        if max_allowed_length is not None:
            inputs = inputs[:max_allowed_length]
            targets = targets[:max_allowed_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def fetch_json(file_path: str, url: str) -> list[dict]:
    if not os.path.exists(file_path):
        g_logger.info(f"Downloading {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        g_logger.info(f"Saved to {file_path}")
    else:
        g_logger.info(f"Loading existing file {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def run_instruction_finetuning_advanced_flow(
    pretrained_model_path: str,
    tuning_set_path: str = None,
    use_alpaca52k: bool = False,
    mask_instructions: bool = False,
    use_phi3_prompt: bool = False,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    train_frac: float = 0.85,
    test_frac: float = 0.1,
    batch_size: int = 8,
    seed: int = 123,
    device_type: str = "auto",
    lr: float = 5e-5,
    n_epochs: int = 2,
    weight_decay: float = 0.1,
    eval_freq: int = 5,
    eval_iter: int = 5,
    loss_plot_save_path: str = None,
    model_save_path: str = "assistant-advanced.pth",
    max_new_tokens: int = 256,
    test_output_path: str = "instruction-test-responses-advanced.json"
) -> None:

    g_logger.info("Starting advanced instruction fine-tuning flow")
    g_logger.warning("\033[93mThis flow is experimental and may not be fully tested.\033[0m")

    # Validate conflicting options
    if mask_instructions and use_phi3_prompt:
        raise ValueError("Simultaneous support for instruction masking and Phi-3 prompt template has not been implemented.")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    # Load pretrained model
    g_logger.info(f"Loading pretrained model from '{pretrained_model_path}'")
    model = load_model(pretrained_model_path, device)[0]
    model.eval()

    # Download and prepare dataset
    if tuning_set_path is None:
        tuning_set_path = "instruction-data.json"
        if use_alpaca52k:
            url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        else:
            url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
        data = fetch_json(tuning_set_path, url)
    else:
        g_logger.info(f"Loading tuning dataset from {tuning_set_path}")
        with open(tuning_set_path, "r") as f:
            data = json.load(f)

    # Split dataset
    train_portion = int(len(data) * train_frac)
    test_portion = int(len(data) * test_frac)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    g_logger.info(f"Dataset split: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} testing samples")

    # Configure dataset and collate function based on options
    allowed_max_length = 512 if use_alpaca52k else model.config.context_length
    customized_collate_fn = partial(custom_collate_fn, device=device, max_allowed_length=allowed_max_length)

    if mask_instructions:
        CustomDataset = InstructionDatasetWithMasking
    elif use_phi3_prompt:
        CustomDataset = InstructionDatasetPhi
    else:
        CustomDataset = InstructionDataset

    # Adjust batch size for large dataset
    if use_alpaca52k:
        batch_size = 4
        g_logger.info(f"Using adjusted batch size of {batch_size} for Alpaca 52k dataset")

    # Create dataloaders
    torch.manual_seed(seed)
    train_loader = DataLoader(
        dataset=CustomDataset(train_data),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        dataset=CustomDataset(val_data),
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # Apply LoRA if requested
    if use_lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        g_logger.info(f"Total trainable parameters before LoRA: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        g_logger.info(f"Total trainable parameters after freezing: {total_params:,}")

        replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        g_logger.info(f"Total trainable LoRA parameters: {total_params:,}")

    g_logger.info("Moving model to device")
    model.to(device)

    # Calculate initial losses
    g_logger.info("Calculating initial losses before fine-tuning")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_batch, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_batch, model, device, n_batches=5)
    g_logger.info(f"   Training loss: {train_loss:.3f}")
    g_logger.info(f"   Validation loss: {val_loss:.3f}")

    # Setup optimizer
    g_logger.info(f"Setting up optimizer with learning rate of {lr} and weight decay of {weight_decay}")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Start fine-tuning
    g_logger.info("Starting instruction fine-tuning...")
    start_time = time.time()
    start_context = InstructionDatasetPhi.format_input_phi(val_data[0]) if use_phi3_prompt else format_input(val_data[0])
    results = train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    g_logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plot losses
    if loss_plot_save_path:
        g_logger.info(f"Saving loss plot to {loss_plot_save_path}")
        epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
        plot_metrics(epochs_tensor, results.tokens_seen, results.train_losses, results.val_losses,
                    label="loss", savefig_path=loss_plot_save_path, legend_loc="upper right", simplify_x_axis=True)

    # Generate responses on test set
    g_logger.info("Generating responses on test set...")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating responses", leave=True):
        input_text = InstructionDatasetPhi.format_input_phi(entry) if use_phi3_prompt else format_input(entry)

        token_ids = model.generate(
            idx=tok.text_to_token_ids(input_text).to(device),
            max_new_tokens=max_new_tokens,
            context_size=model.config.context_length,
            eos_id=tok.PAD_IDX
        )
        generated_text = tok.token_ids_to_text(token_ids)

        if use_phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # Save test responses
    with open(test_output_path, "w") as file:
        json.dump(test_data, file, indent=4)
    g_logger.info(f"Responses saved as {test_output_path}")

    # Save model
    save_model(model, model_save_path, optimizer)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to a pre-trained foundation GPT2 model.")
    parser.add_argument("--tuning-set-path", type=str, default=None, help="Path to the instruction tuning JSON file (downloads default if not provided).")
    parser.add_argument("--use-alpaca52k", action="store_true", help="Use Alpaca 52k dataset instead of default.")
    parser.add_argument("--mask-instructions", action="store_true", help="Mask instruction tokens in loss calculation.")
    parser.add_argument("--use-phi3-prompt", action="store_true", help="Use Phi-3 prompt template.")
    parser.add_argument("--use-lora", action="store_true", help="Apply Low-Rank Adaptation (LoRA).")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank parameter.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha parameter.")
    parser.add_argument("--train-frac", type=float, default=0.85, help="Fraction of data for training.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data for testing.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, mps, auto).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency (in steps).")
    parser.add_argument("--eval-iter", type=int, default=5, help="Number of batches to evaluate.")
    parser.add_argument("--loss-plot-save-path", type=str, default=None, help="Path to save loss plot (None to skip).")
    parser.add_argument("--model-save-path", type=str, default="assistant-advanced.pth", help="Path to save the fine-tuned model.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate for test responses.")
    parser.add_argument("--test-output-path", type=str, default="instruction-test-responses-advanced.json", help="Path to save test responses JSON.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced instruction fine-tuning with LoRA, masking, and alternative prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    run_instruction_finetuning_advanced_flow(
        pretrained_model_path=args.pretrained_model_path,
        tuning_set_path=args.tuning_set_path,
        use_alpaca52k=args.use_alpaca52k,
        mask_instructions=args.mask_instructions,
        use_phi3_prompt=args.use_phi3_prompt,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
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
        test_output_path=args.test_output_path
    )


if __name__ == "__main__":
    main()
