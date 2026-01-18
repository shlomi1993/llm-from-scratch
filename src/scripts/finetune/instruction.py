import argparse
import json
import time
import torch

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

from src.dataset import InstructionDataset
from src.model.gpt import GptModel
from src.scripts.train import train_model
from src.utils.checkpoint import save_model, load_model
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.losses import calc_loss_loader, calc_loss_batch
from src.utils.ollama import OllamaEvaluator, format_input
from src.utils.tokenization.tokenizer import EOT_IDX, IGNORE_IDX, g_tokenizer
from src.utils.visualization import plot_metrics


def instruction_collate_fn(batch: list[int], device: Device, pad_token_id: int = EOT_IDX,
                           ignore_index: int = IGNORE_IDX, max_allowed_length: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for instruction tuning dataset.

    Args:
        batch (list[list[int]]): List of token ID sequences.
        device (Device): Target device for tensors.
        pad_token_id (int): Token ID used for padding.
        ignore_index (int): Index to ignore in loss computation.
        max_allowed_length (int, optional): Maximum allowed sequence length. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Collated input and target tensors.
    """

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


def create_instruction_dataloaders(tuning_set_path: str, train_frac: float, test_frac: float, device: Device,
                                   batch_size: int = None, max_allowed_length: int = 1024, n_workers: int = 0,
                                   seed: int = 123) -> tuple[DataLoader, DataLoader, list[dict]]:
    """
    Create DataLoaders for instruction tuning.

    Args:
        tuning_set_path (str): Path to the tuning JSON file.
        train_frac (float): Fraction of data to use for training.
        test_frac (float): Fraction of data to use for testing.
        device (Device): Target device for tensors.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to None.
        max_allowed_length (int, optional): Maximum allowed sequence length. Defaults to 1024.
        n_workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to 123.

    Returns:
        tuple[DataLoader, DataLoader, list[dict]]: Train, validation, and test DataLoaders.
    """

    # Load the tuning dataset
    with open(tuning_set_path, "r") as f:
        tuning_data = json.load(f)

    # Split the dataset
    train_portion = int(len(tuning_data) * train_frac)  # 85% for training
    test_portion = int(len(tuning_data) * test_frac)    # 10% for testing

    # Split the dataset into training, testing, and validation sets
    train_data = tuning_data[:train_portion]
    test_data = tuning_data[train_portion:train_portion + test_portion]
    val_data = tuning_data[train_portion + test_portion:]
    g_logger.info(f"Dataset split: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} testing samples")

    # Partially initialize the collate function with device and max length
    customized_collate_fn = partial(instruction_collate_fn, device=device, max_allowed_length=max_allowed_length)

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
    return train_loader, val_loader, test_data  # NOTE: Test data is returned as a list of dicts


def test_assistant(model: GptModel, test_data: list[dict], device: Device, max_new_tokens: int,
                   test_output_path: str, format_func: Callable, evaluate: bool = False, seed: int = 123) -> None:
    """
    Generate responses for the test dataset, save them in the given test_data dict, and optionally evaluate them.

    Notes:
    1. This function sets the model to eval mode during generation and back to train mode afterwards.
    2. If evaluation is enabled, it uses the Ollama API to score the responses
    3. The test_data list is modified in-place to include the model responses.

    Args:
        model (GptModel): The instruction-finetuned model to test.
        test_data (list[dict]): List of test samples.
        device (Device): Target device for tensors.
        max_new_tokens (int): Maximum number of tokens to generate for each response.
        test_output_path (str): Path to save the test responses JSON.
        format_func (Callable): Function to format the prompt from a test sample.
        evaluate (bool, optional): Whether to evaluate the model responses using Ollama API. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 123.
    """

    # Generate responses
    g_logger.info("Generating model responses...")
    model.eval()
    for entry in tqdm(test_data, total=len(test_data), desc="Generating responses", leave=True):
        prompt = format_func(entry)
        token_ids = model.generate(
            idx=g_tokenizer.text_to_token_ids(prompt).to(device),
            max_new_tokens=max_new_tokens,
            context_size=model.config.context_length,
            eos_id=EOT_IDX
        )
        generated_text = g_tokenizer.token_ids_to_text(token_ids)
        response = generated_text[len(prompt):].replace("### Response:", "").strip()
        entry["model_response"] = response  # Add response to the entry in-place
    model.train()

    # Save responses
    with open(test_output_path, "w") as file:
        json.dump(test_data, file, indent=4)
    g_logger.info(f"Responses saved as {test_output_path}")

    # Evaluation
    if evaluate:
        g_logger.info("Evaluating responses with Ollama...")
        evaluator = OllamaEvaluator(seed=seed, formatter=format_func)
        avg_score, scores = evaluator.evaluate(test_output_path)
        g_logger.info(f"Average score {avg_score:.2f}% across {len(scores)} samples")



def run_instruction_finetuning_flow(pretrained_model_path: str, tuning_set_path: str, train_frac: float = 0.85,
                                    test_frac: float = 0.1, batch_size: int = 8, seed: int = 123,
                                    device_type: str = "auto", lr: float = 5e-5, n_epochs: int = 2,
                                    weight_decay: float = 0.1, eval_freq: int = 5, eval_iter: int = 5,
                                    loss_plot_save_path: str = None, model_save_path: str = "assistant.pth",
                                    max_new_tokens: int = 256, test_output_path: str = "instruction-test-responses.json",
                                    evaluate: bool = False) -> None:
    """
    Fine-tune a GPT model for instruction following.

    Args:
        pretrained_model_path (str): Path to a pre-trained foundation GPT2 model.
        tuning_set_path (str): Path to the instruction tuning JSON file.
        train_frac (float): Fraction of data to use for training.
        test_frac (float): Fraction of data to use for testing.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        device_type (str): Device to use for training (cpu, cuda, mps, auto).
        lr (float): Learning rate for the optimizer.
        n_epochs (int): Number of training epochs.
        weight_decay (float): Weight decay for the optimizer.
        eval_freq (int): Evaluation frequency (in steps).
        eval_iter (int): Number of batches to evaluate.
        loss_plot_save_path (str, optional): Path to save loss plot (None to skip). Defaults to None.
        model_save_path (str, optional): Path to save the fine-tuned model. Defaults to "assistant.pth".
        max_new_tokens (int): Maximum number of tokens to generate for test responses.
        test_output_path (str): Path to save test responses JSON.
        evaluate (bool): Whether to evaluate the model responses using Ollama API.
    """
    g_logger.info("Running instruction finetuning flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    model = load_model(pretrained_model_path, device)[0]
    model.eval()
    model.to(device)  # Maybe redundant, but safe for comparison with notebook runs

    train_loader, val_loader, test_data = create_instruction_dataloaders(
        tuning_set_path, train_frac, test_frac, device, batch_size, model.config.context_length,
        n_workers=0, seed=seed
    )
    g_logger.info(f"Created DataLoaders with batch size {batch_size}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, calc_loss_batch, model, device, n_batches=5)
        val_loss = calc_loss_loader(val_loader, calc_loss_batch, model, device, n_batches=5)
    g_logger.info(f"Loss before fine-tuning: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g_logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")

    # Prompt format used in dataset and reset seed
    formatted_input = format_input(val_loader.dataset.data[0])
    torch.manual_seed(seed)

    g_logger.info("Starting instruction fine-tuning...")
    start_time = time.time()
    results = train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, formatted_input)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    g_logger.info(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    save_model(model, model_save_path, optimizer)

    if loss_plot_save_path:
        epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
        plot_metrics(epochs_tensor, results.tokens_seen, results.train_losses, results.val_losses, label="loss",
                     savefig_path=loss_plot_save_path, legend_loc="upper right")


    test_assistant(model, test_data, device, max_new_tokens, test_output_path, format_input, evaluate, seed)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments for instruction fine-tuning.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which to add arguments.
    """
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to a pre-trained foundation GPT2 model.")
    parser.add_argument("--tuning-set-path", type=str, required=True, help="Path to the instruction tuning JSON file.")
    parser.add_argument("--train-frac", type=float, default=0.85, help="Fraction of data for training.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data for testing.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use for tuning.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--n-epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency (in steps).")
    parser.add_argument("--eval-iter", type=int, default=5, help="Number of batches to evaluate.")
    parser.add_argument("--loss-plot-save-path", type=str, default=None, help="Path to save loss plot (None to skip).")
    parser.add_argument("--model-save-path", type=str, default="assistant.pth", help="Path to save the fine-tuned model.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate for test responses.")
    parser.add_argument("--test-output-path", type=str, default="instruction-test-responses.json", help="Path to save test responses JSON.")
    parser.add_argument("--evaluate", action="store_true", help="Whether to evaluate the model responses using Ollama API.")


def main() -> None:
    """
    Main function to run instruction fine-tuning from command-line. Called when the script is executed directly.
    """
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
        evaluate=args.evaluate
    )


if __name__ == "__main__":
    main()
