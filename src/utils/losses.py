import torch

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from src.model.gpt import GptModel
from src.utils.device import Device


LossFunc = Callable[[Tensor, Tensor, GptModel, Device], Tensor]  # A custom type for loss functions


def calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    """
    Calculate the cross-entropy loss for a batch of input and target tokens.

    Args:
        input_batch (Tensor): A batch of input token IDs of shape (batch_size, seq_len).
        target_batch (Tensor): A batch of target token IDs of shape (batch_size, seq_len).
        model (GptModel): The GPT model used for generating logits.
        device (Device): The device to perform computations on.

    Returns:
        Tensor: The cross-entropy (negative log likelihood) loss for the batch.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss  # Negative average log probability


def calc_loss_last_token(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    """
    Calculate the cross-entropy loss for the last token in each sequence of the batch.

    Args:
        input_batch (Tensor): A batch of input token IDs of shape (batch_size, seq_len).
        target_batch (Tensor): A batch of target token IDs of shape (batch_size,).
        model (GptModel): The GPT model used for generating logits.
        device (Device): The device to perform computations on.

    Returns:
        Tensor: The cross-entropy (negative log likelihood) loss for the last token in the batch.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss  # Negative average log probability for last token


def calc_loss_loader(data_loader: DataLoader, criterion: LossFunc, model: GptModel, device: Device, n_batches: int = None) -> float:
    """
    Calculate the average loss over a DataLoader using the specified loss function.

    Args:
        data_loader (DataLoader): The DataLoader providing batches of (input, target) pairs.
        criterion (LossFunc): The loss function to use for calculating loss per batch.
        model (GptModel): The GPT model used for generating logits.
        device (Device): The device to perform computations on.
        n_batches (int, optional): The number of batches to evaluate. If None, evaluates all batches. Default is None.

    Returns:
        float: The average loss across all batches.
    """
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.
    n_batches = min(n_batches, len(data_loader)) if n_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= n_batches:
            break

        loss: Tensor = criterion(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / n_batches


def calc_losses(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: Device, n_batches: int,
                criterion: LossFunc) -> tuple[float, float]:
    """
    Calculate training and validation losses over specified DataLoaders.

    Note: The model is set to evaluation mode during loss calculation and reverted back to training mode afterwards.

    Args:
        model (GptModel): The GPT model used for generating logits.
        train_loader (DataLoader): The DataLoader for the training dataset.
        val_loader (DataLoader): The DataLoader for the validation dataset.
        device (Device): The device to perform computations on.
        n_batches (int): The number of batches to evaluate.
        criterion (LossFunc): The loss function to use for calculating loss per batch.

    Returns:
        tuple[float, float]: A tuple containing the average training loss and the average validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, criterion, model, device, n_batches)
        val_loss = calc_loss_loader(val_loader, criterion, model, device, n_batches)
    model.train()
    return train_loss, val_loss
