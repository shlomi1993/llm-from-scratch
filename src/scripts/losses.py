import torch

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from src.model.gpt import GptModel
from src.utils.device import Device


LossFunc = Callable[[Tensor, Tensor, GptModel, Device], Tensor]


def calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss  # Negative average log probability


def calc_loss_last_token(input_batch: Tensor, target_batch: Tensor, model: GptModel, device: Device) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: Tensor = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss  # Negative average log probability for last token


def calc_loss_loader(data_loader: DataLoader, loss_func: LossFunc, model: GptModel, device: Device, n_batches: int = None) -> float:
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


def evaluate_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: Device, eval_iter: int, loss_func: LossFunc) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, loss_func, model, device, n_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, loss_func, model, device, n_batches=eval_iter)
    model.train()
    return train_loss, val_loss
