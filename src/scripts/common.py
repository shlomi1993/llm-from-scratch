import os
import torch

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.utils.device import Device


########################################################################################################################
################################################### Model Evaluation ###################################################
########################################################################################################################


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


def calc_loss_loader(data_loader: DataLoader, criterion: LossFunc, model: GptModel, device: Device, n_batches: int = None) -> float:
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


def evaluate_losses(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: Device, n_batches: int,
                    criterion: LossFunc) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, criterion, model, device, n_batches)
        val_loss = calc_loss_loader(val_loader, criterion, model, device, n_batches)
    model.train()
    return train_loss, val_loss


########################################################################################################################
################################################# Model Save and Load ##################################################
########################################################################################################################

"""
Project saved model format:
{
    "model_state_dict": <state dict of the model>,
    "config": <model config as a dict>,
    "optimizer_state_dict": <state dict of the optimizer> (optional)
}
"""


def save_model(model: GptModel, save_path: str, optimizer: Optimizer = None) -> None:
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config.__dict__
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, save_path)


def load_model(saved_model_path: str, device: Device) -> tuple[GptModel, Optimizer | None]:
    checkpoint: dict = torch.load(saved_model_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    gpt_config = GptConfig(**config)
    model = GptModel(gpt_config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    if optimizer_state_dict is None:
        return model, None

    learning_rate = optimizer_state_dict['param_groups'][0]['lr']
    weight_decay = optimizer_state_dict['param_groups'][0]['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(optimizer_state_dict)
    return model, optimizer
