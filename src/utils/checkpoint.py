import os
import torch

from torch.optim import Optimizer

from src.model.config import GptConfig
from src.utils.device import Device
from src.model.gpt import GptModel
from src.utils.logger import g_logger


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
        "config": vars(model.config)
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, save_path)
    g_logger.info(f"Saved model to {save_path}")


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
        g_logger.info(f"Loaded model from {saved_model_path}")
        return model, None

    learning_rate = optimizer_state_dict['param_groups'][0]['lr']
    weight_decay = optimizer_state_dict['param_groups'][0]['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(optimizer_state_dict)

    g_logger.info(f"Loaded model and optimizer from {saved_model_path}")
    return model, optimizer
