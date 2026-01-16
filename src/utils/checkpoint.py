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
    """
    Save the model and its configuration to the specified path. If an optimizer is provided, it will also be saved.

    Args:
        model (GptModel): The model to save.
        save_path (str): The file path where the model will be saved.
        optimizer (Optimizer, optional): The optimizer to save. Defaults to None.
    """

    # Create directory if it doesn't exist
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Create a checkpoint dict containing model state, config, and optionally optimizer state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": vars(model.config)
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Save the checkpoint to a file
    torch.save(checkpoint, save_path)
    g_logger.info(f"Saved model to {save_path}")


def load_model(saved_model_path: str, device: Device) -> tuple[GptModel, Optimizer | None]:
    """
    Load a model and its configuration from the specified path. If an optimizer state is found, it will also be loaded.

    Args:
        saved_model_path (str): The file path from which the model will be loaded.
        device (Device): The device on which the model will be loaded.

    Returns:
        tuple[GptModel, Optimizer | None]: A tuple containing the loaded model and the loaded optimizer. If no optimizer
            state is found, None is returned for the optimizer.
    """
    # Load the checkpoint from the file
    checkpoint: dict = torch.load(saved_model_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    # Create the model and load its state
    gpt_config = GptConfig(**config)
    model = GptModel(gpt_config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # If no optimizer state is found, return only the model
    if optimizer_state_dict is None:
        g_logger.info(f"Loaded model from {saved_model_path}")
        return model, None

    # Create the optimizer and load its state
    learning_rate = optimizer_state_dict['param_groups'][0]['lr']
    weight_decay = optimizer_state_dict['param_groups'][0]['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(optimizer_state_dict)

    g_logger.info(f"Loaded model and optimizer from {saved_model_path}")
    return model, optimizer
