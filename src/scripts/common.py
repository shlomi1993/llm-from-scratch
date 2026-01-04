import json
import numpy as np
import os
import tensorflow as tf
import time
import torch

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.model.transformer import TransformerBlock
from src.utils.device import Device


########################################################################################################################
################################################# GPT-2 Weight Loading #################################################
########################################################################################################################


def _load_gpt2_params_from_tf_ckpt(ckpt_path : dict[str, str], settings: dict[str, int]) -> dict:

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        name: str

        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def _load_gpt2_params(model_size: str, models_dir: str) -> dict:
    model_dir = os.path.join(models_dir, model_size)
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' does not exist. Please download the model first.")
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return params


def _assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_tf_weights_into_gpt(model_size: str, models_dir: str, config: GptConfig) -> GptModel:
    params = _load_gpt2_params(model_size, models_dir)

    gpt = GptModel(config)
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte"])

    # Load transformer block weights
    for i in range(len(params["blocks"])):
        block: TransformerBlock = gpt.trf_blocks[i]

        # Attention weights
        q_w, k_w, v_w = np.split((params["blocks"][i]["attn"]["c_attn"])["w"], 3, axis=-1)
        block.att.W_query.weight = _assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = _assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = _assign(block.att.W_value.weight, v_w.T)

        # Attention biases
        q_b, k_b, v_b = np.split((params["blocks"][i]["attn"]["c_attn"])["b"], 3, axis=-1)
        block.att.W_query.bias = _assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = _assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = _assign(block.att.W_value.bias, v_b)

        # Output projection
        block.att.out_proj.weight = _assign(block.att.out_proj.weight, params["blocks"][i]["attn"]["c_proj"]["w"].T)
        block.att.out_proj.bias = _assign(block.att.out_proj.bias, params["blocks"][i]["attn"]["c_proj"]["b"])

        # Feed-forward weights and biases
        block.ff.layers[0].weight = _assign(block.ff.layers[0].weight, params["blocks"][i]["mlp"]["c_fc"]["w"].T)
        block.ff.layers[0].bias = _assign(block.ff.layers[0].bias, params["blocks"][i]["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight = _assign(block.ff.layers[2].weight, params["blocks"][i]["mlp"]["c_proj"]["w"].T)
        block.ff.layers[2].bias = _assign(block.ff.layers[2].bias, params["blocks"][i]["mlp"]["c_proj"]["b"])

        # Layer norm parameters
        block.norm1.scale = _assign(block.norm1.scale, params["blocks"][i]["ln_1"]["g"])
        block.norm1.shift = _assign(block.norm1.shift, params["blocks"][i]["ln_1"]["b"])
        block.norm2.scale = _assign(block.norm2.scale, params["blocks"][i]["ln_2"]["g"])
        block.norm2.shift = _assign(block.norm2.shift, params["blocks"][i]["ln_2"]["b"])

    # Final layer norm and output head
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])

    return gpt


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


def evaluate_model(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, device: Device, n_batches: int, criterion: LossFunc) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, criterion, model, device, n_batches)
        val_loss = calc_loss_loader(val_loader, criterion, model, device, n_batches)
    model.train()
    return train_loss, val_loss
