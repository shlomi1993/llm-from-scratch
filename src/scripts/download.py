import argparse
import json
import numpy as np
import os
import requests
import tensorflow as tf
import torch

from tqdm import tqdm

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.model.transformer import TransformerBlock
from src.utils.checkpoint import save_model
from src.utils.logger import g_logger


DOWNLOAD_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_DOWNLOAD_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
ALLOWED_MODEL_SIZES = ["124M", "355M", "774M", "1558M"]
FILES_TO_DOWNLOAD = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe"
]


########################################################################################################################
########################################### Download TensorFlow GPT-2 Files ############################################
########################################################################################################################


def _download_file(url: str, destination: str) -> None:

    # Send a GET request to download the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("Content-Length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size and file_size == file_size_local:
            g_logger.info(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 KB

    # Initialize the progress bar with total file size
    desc = os.path.basename(url)
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    g_logger.info(f"Successfully downloaded {destination}")


def download_gpt2(model_size: str, models_dir: str) -> str:
    if model_size not in ALLOWED_MODEL_SIZES:
        raise ValueError(f"Model size {model_size} not in {ALLOWED_MODEL_SIZES}")

    # Make model directory
    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    # Download files
    for filename in FILES_TO_DOWNLOAD:
        file_url = os.path.join(DOWNLOAD_URL, model_size, filename)
        backup_url = os.path.join(BACKUP_DOWNLOAD_URL, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        try:
            _download_file(file_url, file_path)
        except requests.exceptions.RequestException:
            g_logger.warning(f"Primary URL ({file_url}) failed. Attempting backup URL: {backup_url}")
            try:
                _download_file(backup_url, file_path)
            except requests.exceptions.RequestException as e:
                g_logger.error(f"Failed to download {filename} from both primary and backup URLs.")
                raise e

    # Return model size directory
    g_logger.info(f"GPT-2 model files downloaded to: {model_dir}")
    return model_dir


########################################################################################################################
############################################# Load TensorFlow GPT-2  Files #############################################
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


def _load_gpt2_params(model_size: str, models_dir: str) -> tuple[dict, dict]:
    model_dir = os.path.join(models_dir, model_size)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist. Please download the model first.")
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return params, settings


def _assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def convert_tf_weights_into_pytorch_model(model_size: str, models_dir: str, file_name: str = "model.pth") -> GptModel:
    target_path = os.path.join(models_dir, model_size, file_name)
    g_logger.info(f"Converting TensorFlow model to PyTorch format in {target_path}")

    params, settings = _load_gpt2_params(model_size, models_dir)

    # Create GPT config
    config = GptConfig(
        vocab_size=settings["n_vocab"],
        context_length=settings["n_ctx"],
        emb_dim=settings["n_embd"],
        n_heads=settings["n_head"],
        n_layers=settings["n_layer"],
        drop_rate=0.0,  # GPT-2 does not use dropout during inference
        qkv_bias=True,  # GPT-2 uses biases in QKV projections
    )

    # Initialize GPT model

    # Load embedding weights
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

    # Load final layer norm and output head weights
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])

    # Save model in PyTorch format
    save_model(gpt, target_path)

    return gpt


########################################################################################################################
###################################################### Flow + CLI ######################################################
########################################################################################################################


def run_download_flow(model_sizes: list[str], models_dir: str, convert: bool = False) -> None:
    for model_size in model_sizes:
        download_gpt2(model_size, models_dir)
        if convert:
            convert_tf_weights_into_pytorch_model(model_size, models_dir)

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sizes", type=str, nargs="+", default=["124M"], choices=ALLOWED_MODEL_SIZES, help="Size(s) of the GPT-2 model(s) to download. You can specify multiple sizes.")
    parser.add_argument("--dir", type=str, default="models", help="Directory to save the downloaded model files.")
    parser.add_argument("--convert", action="store_true", help="Whether to convert and save the model in PyTorch format.")
    parser.add_argument("--converted-filename", type=str, default="model.pth", help="Filename for the converted PyTorch model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GPT-2 model files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_download_flow(args.sizes, args.dir, args.convert)


if __name__ == "__main__":
    main()
