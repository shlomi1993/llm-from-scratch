import os

import json
import numpy as np
import requests
import tensorflow as tf

from tqdm import tqdm


DOWNLOAD_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_DOWNLOAD_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
FILES_TO_DOWNLOAD = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe"
]


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
            print(f"File already exists and is up-to-date: {destination}")
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

    print(f"Successfully downloaded {destination}")


def _load_gpt2_params_from_tf_ckpt(ckpt_path : dict[str, str], settings: dict[str, int]) -> dict:

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
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


def _validate_model_size(model_size: str) -> None:
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")


def download_and_load_gpt2(model_size: str, models_dir: str) -> tuple[dict, dict]:
    _validate_model_size(model_size)

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
            print(f"\033[93mWARNING\033[0m: Primary URL ({file_url}) failed. Attempting backup URL: {backup_url}")
            _download_file(backup_url, file_path)


    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params
