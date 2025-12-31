import json
import numpy as np
import os
import tensorflow as tf
import time
import torch

from src.common import Device, get_device, text_to_token_ids, token_ids_to_text
from src.config import GptConfig
from src.gpt import GptModel
from src.gpt_utils.download import download_gpt2
from src.transformer import TransformerBlock


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


def _load_gpt2_params(model_size: str, models_dir: str, download: bool = False) -> dict:
    if download:
        model_dir = download_gpt2(model_size, models_dir)
    else:
        model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return params


def _assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(model_size: str, models_dir: str, config: GptConfig, download: bool = False) -> GptModel:
    params = _load_gpt2_params(model_size, models_dir, download)

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


def _load_eval_gpt(config: GptConfig, model_size: str, models_dir: str, device: Device, seed: int = 123) -> GptModel:

    # General setup
    torch.manual_seed(seed)

    # Set generation configs
    gen_config = GptConfig(
        context_length = config.context_length,
        vocab_size = config.vocab_size,
        emb_dim = config.emb_dim,
        n_layers = config.n_layers,
        n_heads = config.n_heads,
        drop_rate = 0.0,  # No dropout during generation
        qkv_bias = True  # TODO needed only for downloaded models or also trained locally?
    )

    # Load model
    gpt = GptModel(gen_config)
    load_weights_into_gpt(model_size, models_dir, gen_config)
    gpt.to(device)
    gpt.eval()

    return gpt


def run_model_generation_flow(config: GptConfig, prompt: str, models_dir: str, model_size: str, tokenizer: str,
                              max_new_tokens: int = 25, temperature: float = 1.0, top_k: int = 50, device: str = "cpu",
                              seed: int = 123, measure_time: bool = False, measure_memory: bool = False) -> str:

    # General setup
    torch.manual_seed(seed)
    device = get_device(device)

    # Time and memory measurement setup for model loading
    load_duration = load_max_mem_gb = gen_duration = gen_max_mem_gb = tps = None
    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_start_time = time.time()
    if measure_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load model
    gpt = None
    if measure_time or (measure_memory and torch.cuda.is_available()):
        if measure_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gpt = _load_eval_gpt(config, model_size, models_dir, device, seed)
        if measure_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_duration = time.time() - load_start_time
        if measure_memory and torch.cuda.is_available():
            load_max_mem_bytes = torch.cuda.max_memory_allocated()
            load_max_mem_gb = load_max_mem_bytes / (1024 ** 3)
    else:
        gpt = _load_eval_gpt(config, model_size, models_dir, device, seed)

    # Time and memory measurement setup for generation
    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start_time = time.time()
    if measure_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Generate text
    token_ids = gpt.generate(
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=temperature,
        top_k=top_k
    )

    # Time and memory measurement results for generation
    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_duration = time.time() - gen_start_time
        tps = token_ids.shape[1] / gen_duration
    if measure_memory and torch.cuda.is_available():
        gen_max_mem_bytes = torch.cuda.max_memory_allocated()
        gen_max_mem_gb = gen_max_mem_bytes / (1024 ** 3)

    # Decode generated text
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # Print all results
    print("Output text:\n", generated_text, end="\n\n")
    if measure_time and load_duration is not None:
        print(f"[Model Load] Duration:   {load_duration:.2f} sec")
    if measure_memory and load_max_mem_gb is not None:
        print(f"[Model Load] Max memory: {load_max_mem_gb:.2f} GB")
    if measure_time and gen_duration is not None:
        print(f"[Generation] Duration:   {gen_duration:.2f} sec")
    if measure_time and tps is not None:
        print(f"[Generation] TPS:        {tps:.2f} tokens/sec")
    if measure_memory and gen_max_mem_gb is not None:
        print(f"[Generation] Max memory: {gen_max_mem_gb:.2f} GB")

    return generated_text


def run_model_interactive_flow(config: GptConfig, models_dir: str, model_size: str, tokenizer: str,
                               max_new_tokens: int = 25, temperature: float = 1.0, top_k: int = 50, device: str = "cpu",
                               seed: int = 123) -> None:

    # General setup
    torch.manual_seed(seed)
    device = get_device(device)

    # Load model
    gpt = _load_eval_gpt(config, model_size, models_dir, device, seed)

    # Run interactive mode
    print("\nInteractive mode. Type your prompt and press Enter. Press Ctrl+C/CMD+C to exit.\n")
    try:
        while True:
            try:
                prompt = input("Prompt: ")
            except EOFError:
                print("\nExiting interactive mode.")
                break
            if not prompt.strip():
                continue
            token_ids = gpt.generate(
                idx=text_to_token_ids(prompt, tokenizer).to(device),
                max_new_tokens=max_new_tokens,
                context_size=config.context_length,
                temperature=temperature,
                top_k=top_k
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            print("Output:\n", generated_text)
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
