import argparse
import json
import numpy as np
import os
import tensorflow as tf
import time
import torch

from logging import getLogger as get_logger

from src.model.config import GptConfig, add_arguments as add_gpt_config_arguments
from src.model.gpt import GptModel
from src.model.transformer import TransformerBlock
from src.utils.device import Device, get_device
from src.utils.tokenization import text_to_token_ids, token_ids_to_text


_logger = get_logger(__name__)

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


def load_weights_into_gpt(model_size: str, models_dir: str, config: GptConfig) -> GptModel:
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


def run_model_generation_flow(config: GptConfig, prompt: str, models_dir: str, model_size: str,
                              max_new_tokens: int = 25, temperature: float = 1.0, top_k: int = 50, device: str = "auto",
                              seed: int = 123, measure_time: bool = False, measure_memory: bool = False) -> str:

    _logger.info("Running model generation flow...")

    _logger.info(f"Using device '{device}' and random seed {seed}.")
    torch.manual_seed(seed)
    device = get_device(device)

    requested_measurements = []
    load_duration = load_max_mem_gb = gen_duration = gen_max_mem_gb = tps = None
    if measure_time:
        requested_measurements.append("time")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_start_time = time.time()
    if measure_memory and torch.cuda.is_available():
        requested_measurements.append("GPU memory")
        torch.cuda.reset_peak_memory_stats()
    if requested_measurements:
        _logger.info(f"Measuring {' and '.join(requested_measurements)} for model loading and generation")

    _logger.info(f"Loading GPT model of size '{model_size}' from '{models_dir}'")
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

    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start_time = time.time()
    if measure_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _logger.info("Generating text...")
    token_ids = gpt.generate(
        idx=text_to_token_ids(prompt).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=temperature,
        top_k=top_k
    )

    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_duration = time.time() - gen_start_time
        tps = token_ids.shape[1] / gen_duration
    if measure_memory and torch.cuda.is_available():
        gen_max_mem_bytes = torch.cuda.max_memory_allocated()
        gen_max_mem_gb = gen_max_mem_bytes / (1024 ** 3)

    generated_text = token_ids_to_text(token_ids)

    out_section_divider = "-" * 20
    _logger.info(f"Output text:\n{out_section_divider}\n{generated_text}\n{out_section_divider}\n")
    if measure_time and load_duration is not None:
        _logger.info(f"[Model Load] Duration:   {load_duration:.2f} sec")
    if measure_memory and load_max_mem_gb is not None:
        _logger.info(f"[Model Load] Max memory: {load_max_mem_gb:.2f} GB")
    if measure_time and gen_duration is not None:
        _logger.info(f"[Generation] Duration:   {gen_duration:.2f} sec")
    if measure_time and tps is not None:
        _logger.info(f"[Generation] TPS:        {tps:.2f} tokens/sec")
    if measure_memory and gen_max_mem_gb is not None:
        _logger.info(f"[Generation] Max memory: {gen_max_mem_gb:.2f} GB")

    return generated_text


def run_model_interactive_flow(config: GptConfig, models_dir: str, model_size: str, max_new_tokens: int = 25,
                               temperature: float = 1.0, top_k: int = 50, device: str = "auto", seed: int = 123) -> None:

    # General setup
    torch.manual_seed(seed)
    device = get_device(device)

    # Load model
    gpt = _load_eval_gpt(config, model_size, models_dir, device, seed)

    # Run interactive mode
    _logger.info("Entering interactive mode. Type your prompt and press Enter. Press Ctrl+C/CMD+C to exit.")
    try:
        while True:
            try:
                prompt = input("Prompt: ")
            except EOFError:
                _logger.info("Exiting interactive mode.")
                break
            if not prompt.strip():
                continue
            token_ids = gpt.generate(
                idx=text_to_token_ids(prompt).to(device),
                max_new_tokens=max_new_tokens,
                context_size=config.context_length,
                temperature=temperature,
                top_k=top_k
            )
            generated_text = token_ids_to_text(token_ids)
            _logger.info("Output:\n" + generated_text)
    except KeyboardInterrupt:
        _logger.info("Exiting interactive mode.")
    except Exception as e:
        _logger.error(f"An error occurred: {e}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation. If not provided, enters interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=25, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling parameter.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on (e.g., 'cpu', 'cuda', or 'auto').")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory where the GPT-2 models are stored.")
    parser.add_argument("--model-size", type=str, default="124M", choices=["124M", "355M", "774M", "1558M"], help="Size of the GPT-2 model to use.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--measure-time", action="store_true", help="Measure and report time taken for model loading and generation.")
    parser.add_argument("--measure-memory", action="store_true", help="Measure and report peak GPU memory usage for model loading and generation.")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text using a pre-trained GPT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(parser)
    add_arguments(parser)
    args = parser.parse_args()

    config = GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.qkv_bias,
        kv_window_size=args.kv_window_size
    )

    if args.prompt is not None:
        run_model_generation_flow(
            config=config,
            prompt=args.prompt,
            models_dir=args.models_dir,
            model_size=args.model_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            seed=args.seed,
            measure_time=args.measure_time,
            measure_memory=args.measure_memory
        )
    else:
        run_model_interactive_flow(
            config=config,
            models_dir=args.models_dir,
            model_size=args.model_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
