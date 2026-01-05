import argparse
import time
import torch

from logging import getLogger as get_logger

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.common import load_tf_weights_into_gpt
from src.utils.device import Device, get_device
from src.utils.tokenization import text_to_token_ids, token_ids_to_text


_logger = get_logger(__name__)


def _load_eval_gpt(model_size: str, models_dir: str, device: Device, seed: int = 123) -> GptModel:
    torch.manual_seed(seed)
    gpt = load_tf_weights_into_gpt(model_size, models_dir, drop_rate=0.0, qkv_bias=True)  # No dropout during generation
    gpt.to(device)
    gpt.eval()
    return gpt


def run_model_generation_flow(prompt: str, models_dir: str, model_size: str, max_new_tokens: int = 25,
                              temperature: float = 1.0, top_k: int = 50, device_type: str = "auto", seed: int = 123,
                              measure_time: bool = False, measure_memory: bool = False) -> str:

    _logger.info("Running model generation flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}")

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
        gpt = _load_eval_gpt(model_size, models_dir, device, seed)
        if measure_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_duration = time.time() - load_start_time
        if measure_memory and torch.cuda.is_available():
            load_max_mem_bytes = torch.cuda.max_memory_allocated()
            load_max_mem_gb = load_max_mem_bytes / (1024 ** 3)
    else:
        gpt = _load_eval_gpt(model_size, models_dir, device, seed)

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
        context_size=gpt.config.context_length,
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


def run_model_interactive_flow(models_dir: str, model_size: str, max_new_tokens: int = 25, temperature: float = 1.0,
                               top_k: int = 50, device_type: str = "auto", seed: int = 123) -> None:

    # General setup
    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}.")

    # Load model
    gpt = _load_eval_gpt(model_size, models_dir, device, seed)

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
                context_size=gpt.config.context_length,
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
    add_arguments(parser)
    args = parser.parse_args()

    config = GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.use_qkv_bias,
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
            device_type=args.device,
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
            device_type=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
