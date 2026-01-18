import argparse
import time
import torch

from src.scripts.interactive_session import InteractiveSession
from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.tokenization.tokenizer import g_tokenizer


def run_generation_flow(model_path: str, prompt: str, max_new_tokens: int = 25, temperature: float = 1.0,
                        top_k: int = 50, device_type: str = "auto", seed: int = 123, measure_time: bool = False,
                        measure_memory: bool = False) -> str:
    """
    Run a one-off text generation flow using a pre-trained GPT model.

    Args:
        model_path (str): The path to the pre-trained GPT model.
        prompt (str): The prompt text for generation.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 25.
        temperature (float): The sampling temperature for generation. Use 0 for greedy decoding. Default is 1.0.
        top_k (int): The Top-K sampling parameter. Use 0 to disable Top-K sampling. Default is 50.
        device_type (str): The device to run the model on (e.g., 'cpu', 'cuda', or 'auto'). Default is 'auto'.
        seed (int): The random seed for reproducibility. Default is 123.
    """
    g_logger.info("Running model generation flow...")

    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

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
        g_logger.info(f"Measuring {' and '.join(requested_measurements)} for model loading and generation")

    if measure_time or (measure_memory and torch.cuda.is_available()):
        if measure_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gpt = load_model(model_path, device)[0]
        gpt.eval()
        if measure_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_duration = time.time() - load_start_time
        if measure_memory and torch.cuda.is_available():
            load_max_mem_bytes = torch.cuda.max_memory_allocated()
            load_max_mem_gb = load_max_mem_bytes / (1024 ** 3)
    else:
        gpt = load_model(model_path, device)[0]
        gpt.eval()

    if measure_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start_time = time.time()
    if measure_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Reset random seed immediately before generation for reproducibility
    torch.manual_seed(seed)

    g_logger.info("Generating text...")
    token_ids = gpt.generate(
        idx=g_tokenizer.text_to_token_ids(prompt).to(device),
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

    generated_text = g_tokenizer.token_ids_to_text(token_ids)

    g_logger.info(f"Output text:\n{generated_text.strip()}")
    if measure_time and load_duration is not None:
        g_logger.info(f"[Model Load] Duration:   {load_duration:.2f} sec")
    if measure_memory and load_max_mem_gb is not None:
        g_logger.info(f"[Model Load] Max memory: {load_max_mem_gb:.2f} GB")
    if measure_time and gen_duration is not None:
        g_logger.info(f"[Generation] Duration:   {gen_duration:.2f} sec")
    if measure_time and tps is not None:
        g_logger.info(f"[Generation] TPS:        {tps:.2f} tokens/sec")
    if measure_memory and gen_max_mem_gb is not None:
        g_logger.info(f"[Generation] Max memory: {gen_max_mem_gb:.2f} GB")

    return generated_text


class InteractiveGenerationSession(InteractiveSession):
    """
    An interactive session for text generation using a pre-trained GPT2 model.
    """

    @property
    def welcome_msg(self) -> str:
        """
        Returns the welcome message for the interactive session.
        """
        return (
            "Interactive Session with GPT2 Model\n"
            "Type your prompt and press Enter, or type /bye to exit\n"
            "You can also hit Ctrl+C / Command+C to abort generation"
        )

    def format_input(self, user_input: str) -> str:
        """
        Formats the user input as a prompt for generation.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            str: The formatted prompt.
        """
        return user_input


def run_interactive_generation_flow(model_path: str, max_new_tokens: int = 25, temperature: float = 1.0,
                                    top_k: int = 50, device_type: str = "auto", seed: int = 123) -> None:
    """
    Run an interactive text generation session using a pre-trained GPT model.

    Args:
        model_path (str): The path to the pre-trained GPT model.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 25.
        temperature (float): The sampling temperature for generation. Use 0 for greedy decoding. Default is 1.0.
        top_k (int): The Top-K sampling parameter. Use 0 to disable Top-K sampling. Default is 50.
        device_type (str): The device to run the model on (e.g., 'cpu', 'cuda', or 'auto'). Default is 'auto'.
        seed (int): The random seed for reproducibility. Default is 123.
    """
    g_logger.info("Running model interactive generation flow...")
    session = InteractiveGenerationSession(model_path, max_new_tokens, temperature, top_k, device_type, seed)
    session.start()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments for text generation to the given argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
    """
    parser.add_argument("--model-path", type=str, required=True, help="Path to a pre-trained GPT2 model saved in Pytorch format (as described in src/scripts/common.py).")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation. If not provided, enters interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=25, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation. Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling parameter. Use 0 to disable Top-K sampling.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--measure-time", action="store_true", help="Measure and report time taken for model loading and generation.")
    parser.add_argument("--measure-memory", action="store_true", help="Measure and report peak GPU memory usage for model loading and generation.")



def main() -> None:
    """
    Main function to run one of the text generation flows. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(
        description="Generate text using a pre-trained GPT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.prompt is not None:
        run_generation_flow(
            model_path=args.model_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature or None,
            top_k=args.top_k or None,
            device_type=args.device,
            seed=args.seed,
            measure_time=args.measure_time,
            measure_memory=args.measure_memory
        )
    else:
        if args.measure_time or args.measure_memory:
            raise ValueError("Measuring time or memory is not supported in interactive mode.")
        run_interactive_generation_flow(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature or None,
            top_k=args.top_k or None,
            device_type=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
