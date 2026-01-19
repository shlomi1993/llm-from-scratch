import argparse

from src.dataset import AlpacaCodeDataset
from src.scripts.interactive_session import InteractiveSession
from src.utils.logger import g_logger


class InteractiveCoderSession(InteractiveSession):
    """
    Interactive session class for coding tasks using a fine-tuned GPT2 model.
    """

    @property
    def welcome_msg(self) -> str:
        return (
            "Interactive Session with CoderGPT2\n"
            "Describe your coding task or type /bye to end the session\n"
            "You can also hit Ctrl+C / Command+C to abort generation"
        )

    def format_input(self, user_input: str) -> str:
        """
        Format the user input into a prompt suitable for the coder model.

        Args:
            user_input (str): The user's input describing the coding task.

        Returns:
            str: The formatted prompt.
        """
        return AlpacaCodeDataset.format_input({"instruction": user_input})


def run_coder_flow(model_path: str, max_new_tokens: int = 256, temperature: float = 0.0, top_k: int = None,
                   device_type: str = "auto", seed: int = 123) -> None:
    """
    Run an interactive coding session with the fine-tuned coder model.

    Args:
        model_path (str): The path to the finetuned coder model.
        max_new_tokens (int): Maximum number of code tokens to generate.
        temperature (float): Sampling temperature for generation.
        top_k (int): Top-K sampling parameter.
        device_type (str): Device to use (cpu, cuda, mps, auto).
        seed (int): Random seed for reproducibility.
    """
    g_logger.info("Starting interactive coding session...")
    session = InteractiveCoderSession(model_path, max_new_tokens, temperature, top_k, device_type, seed)
    session.start()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments for the interactive coder session.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser instance.
    """
    parser.add_argument("--model-path", type=str, default="coder.pth", help="Path to the finetuned coder model")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum code tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation. If not set, uses model default.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K sampling parameter. If not set, uses model default.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="cpu", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")


def main() -> None:
    """
    Main function to run the interactive coder session. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(
        description="Interactive coding session with Fine-Tuned GPT2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_coder_flow(args.model_path, args.max_new_tokens, args.temperature, args.top_k, args.device, args.seed)


if __name__ == "__main__":
    main()
