import argparse


from src.scripts.interactive_session import InteractiveSession
from src.utils.logger import g_logger


class InteractiveChatSession(InteractiveSession):
    """
    An interactive chat session with a code-instruction-finetuned model.
    """
    INPUT_PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n"
    )

    @property
    def welcome_msg(self) -> str:
        """
        Returns the welcome message for the chat session.
        """
        return (
            "Interactive Chat with GPT2 Assistant\n"
            "Type /bye to end the session\n"
            "You can also hit Ctrl+C / Command+C to abort generation"
        )

    def format_input(self, user_input: str) -> str:
        """
        Format the input prompt for the chat session based on user input.

        Args:
            user_input (str): The user's input for the chat session. Example: "Explain the theory of relativity."

        Returns:
            str: The formatted prompt for the chat session.

        Example:
            >>> format_prompt("Explain the theory of relativity.")
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.
            "### Instruction:
            You are a helpful assistant. Provide clear and concise responses.
            "### Input:
            Explain the theory of relativity.
            "### Response:
            \n\n"
        """
        return self.INPUT_PROMPT_TEMPLATE.format(
            instruction="You are a helpful assistant. Provide clear and concise responses.",
            input=user_input
        )


def run_chat_flow(model_path: str, max_new_tokens: int = 256, temperature: float = 0.0, top_k: int = None,
                  device_type: str = "auto", seed: int = 123) -> None:
    """
    Run the interactive chat flow with the specified model.

    Args:
        model_path (str): The path to the instruction-finetuned model.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 256.
        temperature (float, optional): The sampling temperature for generation. Defaults to 0.0.
        top_k (int, optional): The top-k sampling parameter. Defaults to None.
        device_type (str, optional): The device to use for inference. Defaults to "auto".
        seed (int, optional): The random seed for reproducibility. Defaults to 123.
    """
    g_logger.info("Starting interactive chat session...")
    session = InteractiveChatSession(model_path, max_new_tokens, temperature, top_k, device_type, seed)
    session.start()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments for the interactive chat session.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """
    parser.add_argument("--model-path", type=str, required=True, help="Path to the instruction-finetuned assistant model")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation. If not set, uses model default.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K sampling parameter. If not set, uses model default.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="cpu", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")


def main() -> None:
    """
    Main function to run the interactive chat flow. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(
        description="Interactive chat with an instruction-finetuned assistant model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_chat_flow(args.model_path, args.max_new_tokens, args.temperature, args.top_k, args.device, args.seed)


if __name__ == "__main__":
    main()
