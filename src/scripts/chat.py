import argparse
import torch

from logging import getLogger as get_logger

from src.scripts.common import load_model
from src.utils.device import get_device
from src.utils.ollama import format_input
from src.utils.tokenization.tokenizer import PAD_TOKEN_ID, text_to_token_ids


_logger = get_logger(__name__)


def run_chat_flow(model_path: str, max_new_tokens: int = 256, device_type: str = "auto", seed: int = 123) -> None:
    _logger.info("Starting interactive chat session...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}")

    _logger.info(f"Loading assistant model from {model_path}")
    assistant = load_model(model_path, device)[0]
    assistant.eval()
    _logger.info("Model loaded successfully!")

    print("\n" + "=" * 70)
    print("Interactive Chat with GPT2 Assistant")
    print("=" * 70)
    print("Type your instructions below. Type /bye to end the session.")
    print("=" * 70 + "\n")

    while True:
        try:
            # Get user input
            user_input = input(">>> ").strip()

            # Check for exit commands
            if user_input.lower() in ["/bye"]:
                print("\nGoodbye!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Format as instruction
            instruction_dict = {"instruction": user_input}
            prompt = format_input(instruction_dict)
            idx = text_to_token_ids(prompt).to(device)

            # Generate response live
            assistant.generate(idx, max_new_tokens, assistant.config.context_length, eos_id=PAD_TOKEN_ID, live=True)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            _logger.error(f"Error generating response: {e}")
            print(f"\nError: {e}\n")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True, help="Path to the instruction-finetuned assistant model")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, mps, auto)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with an instruction-finetuned assistant model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_chat_flow(args.model_path, args.max_new_tokens, args.device, args.seed)


if __name__ == "__main__":
    main()
