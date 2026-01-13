import argparse
import torch

from shutil import get_terminal_size

from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.tokenization.tokenizer import PAD_IDX, text_to_token_ids


INPUT_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n"
)


def run_chat_flow(model_path: str, max_new_tokens: int = 256, device_type: str = "auto", seed: int = 123) -> None:
    g_logger.info("Starting interactive chat session...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    assistant = load_model(model_path, device)[0]
    assistant.eval()

    sep = "=" * get_terminal_size().columns
    print(f"{sep}\nInteractive Chat with GPT2 Assistant\nType /bye to end the session\n{sep}")

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/bye":
                print("Goodbye!")
                break

            prompt = INPUT_PROMPT_TEMPLATE.format(
                instruction="You are a helpful assistant. Provide clear and concise responses.",
                input=user_input
            )
            idx = text_to_token_ids(prompt).to(device)

            assistant.generate(idx, max_new_tokens, assistant.config.context_length, eos_id=PAD_IDX, live=True)
            print()  # Newline after generation ends

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            g_logger.error(f"An error occurred: {e}")


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
