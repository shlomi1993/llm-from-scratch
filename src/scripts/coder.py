import argparse
import torch

from shutil import get_terminal_size

from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.tokenization.tokenizer import PAD_IDX, text_to_token_ids


# Matches the format_prompt logic in AlpacaCodeDataset (no system preamble)
CODER_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def run_coder_flow(model_path: str, max_new_tokens: int = 256, device_type: str = "auto", seed: int = 123) -> None:
    g_logger.info("Starting interactive coding session...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    coder = load_model(model_path, device)[0]
    coder.eval()

    sep = "=" * get_terminal_size().columns
    welcome_msg = "\n".join([
        sep,
        "Interactive Session with CoderGPT2",
        "Describe your coding task or type /bye to end the session",
        "You can also hit Ctrl+C / Command+C to abort generation",
        sep
    ])
    print(welcome_msg)

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/bye":
                print("Goodbye!")
                break

            # Format prompt into tokens
            prompt = CODER_PROMPT_TEMPLATE.format(instruction=user_input)
            idx = text_to_token_ids(prompt).to(device)

            try:
                coder.generate(idx, max_new_tokens, coder.config.context_length, eos_id=PAD_IDX, live=True)
                print() # Newline after generation ends
            except KeyboardInterrupt:
                print("\nGeneration interrupted by user")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            g_logger.error(f"An error occurred: {e}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, default="coder.pth", help="Path to the finetuned coder model")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum code tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, mps, auto)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive coding session with Fine-Tuned GPT2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_coder_flow(args.model_path, args.max_new_tokens, args.device, args.seed)


if __name__ == "__main__":
    main()
