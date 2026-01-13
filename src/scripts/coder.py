import argparse

from src.scripts.interactive_session import InteractiveSession
from src.utils.logger import g_logger


class InteractiveCoderSession(InteractiveSession):
    CODER_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"  # Matches the format_prompt logic in AlpacaCodeDataset (no system preamble)

    @property
    def welcome_msg(self) -> str:
        return (
            "Interactive Session with CoderGPT2\n"
            "Describe your coding task or type /bye to end the session\n"
            "You can also hit Ctrl+C / Command+C to abort generation"
        )

    def format_prompt(self, user_input: str) -> str:
        return self.CODER_PROMPT_TEMPLATE.format(instruction=user_input)


def run_coder_flow(model_path: str, max_new_tokens: int = 256, device_type: str = "auto", seed: int = 123) -> None:
    g_logger.info("Starting interactive coding session...")
    session = InteractiveCoderSession(model_path, max_new_tokens, device_type=device_type, seed=seed)
    session.start()


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
