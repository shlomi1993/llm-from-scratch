import argparse


from src.scripts.interactive_session import InteractiveSession
from src.utils.logger import g_logger


class InteractiveChatSession(InteractiveSession):
    INPUT_PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n"
    )

    @property
    def welcome_msg(self) -> str:
        return (
            "Interactive Chat with GPT2 Assistant\n"
            "Type /bye to end the session\n"
            "You can also hit Ctrl+C / Command+C to abort generation"
        )

    def format_prompt(self, user_input: str) -> str:
        return self.INPUT_PROMPT_TEMPLATE.format(
            instruction="You are a helpful assistant. Provide clear and concise responses.",
            input=user_input
        )


def run_chat_flow(model_path: str, max_new_tokens: int = 256, device_type: str = "auto", seed: int = 123) -> None:
    g_logger.info("Starting interactive chat session...")
    session = InteractiveChatSession(model_path, max_new_tokens, device_type=device_type, seed=seed)
    session.start()


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
