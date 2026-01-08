import argparse
import torch

from logging import getLogger as get_logger

from src.model.gpt import GptModel
from src.scripts.finetune.classification import load_classifier, classify_review
from src.utils.device import Device, get_device


_logger = get_logger(__name__)


def _classification_setup(model_path: str, device_type: str, seed: int) -> tuple[GptModel, Device]:
    _logger.info("Running review classification flow...")
    torch.manual_seed(seed)
    device = get_device(device_type)
    _logger.info(f"Using device '{device.type}' and random seed {seed}.")
    _logger.info(f"Loading review classification model from '{model_path}'")
    model = load_classifier(model_path, device, n_classes=2)
    return model, device


def run_classification_flow(model_path: str, text: str, device_type: str = "auto", seed: int = 123) -> str:
    model, device = _classification_setup(model_path, device_type, seed)
    _logger.info("Classifying text...")
    result = classify_review(text, model, device, model.config.context_length)
    _logger.info(f"Input text: {text}")
    _logger.info(f"Classification: {result}")
    return result


def run_classification_interactive_flow(model_path: str, device_type: str = "auto", seed: int = 123) -> None:
    model, device = _classification_setup(model_path, device_type, seed)
    _logger.info("Entering interactive mode. Type your text and press Enter. Type /bye to exit.")
    try:
        while True:
            try:
                text = input(">>> ")
                if text == "/bye":
                    _logger.info("Exiting interactive mode.")
                    break
            except EOFError:
                _logger.info("Exiting interactive mode.")
                break
            if not text.strip():
                continue

            label, confidence = classify_review(text, model, device, model.config.context_length)
            print(f"{label} (confidence: {confidence * 100:.2f}%)")
    except Exception as e:
        _logger.error(f"An error occurred: {e}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True, help="Path to a fine-tuned classification model saved in PyTorch format.")
    parser.add_argument("--text", type=str, default=None, help="Text to classify. If not provided, enters interactive mode.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps', or 'auto').")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify text using a fine-tuned classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.text is not None:
        run_classification_flow(
            model_path=args.model_path,
            text=args.text,
            device_type=args.device,
            seed=args.seed
        )
    else:
        run_classification_interactive_flow(
            model_path=args.model_path,
            device_type=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
