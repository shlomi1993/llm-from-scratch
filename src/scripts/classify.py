import argparse
import torch


from src.model.gpt import GptModel
from src.scripts.finetune.classification import load_classifier, classify_review
from src.utils.device import Device, get_device
from src.utils.logger import g_logger

"""
WARNING: There are no dedicated tests for this script. Classification tuning and prediction are covered in the file
tests/test_classifier_cli.py by validating scripts/finetune/classification.py flow and classify_review function.
"""


def _classification_setup(model_path: str, device_type: str, seed: int) -> tuple[GptModel, Device]:
    g_logger.info("Running spam/ham classification flow...")
    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}.")
    model = load_classifier(model_path, device, n_classes=2)
    model.eval()
    g_logger.info(f"Loaded classification model from {model_path}")
    return model, device


def run_spam_ham_flow(model_path: str, text: str, device_type: str = "auto", seed: int = 123) -> str:
    model, device = _classification_setup(model_path, device_type, seed)
    g_logger.info("Classifying text...")
    result = classify_review(text, model, device, model.config.context_length)
    g_logger.info(f"Input text: {text}")
    g_logger.info(f"Prediction: {result}")
    return result


def run_spam_ham_interactive_flow(model_path: str, device_type: str = "auto", seed: int = 123) -> None:
    model, device = _classification_setup(model_path, device_type, seed)
    g_logger.info("Entering interactive mode. Type your text and press Enter. Type /bye to exit.")
    try:
        while True:
            try:
                text = input(">>> ").strip()
                if not text:
                    continue
                if text == "/bye":
                    g_logger.info("Exiting interactive mode.")
                    break
            except EOFError:
                g_logger.info("Exiting interactive mode.")
                break

            label, confidence = classify_review(text, model, device, model.config.context_length)
            print(f"{label} (confidence: {confidence * 100:.2f}%)")
    except Exception as e:
        g_logger.error(f"An error occurred: {e}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True, help="Path to a fine-tuned spam/ham classification model saved in PyTorch format.")
    parser.add_argument("--text", type=str, default=None, help="Text to classify. If not provided, enters interactive mode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use for inference.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify text as spam or ham using a fine-tuned classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.text is not None:
        run_spam_ham_flow(
            model_path=args.model_path,
            text=args.text,
            device_type=args.device,
            seed=args.seed
        )
    else:
        run_spam_ham_interactive_flow(
            model_path=args.model_path,
            device_type=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
