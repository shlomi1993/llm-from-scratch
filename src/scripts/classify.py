import argparse
import torch

from torch import Tensor

from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.interactive_session import InteractiveSession
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.tokenization.tokenizer import EOT_IDX, g_tokenizer


"""
WARNING: There are no dedicated tests for this script. Classification tuning and prediction are covered in the file
tests/test_classifier_cli.py by validating scripts/finetune/classification.py flow and classify_review function.
"""


def load_classifier(model_path: str, device: Device, n_classes: int) -> GptModel:
    """
    Load a fine-tuned classification model from a checkpoint and return it as a GptModel instance in eval mode.

    Args:
        model_path (str): Path to the checkpoint file.
        device (Device): Device to load the model on.
        n_classes (int): Number of output classes.

    Returns:
        GptModel: Loaded classification model.
    """

    # Load the checkpoint to extract config
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = GptConfig(**checkpoint["config"])

    # Create model with the same architecture
    model = GptModel(config)

    # Replace output head with classification head BEFORE loading weights
    model.out_head = torch.nn.Linear(in_features=config.emb_dim, out_features=n_classes)

    # Load the fine-tuned weights on the device
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


def classify_review(text: str, model: GptModel, device: Device, max_length: int, pad_token_id: int = EOT_IDX) -> tuple[str, float]:
    """
    Classify a single SMS review as "spam" or "not spam" using the fine-tuned classification model.

    Note that this function change the model to eval mode.

    Args:
        text (str): Input text to classify.
        model (GptModel): Fine-tuned classification model.
        device (Device): Device to perform computation on.
        max_length (int): Maximum input length for the model.
        pad_token_id (int, optional): Token ID used for padding. Defaults to PAD_IDX.

    Returns:
        tuple[str, float]: Predicted label ("spam" or "not spam") and confidence score as a float.
    """
    model.eval()

    # Verify that the input length does not exceed model context length
    supported_context = model.pos_emb.weight.shape[0]
    if max_length > supported_context:
        raise ValueError(f"max_length ({max_length}) exceeds model context ({supported_context}).")

    # Tokenize and truncate
    input_ids = g_tokenizer.encode(text)[:max_length]

    # Pad
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits: Tensor = model(input_tensor)[:, -1]

    # Get predicted label and confidence
    probabilities = torch.softmax(logits, dim=-1)
    label_id = torch.argmax(probabilities, dim=-1).item()
    label = "spam" if label_id == 1 else "not spam"
    confidence = probabilities[0, label_id].item()

    # Decode label
    return label, confidence


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
    result = classify_review(text, model, device, max_length=120)
    g_logger.info(f"Input text: \"{text}\"")
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
                if text.lower() in InteractiveSession.EXIT_COMMANDS:
                    g_logger.info("Exiting interactive mode.")
                    break
            except EOFError:
                g_logger.info("Exiting interactive mode.")
                break

            label, confidence = classify_review(text, model, device, max_length=120)
            print(f"{label} (confidence: {confidence * 100:.2f}%)")
    except Exception as e:
        g_logger.error(f"An error occurred: {e}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True, help="Path to a fine-tuned spam/ham classification model saved in PyTorch format.")
    parser.add_argument("--text", type=str, default=None, help="Text to classify. If not provided, enters interactive mode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="cpu", help="Device to use for inference.")
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
