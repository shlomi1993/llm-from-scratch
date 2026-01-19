import os
import re
import sys
import torch
import tiktoken

from pathlib import Path

from src.scripts.classify import classify_review, load_classifier
from tests.chapters_code import GPTModel
from tests.common import run_subprocess, extract_losses, extract_accuracies, compare_losses, compare_accuracies


PREDICTION_TEST_SAMPLES = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    "URGENT! Your account has been compromised. Click here to verify your identity now.",
    "Thanks for the meeting notes. I'll review them and get back to you tomorrow."
]


def classify_review_chapter(text: str, model: GPTModel, tokenizer: tiktoken.Encoding, device: torch.device,
                            max_length: int = 120, pad_token_id: int = 50256) -> tuple[str, float]:
    model.eval()

    # Prepare inputs
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate if too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad to max_length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    # Get prediction and confidence
    probabilities = torch.softmax(logits, dim=-1)
    predicted_label_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = "spam" if predicted_label_id == 1 else "not spam"
    confidence = probabilities[0, predicted_label_id].item()

    return predicted_label, confidence


def load_chapter_model(model_path: str, device: torch.device) -> GPTModel:
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Model configuration
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }

    # Initialize model
    model = GPTModel(config)

    # Convert to classifier
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=config["emb_dim"], out_features=num_classes)

    # Load weights from checkpoint (strict=False because mask buffers aren't saved)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model


def compare_model_predictions(test_samples: list[str], model_path: str) -> None:
    print("Comparing inference predictions")
    device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    print("Loading model using chapter code...")
    chapter_model = load_chapter_model(model_path, device)

    print("Loading model using application code...")
    cli_model = load_classifier(model_path, device, n_classes=2)
    cli_model.eval()

    print("\nComparing predictions on test samples:\n")

    all_match = True
    for i, text in enumerate(test_samples, 1):
        # CLI prediction
        cli_label, cli_conf = classify_review(text, cli_model, device, max_length=120)

        # Chapter prediction
        chapter_label, chapter_conf = classify_review_chapter(text, chapter_model, tokenizer, device, max_length=120)

        # Compare
        labels_match = cli_label == chapter_label
        conf_close = abs(cli_conf - chapter_conf) < 1e-5
        print(f"Sample {i}: {text[:60]}...")
        print(f"  CLI:     {cli_label:10s} (confidence: {cli_conf:.4f})")
        print(f"  Chapter: {chapter_label:10s} (confidence: {chapter_conf:.4f})")

        if labels_match and conf_close:
            print("\033[92m  ✓ Match\033[0m")
        else:
            print("\033[91m  ✗ Mismatch!\033[0m")
            all_match = False
        print()

    assert all_match, "Model predictions don't match between CLI and chapter loading implementations"
    print("\033[92m✓ All inference predictions match!\033[0m")


def test_finetune_classifier_cli_vs_script(tmp_path: Path, chapters_path: Path):
    cli_model_path = tmp_path / "test_classifier_model.pth"

    cli_cmd = [
        "gpt2", "finetune", "classification",
        "--pretrained-model-path", "models/124M/model.pth",
        "--tuning-set-path", "dataset/sms_spam_collection/SMSSpamCollection.tsv",
        "--column-names", "Label", "Text",
        "--train-frac", "0.7",
        "--validation-frac", "0.1",
        "--save-split-dir", str(tmp_path),
        "--batch-size", "8",
        "--seed", "123",
        "--device", "cpu",
        "--lr", "5e-5",
        "--n-epochs", "5",
        "--weight-decay", "0.1",
        "--eval-freq", "50",
        "--eval-iter", "5",
        "--model-save-path", str(cli_model_path)
    ]
    cli_output = run_subprocess(cli_cmd)
    cli_losses = extract_losses(cli_output, ref=False)
    cli_accuracies = extract_accuracies(cli_output, ref=False)

    # Clean up split files - NOTE: uncomment if running the chapter script below
    # for f in tmp_path / "train.csv", tmp_path / "validation.csv", tmp_path / "test.csv":
    #     if os.path.exists(f):
    #         os.remove(f)

    # chapter_cmd = [sys.executable, "-u", str(chapters_path / "ch06/01_main-chapter-code/gpt_class_finetune.py")]
    # chapter_output = run_subprocess(chapter_cmd, cwd=tmp_path)
    with open("tests/ref/classifier.txt", "r") as f:  # Load pre-saved reference output instead of running the script
        chapter_output = f.read()
    chapter_losses = extract_losses(chapter_output, ref=True)
    chapter_accuracies = extract_accuracies(chapter_output, ref=True)

    compare_losses(actual_losses=cli_losses, expected_losses=chapter_losses, tolerance=1e-2)
    compare_accuracies(actual_metrics=cli_accuracies, expected_metrics=chapter_accuracies, tolerance=1.0)
    compare_model_predictions(PREDICTION_TEST_SAMPLES, cli_model_path)
