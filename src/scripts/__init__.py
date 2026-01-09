"""
GPT scripts package

This package contains scripts for downloading pre-trained GPT-2 model files, pre-train GPT-2 model from scratch,
generating text using a pre-trained foundation GPT-2 model, and fine-tuning a pre-trained GPT-2 model on datasets for
spam detection and instruction following.
"""

from .common import calc_loss_batch, calc_loss_last_token, calc_loss_loader, evaluate_losses, save_model, load_model
from .download import download_gpt2, convert_tf_weights_into_pytorch_model
from .pretrain import (
    TrainingResults,
    train_test_split,
    generate_and_print_sample,
    train_foundation_model,
    run_model_training_flow
)
from .generate import (
    run_model_generation_flow,
    run_model_interactive_flow
)
from .finetune.classification import (
    FineTuningResults,
    create_balanced_dataset,
    random_split,
    create_dataloaders,
    calc_accuracy_loader,
    load_classifier,
    classify_review,
    finetune_classifier,
    run_classification_finetuning_flow
)
from .finetune.instruction import (
    custom_collate_fn,
    create_dataloaders,
    load_assistant,
    extract_response,
    generate_response,
    run_instruction_finetuning_flow,
)


__all__ = [
    "calc_loss_batch",
    "calc_loss_last_token",
    "calc_loss_loader",
    "evaluate_losses",
    "save_model",
    "load_model",
    "download_gpt2",
    "convert_tf_weights_into_pytorch_model",
    "TrainingResults",
    "train_test_split",
    "generate_and_print_sample",
    "train_foundation_model",
    "run_model_training_flow",
    "run_model_generation_flow",
    "run_model_interactive_flow",
    "FineTuningResults",
    "create_balanced_dataset",
    "random_split",
    "create_dataloaders",
    "calc_accuracy_loader",
    "load_classifier",
    "classify_review",
    "finetune_classifier",
    "run_classification_finetuning_flow",
    "custom_collate_fn",
    "create_dataloaders",
    "load_assistant",
    "extract_response",
    "generate_response",
    "run_instruction_finetuning_flow",
]
