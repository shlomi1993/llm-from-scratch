"""
GPT scripts package

This package contains scripts for downloading pre-trained GPT-2 model files, pre-train GPT-2 model from scratch,
generating text using a pre-trained foundation GPT-2 model, and fine-tuning a pre-trained GPT-2 model on datasets for
spam detection and instruction following.
"""

from .common import load_tf_weights_into_gpt, calc_loss_batch, calc_loss_last_token, calc_loss_loader, evaluate_model
from .download import download_gpt2
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
    load_classification_finetuned_model,
    classify_review,
    finetune_classifier,
    run_classification_finetuning_flow
)
from .finetune.instruction import (
    custom_collate_fn,
    create_dataloaders,
    load_instruction_finetuned_model,
    extract_response,
    generate_response,
    run_instruction_finetuning_flow,
)


__all__ = [
    "load_tf_weights_into_gpt",
    "calc_loss_batch",
    "calc_loss_last_token",
    "calc_loss_loader",
    "evaluate_model",
    "download_gpt2",
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
    "load_classification_finetuned_model",
    "classify_review",
    "finetune_classifier",
    "run_classification_finetuning_flow",
    "custom_collate_fn",
    "create_dataloaders",
    "load_instruction_finetuned_model",
    "extract_response",
    "generate_response",
    "run_instruction_finetuning_flow",
]
