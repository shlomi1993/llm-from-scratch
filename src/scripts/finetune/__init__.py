"""
Fine-tuning scripts package

This package contains scripts for fine-tuning a pre-trained GPT-2 model on datasets for spam detection and
instruction following.
"""

from .classification import (
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
from .instruction import (
    custom_collate_fn,
    create_dataloaders,
    load_instruction_finetuned_model,
    extract_response,
    generate_response,
    run_instruction_finetuning_flow,
)

__all__ = [
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
