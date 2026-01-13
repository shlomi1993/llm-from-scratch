"""
Fine-tuning scripts package

This package contains scripts for fine-tuning a pre-trained GPT-2 model on datasets for spam detection and
instruction following.
"""

from .classification import (
    ClassificationFineTuningResults,
    create_balanced_dataset,
    random_split,
    create_classification_dataloaders,
    calc_accuracy_loader,
    load_classifier,
    classify_review,
    finetune_classifier,
    run_classification_finetuning_flow
)
from .instruction import (
    instruction_collate_fn,
    create_instruction_dataloaders,
    test_assistant,
    run_instruction_finetuning_flow,
)
from .code_instruction import (
    coding_collate_fn,
    create_coding_dataloaders,
    run_coding_finetuning_flow,
)


__all__ = [
    "ClassificationFineTuningResults",
    "create_balanced_dataset",
    "random_split",
    "create_classification_dataloaders",
    "calc_accuracy_loader",
    "load_classifier",
    "classify_review",
    "finetune_classifier",
    "run_classification_finetuning_flow",
    "instruction_collate_fn",
    "create_instruction_dataloaders",
    "test_assistant",
    "run_instruction_finetuning_flow",
    "coding_collate_fn",
    "create_coding_dataloaders",
    "run_coding_finetuning_flow",
]
