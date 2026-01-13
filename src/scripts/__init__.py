"""
GPT scripts package

This package contains scripts for downloading pre-trained GPT-2 model files, pre-train GPT-2 model from scratch,
generating text using a pre-trained foundation GPT-2 model, and fine-tuning a pre-trained GPT-2 model on datasets for
spam detection and instruction following.
"""

from .download import download_gpt2, convert_tf_weights_into_pytorch_model
from .train import (
    TrainingResults,
    train_test_split,
    generate_and_print_sample,
    train_model,
    run_training_flow
)
from .generate import (
    run_generation_flow,
    run_interactive_generation_flow
)
from .finetune.classification import (
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
from .finetune.instruction import (
    instruction_collate_fn,
    create_instruction_dataloaders,
    test_assistant,
    run_instruction_finetuning_flow,
)
from .finetune.code_instruction import (
    coding_collate_fn,
    create_coding_dataloaders,
    run_coding_finetuning_flow,
)


__all__ = [
    "download_gpt2",
    "convert_tf_weights_into_pytorch_model",
    "TrainingResults",
    "train_test_split",
    "generate_and_print_sample",
    "train_model",
    "run_training_flow",
    "run_generation_flow",
    "run_interactive_generation_flow",
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
