"""
Data module

This module contains data-related classes and functions, including datasets and dataloaders for training and evaluation.
"""

from .datasets.instruction import InstructionDataset
from .datasets.classification import SpamDataset
from .datasets.pretrain import GptDatasetV1
from .formatting import format_input
from .loaders import GptDataloaderV1

__all__ = [
    "GptDatasetV1",
    "SpamDataset",
    "InstructionDataset",
    "GptDataloaderV1",
    "format_input",
]
