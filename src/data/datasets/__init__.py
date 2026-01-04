"""
Datasets module

This module contains pretraining, classification and instruction dataset classes for data loading tasks.
"""

from .classification import SpamDataset
from .instruction import InstructionDataset
from .pretrain import GptDatasetV1

__all__ = [
    "GptDatasetV1",
    "SpamDataset",
    "InstructionDataset",
]
