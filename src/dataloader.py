import tiktoken
import torch

from typing import Tuple
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# Special tokens
EOT = "<|endoftext|>"

class GptDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={EOT})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt: str, batch_size: int, max_length: int, stride: int, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GptDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
