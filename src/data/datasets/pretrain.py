from torch import Tensor, tensor
from torch.utils.data import Dataset
from typing import Tuple

from src.utils.tokenization import tokenizer


# Special tokens
EOT = "<|endoftext|>"


class GptDatasetV1(Dataset):
    def __init__(self, txt: str, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={EOT})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]
