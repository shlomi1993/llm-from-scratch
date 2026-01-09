import pandas as pd

from torch import Tensor, tensor, long
from torch.utils.data import Dataset
from typing import Tuple

from src.utils.ollama import format_input
from src.utils.tokenization import tokenizer, EOT


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


class SpamDataset(Dataset):

    def __init__(self, csv_file: str, max_length: int = None, pad_token_id: int = tokenizer.PAD_TOKEN_ID) -> None:
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Set max length: either user-provided or longest sample
        self.max_length = max_length or max(len(et) for et in self.encoded_texts)

        # Truncate sequences if they are longer than max_length
        self.encoded_texts = [et[:self.max_length] for et in self.encoded_texts]

        # Pad sequences to the longest sequence
        self.encoded_texts = [et + [pad_token_id] * (self.max_length - len(et)) for et in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        encoded = tensor(self.encoded_texts[index], dtype=long)
        label = tensor(self.data.iloc[index]["Label"], dtype=long)
        return encoded, label


class InstructionDataset(Dataset):

    def __init__(self, data: list[dict[str, str]]) -> None:
        self.data = data
        self.encoded_texts = [  # Pre-tokenize texts
            tokenizer.encode(format_input(entry) + f"\n\n### Response:\n{entry['output']}") for entry in data
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]


class InstructionDatasetWithMasking(Dataset):

    def __init__(self, data: list[dict[str, str]]) -> None:
        self.data = data

        # Separate list for instruction lengths
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

            # Collect instruction lengths
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[int, list[int]]:
        return self.instruction_lengths[index], self.encoded_texts[index]


class InstructionDatasetPhi(Dataset):

    def __init__(self, data: list[dict[str, str]]) -> None:
        self.data = data

        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = self.format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]

    @staticmethod
    def format_input_phi(entry: dict[str, str]) -> str:
        instruction_text = f"<|user|>\n{entry['instruction']}"
        input_text = f"\n{entry['input']}" if entry["input"] else ""
        return instruction_text + input_text
