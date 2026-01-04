from torch.utils.data import Dataset
from src.utils.tokenization import tokenizer
from src.data.formatting import format_input


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict[str, str]]) -> None:
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]
