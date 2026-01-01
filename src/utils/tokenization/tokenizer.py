import tiktoken
import torch


TOKENIZER = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return TOKENIZER.encode(text)


def decode(tokens: list[int]) -> str:
    return TOKENIZER.decode(tokens)


def text_to_token_ids(text: str) -> torch.Tensor:
    encoded = TOKENIZER.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return TOKENIZER.decode(flat.tolist())
