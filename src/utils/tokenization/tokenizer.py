import tiktoken
import torch


TOKENIZER = tiktoken.get_encoding("gpt2")

# Special tokens
EOT = "<|endoftext|>"
PAD_TOKEN_ID = 50656  # <|endoftext|>
IGNORE_INDEX = -100  # Used to ignore padding tokens in cross-entropy loss computation


def encode(text: str, **kwargs) -> list[int]:
    return TOKENIZER.encode(text, **kwargs)


def decode(tokens: list[int], **kwargs) -> str:
    return TOKENIZER.decode(tokens, **kwargs)


def text_to_token_ids(text: str) -> torch.Tensor:
    encoded = TOKENIZER.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return TOKENIZER.decode(flat.tolist())


def tokenize(v: str | torch.Tensor) -> torch.Tensor | str:
    return text_to_token_ids(v) if isinstance(v, str) else token_ids_to_text(v)
