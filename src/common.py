import torch

from tiktoken import Encoding
from torch import Tensor

Device = torch.device


def get_device(device_arg: str = "auto") -> Device:
    if device_arg != "auto":
        return Device(device_arg)
    if torch.cuda.is_available():
        return Device("cuda")
    if torch.backends.mps.is_available():  # Need 'and torch.backends.mps.is_built()' ?
        return Device("mps")
    return Device("cpu")


def text_to_token_ids(text: str, tokenizer: Encoding) -> Tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: Tensor, tokenizer: Encoding) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
