import torch


Device = torch.device


def get_device(device_type: str = "auto") -> Device:
    if device_type != "auto":
        return Device(device_type)
    if torch.cuda.is_available():
        return Device("cuda")
    if torch.backends.mps.is_available():  # Need 'and torch.backends.mps.is_built()' ?
        return Device("mps")
    return Device("cpu")
