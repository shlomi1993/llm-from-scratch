import torch


Device = torch.device


def get_device(device_arg: str = "auto") -> Device:
    if device_arg != "auto":
        return Device(device_arg)
    if torch.cuda.is_available():
        return Device("cuda")
    if torch.backends.mps.is_available():  # Need 'and torch.backends.mps.is_built()' ?
        return Device("mps")
    return Device("cpu")
