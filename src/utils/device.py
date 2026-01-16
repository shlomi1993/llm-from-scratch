import torch


Device = torch.device


def get_device(device_type: str = "auto") -> Device:
    """
    Get the PyTorch device for model computations.

    If device_type is "auto", the function automatically selects the best available device (CUDA > MPS > CPU).
    Otherwise, it uses the specified device type.

    Args:
        device_type (str): A string representing the requested device type ("auto", "cuda", "mps", or "cpu"). Default is "auto".

    Returns:
        torch.device: The selected PyTorch device.
    """
    if device_type != "auto":
        return Device(device_type)
    if torch.cuda.is_available():
        return Device("cuda")
    if torch.backends.mps.is_available():
        return Device("mps")
    return Device("cpu")
