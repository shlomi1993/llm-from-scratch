from torch.utils.data import DataLoader

from src.data.datasets import GptDatasetV1


class GptDataloaderV1(DataLoader):

    def __init__(self, text: str, batch_size: int, max_length: int, stride: int, shuffle: bool = True,
                 drop_last: bool = True, num_workers: int = 0) -> None:
        dataset = GptDatasetV1(text, max_length, stride)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
