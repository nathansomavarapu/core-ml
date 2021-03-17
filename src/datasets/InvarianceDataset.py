import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple

from torchvision.datasets.folder import default_loader

class InvarianceDataset(Dataset):

    def __init__(self, dataset: Dataset, transform1: Callable, transform2: Callable) -> None:
        """Initialize an invariance dataset which takes a datasets and applies two transforms
        or two different loaders to the samples from the dataset.

        :param dataset: Underlying dataset to apply different transforms or loaders
        :type dataset: Dataset
        :param transform1: First transform on dataset
        :type transform1: Callable
        :param transform2: Second transform on dataset
        :type transform2: Callable
        """
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample transformed by transform1 and sample tranformed by transform2
        at index idx.

        :param idx: The index at which to get the underlying sample
        :type idx: int
        :return: Pair of images transformed in two different ways
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        pass

    def __len__(self) -> int:
        """Get the length of the dataset.

        :return: Length of dataset
        :rtype: int
        """
        pass
