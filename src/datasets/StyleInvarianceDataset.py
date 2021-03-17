import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple, Optional

from datasets.DGDatasets import PACSDataset
from loaders.DGLoaders import PACSStyleLoader

# NOTE: Currently only supports PACS

class StyleInvarianceDataset(Dataset):

    def __init__(self, basic_dataset: Dataset, style_dataset: Dataset) -> None:
        """Initialize Style Invariance Testing dataset. Loads
        both stylized and un-stylized images and returns them.

        :param basic_dataset: Dataset with basic loader
        :type basic_dataset: Dataset
        :param style_dataset: Dataset with style loader
        :type style_dataset: Dataset
        """
        self.basic_loader_dataset = basic_dataset
        self.style_loader_dataset = style_dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get stylized sample and basic sample

        :param idx: The index at which to get the underlying sample
        :type idx: int
        :return: Pair of images, basic sample and stylized sample in
        that order
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        img1 = self.basic_loader_dataset[idx][0]
        img2 = self.style_loader_dataset[idx][0]

        return (img1, img2)
    
    def __len__(self) -> int:
        """Get the length of the dataset.

        :return: Length of dataset
        :rtype: int
        """
        assert len(self.basic_loader_dataset) == len(self.style_loader_dataset) # type: ignore
        return len(self.basic_loader_dataset) # type: ignore
        

class PACSStyleInvarianceDataset(StyleInvarianceDataset):

    def __init__(self, target: str, transform: Optional[Callable] = None) -> None:
        """Initialize PACS Style Invariance Dataset.

        :param target: Target Domain
        :type target: str
        :param transform: Image transform, defaults to None
        :type transform: Optional[Callable], optional
        """
        super(PACSStyleInvarianceDataset, self).__init__(
            PACSDataset(target=target, mode='test', transform=transform),
            PACSDataset(target=target, mode='test', loader=PACSStyleLoader('pacs', target, p=1.0), transform=transform)
            )
