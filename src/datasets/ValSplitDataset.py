from torch.utils.data import Dataset
from typing import Tuple, Any, Optional, Callable
from torchvision import datasets

class ValSplitDataset(Dataset):

    def __init__(self, dataset_class: Dataset, mode: str = 'train', split: float = 0.2, **kwargs) -> None:
        """Initializes a ValSplitDataset.

        :param dataset_class: Underlying dataset class to wrap around, which has not been
        initialized
        :type dataset: Dataset
        :param mode: mode of dataset train, val or test, defaults to 'train'
        :type mode: str, optional
        :param split: Percentage of train dataset to use for dataset, defaults to 0.2
        :type split: float, optional
        """
        assert mode == 'train' or mode == 'val' or mode == 'test'

        train = (mode == 'train' or mode == 'val')
        self.dataset = dataset_class(train=train, **kwargs) # type: ignore
        dataset_len = len(self.dataset)

        train_bound = int(dataset_len * split)
        if mode == 'train':
            self.valid_indices = list(range(train_bound))
        if mode == 'val':
            self.valid_indices = list(range(train_bound, dataset_len))
        if mode == 'test':
            self.valid_indices = list(range(dataset_len))

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Return item in dataset at a particular index.

        :param index: Index at which to get data sample
        :type index: int
        :return: Dataset sample
        :rtype: Tuple[Any, int]
        """
        img, cl = self.dataset[self.valid_indices[index]] # type: ignore
        
        return (img, cl)
    
    def __len__(self) -> int:
        """Return length of dataset.

        :return: Length of dataset
        :rtype: int
        """
        return len(self.valid_indices) # type: ignore


class CIFAR10(ValSplitDataset):

    def __init__(self, **kwargs) -> None:
        super(CIFAR10, self).__init__(datasets.CIFAR10, **kwargs)