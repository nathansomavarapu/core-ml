from torch.utils.data import Dataset
from typing import Tuple, Any, Optional, Callable
from torchvision import datasets

class ValSplitDataset(Dataset):
    train_dataset = None
    test_dataset = None

    def __init__(self, dataset: Optional[Dataset], mode: str = 'train', split: float = 0.2, transform: Callable = None) -> None:
        """Initializes a ValSplitDataset, used to provide different val transforms than train
        transforms.

        :param dataset: Underlying dataset to wrap around, which has been initialized
        none can be passed if the train dataset has already been initialized
        :type dataset: Optional[Dataset]
        :param mode: mode of dataset train, val or test, defaults to 'train'
        :type mode: str, optional
        :param split: Percentage of train dataset to use for dataset, defaults to 0.2
        :type split: float, optional
        :param transform: Image transformations, defaults to None
        :type transform: callable, optional
        """
        assert mode == 'train' or mode == 'val' or mode == 'test'

        if not ValSplitDataset.train_dataset and mode == 'train' or mode == 'val':
            train_dataset = dataset
        elif not ValSplitDataset.test_dataset and mode == 'test':
            test_dataset = dataset

        self.dataset = train_dataset if mode == 'train' or mode == 'val' else test_dataset
        dataset_len = len(self.dataset) # type: ignore

        train_bound = int(dataset_len * split)
        if mode == 'train':
            self.valid_indices = list(range(train_bound))
        if mode == 'val':
            self.valid_indices = list(range(train_bound, dataset_len))
        if mode == 'test':
            self.valid_indices = list(range(dataset_len))
        
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Return item in dataset at a particular index.

        :param index: Index at which to get data sample
        :type index: int
        :return: Dataset sample
        :rtype: Tuple[Any, int]
        """
        img, cl = self.dataset[self.valid_indices[index]] # type: ignore

        if self.transform:
            img = self.transform(img)
        
        return (img, cl)
    
    def __len__(self) -> int:
        """Return length of dataset.

        :return: Length of dataset
        :rtype: int
        """
        return len(self.valid_indices) # type: ignore


class CIFAR10(ValSplitDataset):

    def __init__(self, mode = 'train', split=0.2, transform=None, **kwargs) -> None:
        train = (mode == 'train' or mode == 'val')
        dataset = None
        if (train and not ValSplitDataset.train_dataset) or (not train and not ValSplitDataset.test_dataset):
            dataset = datasets.CIFAR10(train=train, **kwargs)
        super(CIFAR10, self).__init__(dataset, mode=mode, split=split, transform=transform)

class MNIST(ValSplitDataset):

    def __init__(self, mode='train', split=0.2, transform=None, **kwargs) -> None:
        train = (mode == 'train' or mode == 'val')
        dataset = None
        if (train and not ValSplitDataset.train_dataset) or (not train and not ValSplitDataset.test_dataset):
            dataset = datasets.MNIST(train=train, **kwargs)
        super(MNIST, self).__init__(dataset, mode=mode, split=split, transform=transform)