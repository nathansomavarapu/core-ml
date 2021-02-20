import copy
import os
from abc import ABC
import torch
from torch.utils.data import ConcatDataset
from FileListDataset import FileListDataset
from torchvision.datasets import ImageFolder
from dg_datasets_dict import dg_root_dict, dg_path_dict

class DGDataset(ABC):

    def __init__(self, dataset_name: str, target: str, mode: str = 'train', loader: Callable = None, transform: Callable = None) -> None:
        """Initialize base DGDataset

        :param dataset_name: [description]
        :type dataset_name: str
        :param target: Target domain, can be one of the 4 PACS domains
        :type target: str
        :param mode: Dataset mode can be train, val or test, defaults to 'train'
        :type mode: str, optional
        :param loader: Image loader should be changed for stylization, defaults to None
        :type loader: Callable, optional
        :param transform: Transforms for the images in the dataset, defaults to None
        :type transform: Callable, optional
        """
        self.dataset_name = dataset_name
        self.target = target
        self.loader = loader
        self.transform = transform

        assert dataset_name in dg_root_dict.keys() and dataset_name in dg_path_dict.keys()

        self.dataset_root = dg_root_dict[dataset_name]
        trainsets = list(dg_path_dict[dataset_name].keys())
        self.trainsets = copy.copy(trainsets)
        self.trainsets.remove(target)

        self.path_dict = dg_path_dict[dataset_name]

        assert mode == 'train' or mode == 'val' or mode == 'test'

        self.dataset = self.get_datasets(mode)
    
    def get_datasets(self, mode) -> torch.util.Dataset:
        """Abstract class which needs to be overridden by subclass. Given a mode
        it returns the dataset from based on the mode and the DG problem and the
        target domain.

        :return: DG dataset based on mode
        :rtype: torch.util.Dataset
        """
        dataset = None
        if mode == 'train' or 'val':
            dataset_list = []
            for domain in self.trainsets:
                data_fp = self.path_dict[domain][mode]
                dataset_list.append(self.generate_dataset(self.dataset_root, data_fp))
            dataset = ConcatDataset(datasets)
        else:
            data_fp = self.path_dict[domain]['test']
            dataset = self.generate_dataset(self.dataset_root, data_fp)
        
        return dataset
    
    @abstractmethod
    def generate_dataset(self, dataset_root: str, data_fp: str) -> torch.util.Dataset:
        """Generate the appropriate dataset for the DG setting that is selected
        will be either a FileListDataset or an ImageFolderDataset. Abstact method to be
        implemented by subclass.

        :param dataset_root: Root path of data
        :type dataset_root: str
        :param data_fp: Path to either a text file containing paths to images or
        a path to the data itself arranged in the structure assumed by the PyTorch
        ImageFolder Dataset
        :type data_fp: str
        :return: Torch Dataset of the appropriate type
        :rtype: torch.util.Dataset
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Get a sample from the dataset based on idx.

        :param idx: Index of dataset sample
        :type idx: int
        :return: Dataset sample
        :rtype: Tuple[Any, int]
        """
        return self.dataset[idx]
    
    def __len__(self) -> int:
        """Return length of the dataset.

        :return: Length of dataset
        :rtype: int
        """
        return len(self.dataset)

class PACSDataset(DGDataset):

    def __init__(self, target: str, **kwargs) -> None:
        super(DGDataset, self).__init__('pacs', **kwargs)
    
    def generate_dataset(self, dataset_root: str, data_fp: str) -> torch.util.Dataset:
        return FileListDataset(dataset_root, data_fp, transform=self.transform, loader=self.loader)

class VLCSDataset(DGDataset):

    def __init__(self, target: str, **kwargs) -> None:
        super(DGDataset, self).__init__('vlcs', **kwargs)
    
    def generate_dataset(self, dataset_root: str, data_fp: str) -> torch.util.Dataset:
        return ImageFolder(os.path.join(dataset_root, data_fp), transform=self.transform, loader=self.loader)

class OHDataset(DGDataset):

    def __init__(self, target: str, **kwargs) -> None:
        super(DGDataset, self).__init__('oh', **kwargs)
    
    def generate_dataset(self, dataset_root: str, data_fp: str) -> torch.util.Dataset:
        return ImageFolder(os.path.join(dataset_root, data_fp), transform=self.transform, loader=self.loader)

class DomainNetDataset(DGDataset):

    def __init__(self, target: str, **kwargs) -> None:
        super(DGDataset, self).__init__('domainnet', **kwargs)
    
    def generate_dataset(self, dataset_root: str, data_fp: str) -> torch.util.Dataset:
        return ImageFolder(os.path.join(dataset_root, data_fp), transform=self.transform, loader=self.loader)