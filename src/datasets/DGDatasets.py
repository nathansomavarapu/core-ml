import copy
import os
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Any, Optional
import torch
from torch.utils.data import ConcatDataset, Dataset
from datasets.FileListDataset import FileListDataset
from torchvision.datasets import ImageFolder
from datasets.dg_paths_dict import dg_root_dict, dg_path_dict
from torchvision.datasets.folder import default_loader

class DGDataset(ABC):

    def __init__(self, dataset_name: str, target: str, mode: str = 'train', loader: Callable[[str], Any] = default_loader, transform: Optional[Callable] = None) -> None:
        """Initialize DGDataset.

        :param dataset_name: Name of the dataset the available datasets can
        be found in dg_datasets_dict file in the datasets directory
        :type dataset_name: str
        :param target: Target domain, can be one of the n domains in the dataset
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

        assert dataset_name in dg_path_dict.keys()

        self.dataset_root = dg_root_dict[dataset_name] if dataset_name in dg_root_dict else None
        trainsets = list(dg_path_dict[dataset_name].keys())
        
        assert target in trainsets

        self.trainsets = copy.copy(trainsets)
        self.trainsets.remove(target)

        self.path_dict = dg_path_dict[dataset_name]

        assert mode == 'train' or mode == 'val' or mode == 'test'

        self.dataset = self.get_datasets(mode)
    
    def get_datasets(self, mode) -> Dataset:
        """Abstract class which needs to be overridden by subclass. Given a mode
        it returns the dataset from based on the mode and the DG problem and the
        target domain.

        :return: DG dataset based on mode
        :rtype: Dataset
        """
        dataset = None
        if mode == 'train' or mode == 'val':
            dataset_list = []
            for domain in self.trainsets:
                data_fp = self.path_dict[domain][mode]
                dataset_list.append(self.generate_dataset(self.dataset_root, data_fp))
            dataset = ConcatDataset(dataset_list)
        else:
            data_fp = self.path_dict[self.target]['test']
            dataset = self.generate_dataset(self.dataset_root, data_fp)
        
        return dataset # type: ignore
    
    @abstractmethod
    def generate_dataset(self, dataset_root:  Optional[str], data_fp: str) -> Dataset:
        """Generate the appropriate dataset for the DG setting that is selected
        will be either a FileListDataset or an ImageFolderDataset. Abstact method to be
        implemented by subclass.

        :param dataset_root: Root path of data, None for the datasets where there
        is no file list in which case the file root is the data_fp
        :type dataset_root: Optional[str]
        :param data_fp: Path to either a text file containing paths to images or
        a path to the data itself arranged in the structure assumed by the PyTorch
        ImageFolder Dataset
        :type data_fp: str
        :raises NotImplementedError: Raised if subclass does not implement abstract method
        :return: Torch Dataset of the appropriate type
        :rtype: Dataset
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset based on idx.

        :param idx: Index of dataset sample
        :type idx: int
        :return: Dataset sample, tuple of image and class index
        :rtype: Tuple[torch.Tensor, int]
        """
        return self.dataset[idx]
    
    def __len__(self) -> int:
        """Return length of the dataset.

        :return: Length of dataset
        :rtype: int
        """
        return len(self.dataset) # type: ignore

class PACSDataset(DGDataset):

    def __init__(self, **kwargs) -> None:
        super(PACSDataset, self).__init__('pacs', **kwargs)
    
    def generate_dataset(self, dataset_root:  Optional[str], data_fp: str) -> Dataset:
        return FileListDataset(dataset_root, data_fp, transform=self.transform, loader=self.loader)

class VLCSDataset(DGDataset):

    def __init__(self, **kwargs) -> None:
        super(VLCSDataset, self).__init__('vlcs', **kwargs)
    
    def generate_dataset(self, dataset_root: Optional[str], data_fp: str) -> Dataset:
        return ImageFolder(data_fp, transform=self.transform, loader=self.loader)

class OHDataset(DGDataset):

    def __init__(self, **kwargs) -> None:
        super(OHDataset, self).__init__('oh', **kwargs)
    
    def generate_dataset(self, dataset_root:  Optional[str], data_fp: str) -> Dataset:
        return ImageFolder(data_fp, transform=self.transform, loader=self.loader) # type: ignore

class DomainNetDataset(DGDataset):

    def __init__(self, **kwargs) -> None:
        super(DomainNetDataset, self).__init__('domainnet', **kwargs)
    
    def generate_dataset(self, dataset_root:  Optional[str], data_fp: str) -> Dataset:
        return FileListDataset(dataset_root, data_fp, transform=self.transform, loader=self.loader, class_idx_offset=0)