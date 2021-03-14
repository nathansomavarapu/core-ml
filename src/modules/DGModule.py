import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple
from omegaconf import DictConfig
from torchvision.datasets.folder import default_loader
from modules.VisualClassificationModule import VisualClassificationModule
from datasets.dg_datasets_dict import dg_datasets_dict
from loaders.dg_loaders_dict import stylized_loader_dict

class DGModule(VisualClassificationModule):

    def __init__(self, conf: DictConfig, device: torch.device) -> None:
        """Initialize loader and then call parents init to setup module
        components.

        :param conf: Configuration file
        :type conf: DictConfig
        :param device: Torch device for module
        :type torch.device: device
        """
        self.loader = self.init_loader(conf)
        super(DGModule, self).__init__(conf, device)

    def init_loader(self, conf: DictConfig) -> Callable:
        """Initialize a loader for samples from the dataset. The loader choices
        are available in the loaders directory. The only use currently is for stylization
        in the DG setting. If no loader is specified in the config the default is selected.

        :param conf: Configuration File
        :type conf: DictConfig
        :return: Sample loader 
        :rtype: Callable
        """
        if 'loader' not in conf or conf.loader is None:
            return default_loader

        loader_conf = conf.loader
        loader_name = loader_conf._name

        if loader_name not in stylized_loader_dict:
            raise NotImplementedError
        
        loader_class = stylized_loader_dict[loader_name]
        loader_conf = dict(loader_conf)
        loader_conf = self.remove_internal_conf_params(loader_conf)

        loader = loader_class(**loader_conf)
        return loader

    def init_trainset(self, conf: DictConfig) -> Dataset:
        """Override parent trainset function to enable custom loader for stylization.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, train
        :rtype: Dataset
        """
        trainset_class, train_conf = self.init_generic_dataset(conf, 'train')
        trainset = trainset_class(**train_conf, transform=self.train_transform, loader=self.loader) # type: ignore
        return trainset

    def init_valset(self, conf: DictConfig) -> Dataset:
        """Override parent valset function to enable custom loader for stylization.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, val
        :rtype: Dataset
        """        
        valset_class, val_conf = self.init_generic_dataset(conf, 'val')
        valset = valset_class(**val_conf, transform=self.test_transform, loader=self.loader) # type: ignore
        return valset

    def setup(self) -> dict:
        """Overrides parent setup to replace the previous dataset dictionary
        with the dg dataset dictionary.

        :return: Dictionary of class attributes to be set
        :rtype: dict
        """
        attrs = super().setup()
        attrs['datasets_dict'] = dg_datasets_dict

        return attrs

    def init_generic_dataset(self, conf: DictConfig, mode: str) -> Tuple[Dataset, dict]:
        """Overrides the parent dataset initialization function to take the dataset name
        and target dataset config values in differently.

        :param conf: Configuration file at the 
        :type conf: DictConfig
        :param mode: train, val or test
        :type mode: str
        :return: Pytorch dataset and configuration dictionary
        :rtype: Tuple[Dataset, Dict]
        """
        dataset_class, dataset_conf = super().init_generic_dataset(conf, mode)
        dataset_conf['target'] = conf.target

        return dataset_class, dataset_conf