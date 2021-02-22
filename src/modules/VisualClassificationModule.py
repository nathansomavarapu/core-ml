import torch
from utils.metrics import correct
from modules.BaseMLModule import BaseMLModule
from typing import Tuple, Union
from torch.utils.data import Dataset
from loaders import dg_loaders_dict
from torchvision.datasets.folder import default_loader

from models.models_dict import models_dict
from optimizers.optimizers_dict import optimizers_dict
from optimizers.schedulers_dict import schedulers_dict
from datasets.datasets_dict import datasets_dict
from loss_fn.loss_fn_dict import loss_fn_dict
from transforms.transforms_dict import transforms_dict

class VisualClassificationModule(BaseMLModule):

    def __init__(self, conf: DictConfig) -> None:
        """Initialize loader and then call parents init
        to setup module components.

        :param conf: Configuration file
        :type conf: DictConfig
        """
        self.loader = self.init_loader(conf)
        super(VisualClassificationModule, self).__init__()
    
    def setup(self) -> dict:
        """Overrides parent class method to setup dictionaries to be used for
        image classification.

        :return: Dictionary specifying attributes to be used in class
        construction
        :rtype: dict
        """
        attrs = {
            'models_dict': models_dict,
            'optimizers_dict': optimizers_dict,
            'schedulers_dict': schedulers_dict,
            'datasets_dict': datasets_dict,
            'loss_fn_dict': loss_fn_dict,
            'transforms_dict': transforms_dict
        }
        return attrs

    def init_loader(self, conf: DictConfig) -> Callable:
        """Initialize a loader for samples from the dataset. The loader choices
        are available in the loaders directory. The only use currently is for stylization
        in the DG setting. If no loader is specified in the config the default is selected.

        :param conf: Configuration File
        :type conf: DictConfig
        :return: Sample loader 
        :rtype: Callable
        """
        if 'loader' not in conf:
            return default_loader

        loader_conf = conf.loader
        loader_name = loader_conf.name

        if loader_name not in dg_loaders_dict:
            raise NotImplementedError
        
        loader_class = dg_loader_dict[loader_name]
        loader_conf = dict(loader_conf)
        del loader_conf['name']

        loader = loader_class(**loader_conf)
        return loader

    def init_trainset(self, conf: DictConfig) -> Dataset:
        """Override parent trainset function to enable custom loader.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, train
        :rtype: Dataset
        """
        trainset_conf = conf.dataset.train

        trainset_class, train_conf = self.init_generic_dataset(trainset_conf)
        trainset = trainset_class(**train_conf, transform=self.train_transform, loader=self.loader)
        return trainset

    def init_valset(self, conf: DictConfig) -> Dataset:
        """Override parent valset function to enable custom loader.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, val
        :rtype: Dataset
        """
        valset_conf = conf.dataset.val
        
        valset_class, val_conf = = self.init_generic_dataset(valset_conf)
        valset = valset_class(**val_conf, transform=self.test_transform, loader=self.loader)
        return valset
    
    def forward_train(self, data: Tuple) -> Tuple[torch.Tensor, dict]:
        """Runs one iteration of classification training.

        :param data: Tuple of data to be used for training
        :type data: Tuple
        :return: Loss tensor to be used for optimization, dict to be used
        for logging
        :rtype: Tuple[torch.Tensor, dict]
        """
        images, labels = data
        pred = self.model(images)
        loss = self.loss_fn(pred, labels)
        correct_pred = correct(pred, labels)

        logging_dict = {
            'loss': float(loss.item()),
            'correct': float(correct_pred),
            'total': float(images.size(0))
        }

        return loss, logging_dict

    def forward_val(self, data: Tuple) -> dict:
        """Runs one iteration of classification validation.

        :param data: Tuple of data to be used for validation
        :type data: Tuple
        :return: Dict to be used for logging
        :rtype: dict
        """
        images, labels = data
        pred = self.model(images)
        loss = self.loss_fn(pred, labels)
        correct_pred = correct(pred, labels)

        logging_dict = {
            'loss': float(loss.item()),
            'correct': float(correct_pred),
            'total': float(images.size(0))
        }

        return logging_dict
    
    def forward_test(self, data: Tuple) -> dict:
        """Runs one iteration of classification testing.
        This function requires that the test model of the
        module be defined.

        :param data: Tuple of data to be used for testing
        :type data: Tuple
        :return: Dict to be used for logging
        :rtype: dict
        """
        assert self.test_model is not None, "No test model avaliable, set self.test_model before testing."
        
        images, labels = data
        pred = self.test_model(images)
        loss = self.loss_fn(pred, labels)
        correct_pred = correct(pred, labels)

        logging_dict = {
            'loss': float(loss.item()),
            'correct': float(correct_pred),
            'total': float(images.size(0))
        }

        return logging_dict