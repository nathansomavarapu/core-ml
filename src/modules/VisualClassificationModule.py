import torch
from omegaconf import DictConfig
from utils.metrics import correct
from modules.BaseMLModule import BaseMLModule
from typing import Tuple, Union, Callable
from torch.utils.data import Dataset

from models.models_dict import models_dict
from optimizers.optimizers_dict import optimizers_dict
from schedulers.schedulers_dict import schedulers_dict
from datasets.datasets_dict import datasets_dict
from loss_fn.loss_fn_dict import loss_fn_dict
from transforms.transforms_dict import transforms_dict

class VisualClassificationModule(BaseMLModule):
    
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
    
    def forward_train(self, data: Tuple) -> Tuple[torch.Tensor, dict]:
        """Runs one iteration of classification training.

        :param data: Tuple of data to be used for training
        :type data: Tuple
        :return: Loss tensor to be used for optimization, dict to be used
        for logging
        :rtype: Tuple[torch.Tensor, dict]
        """
        images, labels = data
        if labels.max() > 344 or labels.min() < 0:
            breakpoint()
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