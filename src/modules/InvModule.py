import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple
from omegaconf import DictConfig

from modules.VisualClassificationModule import VisualClassificationModule
from datasets.invariance_datasets_dict import invariance_datasets_dict

from utils.metrics import get_class_prediction

class InvModule(VisualClassificationModule):

    def __init__(self, conf: DictConfig, device: torch.device) -> None:
        """Initialize loader and then call parents init to setup module
        components.

        :param conf: Configuration file
        :type conf: DictConfig
        :param device: Torch device for module
        :type torch.device: device
        """
        super(InvModule, self).__init__(conf, device)
    
    def setup(self) -> dict:
        """Overrides parent setup to replace the previous dataset dictionary
        with the dg dataset dictionary.

        :return: Dictionary of class attributes to be set
        :rtype: dict
        """
        attrs = super().setup()
        attrs['datasets_dict'] = invariance_datasets_dict

        return attrs
    
    def forward_train(self) -> None:
        pass

    def forward_val(self) -> None:
        pass

    def forward_test(self, data: Tuple) -> dict:
        """Run one iteration of testing for invariance testing. 
        This involves taking two images generated in two different ways and
        seeing if the same classification is returned.

        :param data: Tuple of two images
        :type data: Tuple
        :return: Dictionary with logging info specifically the level
        of agreement on the data after transformation
        :rtype: dict
        """
        imgs1, imgs2 = data
        assert type(imgs1) is torch.Tensor and type(imgs2) is torch.Tensor

        pred1 = self.test_model(imgs1)
        pred2 = self.test_model(imgs2)

        cl1 = get_class_prediction(pred1)
        cl2 = get_class_prediction(pred2)

        correct = (cl1 == cl2).sum().item()
        total = imgs1.size(0)

        log_dict = {
            'correct': int(correct),
            'total': int(total)
        }

        return log_dict