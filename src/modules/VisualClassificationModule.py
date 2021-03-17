import torch
from omegaconf import DictConfig
from utils.metrics import correct
from modules.BaseMLModule import BaseMLModule
from typing import Tuple, Union, Callable, Optional
from torch.utils.data import Dataset

from models.models_dict import models_dict
from optimizers.optimizers_dict import optimizers_dict
from schedulers.schedulers_dict import schedulers_dict
from datasets.datasets_dict import datasets_dict
from loss_fn.loss_fn_dict import loss_fn_dict
from transforms.transforms_dict import transforms_dict
from transform_inverters.inverter_dict import inverter_dict
from class_mapper.class_mapper_dict import class_mapper_dict

from utils.conf_utils import remove_internal_conf_params
from torchvision.utils import save_image

class VisualClassificationModule(BaseMLModule):

    def __init__(self, conf: DictConfig, device: torch.device) -> None:
        super(VisualClassificationModule, self).__init__(conf, device)
        self.inverter_dict = inverter_dict
        self.im_transform_inverter, self.num_images_save = self.init_im_transform_inverter(conf)
        self.saved = False
    
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
            'transforms_dict': transforms_dict,
            'class_mapper_dict': class_mapper_dict
        }
        return attrs
    
    def forward_train(self, data: Tuple) -> Optional[dict]:
        """Runs one iteration of classification training.

        :param data: Tuple of data to be used for training
        :type data: Tuple
        :return: Dict to be used for logging
        :rtype: Optional[dict]
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

        loss.backward()

        return logging_dict

    def forward_val(self, data: Tuple) -> Optional[dict]:
        """Runs one iteration of classification validation.

        :param data: Tuple of data to be used for validation
        :type data: Tuple
        :return: Dict to be used for logging
        :rtype: Optional[dict]
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
    
    def forward_test(self, data: Tuple) -> Optional[dict]:
        """Runs one iteration of classification testing.
        This function requires that the test model of the
        module be defined.

        :param data: Tuple of data to be used for testing
        :type data: Tuple
        :return: Dict to be used for logging
        :rtype: Optional[dict]
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

        if self.num_images_save and not self.saved:
            self.save_images(images, pred)
            self.saved = True

        return logging_dict
    
    def init_im_transform_inverter(self, conf: DictConfig) -> Tuple[Optional[Callable], Optional[int]]:
        """Initialize an optional componenet that will invert image
        transformations before saving images.

        :param conf: Configuration file
        :type conf: DictConfig
        :raises NotImplementedError: Thrown if inverter name not in inverter dict
        :return: Inverter, number of images to save or None, None if no save module
        specified
        :rtype: Tuple[Optional[Callable], Optional[int]]
        """
        save_image_conf = conf.save_images if "save_images" in conf else None
        if not save_image_conf:
            return None, None
        
        num_images = save_image_conf.num_images

        if 'inverter' not in save_image_conf:
            return None, num_images
        
        inverter_conf = save_image_conf.inverter
        inv_name = inverter_conf._name

        if inv_name not in self.inverter_dict:
            raise NotImplementedError
        
        inverter_conf = dict(inverter_conf)
        inv_params = remove_internal_conf_params(inverter_conf)
        inverter_class = self.inverter_dict[inv_name]
        
        return inverter_class(**inv_params), num_images
    
    def save_images(self, images: torch.Tensor, pred: torch.Tensor) -> None:
        """Takes in a batch samples and saves a number of images if the runner
        conf variable runner.save_images = <num_images> is set.

        :param images: Batch of images
        :type images: torch.Tensor
        :param pred: Batch of predictions, should be of shape [batch_size, c]
        :type pred: torch.Tensor
        :param epoch: [description], defaults to None
        :type epoch: int, optional
        """
        images = images[:self.num_images_save].cpu()
        pred = pred[:self.num_images_save].cpu()
        _, labels = pred.max(dim=1)

        label_to_save = [str(x.item()) for x in labels] # type: ignore
        label_str = '_'.join(label_to_save)

        if self.im_transform_inverter:
            images = self.im_transform_inverter(images)
        
        save_image(images, label_str + '.png')
