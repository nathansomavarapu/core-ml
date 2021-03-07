import torch
from omegaconf import DictConfig
from utils.metrics import correct_indices, correct
from modules.VisualClassificationModule import VisualClassificationModule
from typing import Tuple, Union, Callable, Optional

from utils.conf_utils import remove_internal_conf_params


class AdversarialModule(VisualClassificationModule):

    def __init__(self, conf: DictConfig, device: torch.device) -> None:
        super(AdversarialModule, self).__init__(conf, device)

        self.epsilon = self.init_adv(conf)
    
    def init_adv(self, conf: DictConfig) -> float:
        """Return epsilon for adverserial image generation based
        on config. Value taken from, conf.adv.epsilon.

        :param conf: Configuration
        :type conf: DictConfig
        :return: Epsilon for adverserial image generation
        :rtype: float
        """
        return conf.adv.epsilon

    def init_optimizer(self, conf):
        return None
    
    def init_scheduler(self, conf):
        return None
    
    def forward_train(self, data: Tuple) -> None:
        return None

    def forward_val(self, data: Tuple) -> None:
        return None
    
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
        images.requires_grad = True

        pred = self.test_model(images)
        loss = self.loss_fn(pred, labels)

        loss.backward()

        images_adv = images.detach() + torch.sign(images.grad) * self.epsilon
        images_adv = torch.clamp(images_adv, 0, 1)

        self.test_model.zero_grad()
        pred_adv = self.test_model(images_adv)

        correct_pred = correct(pred, labels)
        correct_adv = correct_indices(pred_adv, labels)

        logging_dict = {
            'loss': float(loss.item()),
            'correct': float(correct_pred),
            'correct_adv': float(correct_adv.sum().item()),
            'total': float(images.size(0))
        }
                
        if self.num_images_save and not self.saved: # type: ignore
            self.save_images(images_adv[~correct_adv], pred_adv[~correct_adv])
            self.saved = True

        return logging_dict