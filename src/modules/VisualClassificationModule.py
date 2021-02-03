import torch
from utils.metrics import correct
from modules.BaseMLModule import BaseMLModule
from typing import Tuple, Union

class VisualClassificationModule(BaseMLModule):
    
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