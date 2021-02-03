import torchvision.models as models
import torch.nn as nn
import torch

from typing import Any

models_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}

class resnet(nn.Module):

    def __init__(self, name: str = 'resnet18', pretrained: bool = False, num_classes: int = 1000, **kwargs) -> None:
        """Wrapper on resnet model to deal with pretraining class mismatch.

        :param name: name of resnet model, options are
        resnet18, resnet34, resnet50 and resnet101, defaults to 'resnet18'
        :type name: str, optional
        :param pretrained: imagenet pretrained weights, defaults to False
        :type pretrained: bool, optional
        :param num_classes: number of classes, defaults to 1000
        :type num_classes: int, optional
        """
        super(resnet, self).__init__()
        
        self.model = models_dict[name]
        
        if pretrained and num_classes != 1000:
            self.model = self.model(pretrained=True, **kwargs)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)
        else:
            self.model = self.model(pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the network.

        :param x: Input image tensor
        :type x: torch.Tensor
        :return: Output logits tensor
        :rtype: torch.Tensor
        """
        return self.model(x)
