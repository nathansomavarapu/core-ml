import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from abc import ABC, abstractmethod

class ClassificationWrapper(nn.Module, ABC):

    def __init__(self, pretrained: bool = False, num_classes: int = 1000, save_path: Optional[str] = None, load_path: Optional[str] = None, **kwargs) -> None:
        """Wrapper on for generic classification model to deal with pretraining class mismatch.

        :param pretrained: imagenet pretrained weights, defaults to False
        :type pretrained: bool, optional
        :param num_classes: number of classes, defaults to 1000
        :type num_classes: int, optional
        :param save_path: Path to save model to, None indicates no model to save, defaults to None
        :type save_path: Optional[str], optional
        :param load_path: Path to load model from, None indicates no model to load, defaults to None
        :type load_path: Optional[str], optional
        """
        super(ClassificationWrapper, self).__init__()

        model_class = self.get_model_class()

        if pretrained and num_classes != 1000:
            self.model = self.initialize_pretrained_model(model_class, pretrained, num_classes, **kwargs)
        else:
            self.model = model_class(pretrained=pretrained, num_classes=num_classes)
        
        self.save_path = save_path

        if load_path:
            self.model.load_state_dict(torch.load(load_path, map_location='cpu'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the network.

        :param x: Input image tensor
        :type x: torch.Tensor
        :return: Output logits tensor
        :rtype: torch.Tensor
        """
        return self.model(x)
    
    def save_model(self, epoch=None) -> None:
        """Save the model to the path selected during initialization using epoch
        info if available.
        """
        torch.save(self.model.state_dict(), self.save_path) # type: ignore
    
    @abstractmethod
    def get_model_class(self) -> nn.Module:
        """Return the class of the model of interest, this method is an
        abstract method and must be overridden by the child class.

        :return: Un-instantiated class of model
        :rtype: nn.Module
        """
    
    @abstractmethod
    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        """Return the model instantated with the architecture specific instantiations
        needed based on pretrained and num_classes. The model class attribute should be
        modified in place. This method is an abstract method and must be overridden by 
        the child class.

        :param model_class: Un-instantiated class of model
        :type model_class: nn.Module
        :param pretrained: Pretrained model or not
        :type pretrained: bool
        :param num_classes: Number of classes for the model
        :type num_classes: int
        :return: Instantiated model
        :rtype: nn.Module
        """

class alexnet(ClassificationWrapper):

    def get_model_class(self) -> nn.Module:
        return models.alexnet
    
    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        model = model_class(pretrained=True, **kwargs)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
        return model

class resnet(ClassificationWrapper):

    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        model = model_class(pretrained=True, **kwargs)
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        return model

class resnet18(resnet):

    def get_model_class(self) -> nn.Module:
        return models.resnet18

class resnet34(resnet):

    def get_model_class(self) -> nn.Module:
        return models.resnet34

class resnet50(resnet):

    def get_model_class(self) -> nn.Module:
        return models.resnet50

class resnet101(resnet):

    def get_model_class(self) -> nn.Module:
        return models.resnet101

class vgg(ClassificationWrapper):

    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        model = model_class(pretrained=True, **kwargs)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
        return model

class vgg13(vgg):

    def get_model_class(self):
        return models.vgg13

class vgg13_bn(vgg):

    def get_model_class(self):
        return models.vgg13_bn

class vgg16(vgg):

    def get_model_class(self):
        return models.vgg16

class vgg16_bn(vgg):

    def get_model_class(self):
        return models.vgg16_bn

class vgg19(vgg):

    def get_model_class(self):
        return models.vgg19

class vgg19_bn(vgg):

    def get_model_class(self):
        return models.vgg19_bn
