import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Union
from abc import ABC, abstractmethod
from pytorch_pretrained_vit import ViT

class ClassificationWrapper(nn.Module, ABC):

    def __init__(self, model_class: nn.Module, pretrained: bool = False, num_classes: int = 1000, save_path: Optional[str] = None, load_path: Optional[str] = None, **kwargs) -> None:
        """Wrapper on for generic classification model to deal with pretraining class mismatch.

        :param model_class: Model class uninitialized
        :type model_class: nn.Module
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

        if pretrained and num_classes != 1000:
            self.model = self.initialize_pretrained_model(model_class, pretrained, num_classes, **kwargs)
        else:
            self.model = model_class(pretrained=pretrained, num_classes=num_classes, **kwargs)
        
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

    def __init__(self, **kwargs) -> None:
        super(alexnet, self).__init__(models.alexnet, **kwargs)
    
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

    def __init__(self, **kwargs) -> None:
        super(resnet18, self).__init__(models.resnet18, **kwargs)

class resnet34(resnet):

    def __init__(self, **kwargs) -> None:
        super(resnet34, self).__init__(models.resnet34, **kwargs)

class resnet50(resnet):

    def __init__(self, **kwargs) -> None:
        super(resnet50, self).__init__(models.resnet50, **kwargs)

class resnet101(resnet):

    def __init__(self, **kwargs) -> None:
        super(resnet101, self).__init__(models.resnet101, **kwargs)


class vgg(ClassificationWrapper):

    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        model = model_class(pretrained=True, **kwargs)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
        return model

class vgg13(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg13, self).__init__(models.vgg13, **kwargs)

class vgg13_bn(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg13_bn, self).__init__(models.vgg13_bn, **kwargs)

class vgg16(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg16, self).__init__(models.vgg16, **kwargs)

class vgg16_bn(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg16_bn, self).__init__(models.vgg16_bn, **kwargs)

class vgg19(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg19, self).__init__(models.vgg19, **kwargs)

class vgg19_bn(vgg):

    def __init__(self, **kwargs) -> None:
        super(vgg19_bn, self).__init__(models.vgg19_bn, **kwargs)

class vit(ClassificationWrapper):

    def __init__(self, **kwargs) -> None:
        super(vit, self).__init__(ViT, **kwargs)
    
    def initialize_pretrained_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        kwargs['pretrained'] = pretrained
        kwargs['num_classes'] = num_classes
        return model_class(**kwargs)