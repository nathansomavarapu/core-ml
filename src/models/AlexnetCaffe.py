
import torch
import torch.nn as nn
from collections import OrderedDict
from models.ClassificationWrapper import ClassificationWrapper

class alexnet_caffe(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = False, dropout: bool = True) -> None:
        """Initialize caffe version of alexnet model. 

        :param num_classes: Number of classes, defaults to 100
        :type num_classes: int, optional
        :param pretrained: ImageNet pretraining of the model, defaults to False
        :type pretrained: bool, optional
        :param dropout: Use dropout in the model, defaults to True
        :type dropout: bool, optional
        """
        super(alexnet_caffe, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else nn.Identity()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else nn.Identity())]))

        self.class_classifier = nn.Linear(4096, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        if pretrained:
            state_dict = torch.load("/srv/share3/nsomavarapu3/weights/alexnet_caffe.pth.tar")
            del state_dict["classifier.fc8.weight"]
            del state_dict["classifier.fc8.bias"]
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)
        #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x)

class AlexnetCaffe(ClassificationWrapper):

    def get_model_class(self) -> nn.Module:
        return alexnet_caffe # type: ignore
    
    def initialize_model(self, model_class: nn.Module, pretrained: bool, num_classes: int, **kwargs) -> nn.Module:
        model = model_class(pretrained=pretrained, num_classes=num_classes, **kwargs)        
        return model