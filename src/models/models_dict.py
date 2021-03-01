from models import ClassificationWrapper
from models import AlexnetCaffe
import torchvision.models as models

models_dict = {
    'resnet18': ClassificationWrapper.resnet18,
    'resnet34': ClassificationWrapper.resnet34,
    'resnet50': ClassificationWrapper.resnet50,
    'resnet101': ClassificationWrapper.resnet101,
    'alexnet': ClassificationWrapper.alexnet,
    'alexnet_caffe': AlexnetCaffe.AlexnetCaffe,
    'vgg13': ClassificationWrapper.vgg13,
    'vgg13_bn' : ClassificationWrapper.vgg13_bn,
    'vgg16': ClassificationWrapper.vgg16,
    'vgg16_bn' : ClassificationWrapper.vgg16_bn,
    'vgg19': ClassificationWrapper.vgg19,
    'vgg19_bn' : ClassificationWrapper.vgg19_bn,
    'vit': ClassificationWrapper.vit
}