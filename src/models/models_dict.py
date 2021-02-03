from models.ResnetWrapper import resnet
import torchvision.models as models

models_dict = {
    'resnet18': resnet,
    'resnet34': resnet,
    'resnet50': resnet,
    'alexnet': models.alexnet,
    'vgg16' : models.vgg16
}