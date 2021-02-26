from transforms.BasicSmallTransform import BasicSmallTransform
from transforms.BasicImTransform import BasicImTransform
from transforms.CifarTransform import CifarTransform
from transforms.BasicAugmentation import BasicAugmentation

transforms_dict = {
    'basic_im': BasicImTransform,
    'basic_small': BasicSmallTransform,
    'cifar': CifarTransform,
    'basic_aug': BasicAugmentation
}