from transforms.BasicSmallTransform import BasicSmallTransform
from transforms.BasicImTransform import BasicImTransform
from transforms.CifarTransform import CifarTransform

transforms_dict = {
    'basic_im': BasicImTransform,
    'basic_small': BasicSmallTransform,
    'cifar': CifarTransform
}