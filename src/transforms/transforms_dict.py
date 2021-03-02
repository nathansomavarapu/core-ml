import transforms.BasicSmallTransform as BasicSmallTransform
from transforms.BasicImTransform import BasicImTransform
from transforms.CifarTransform import CifarTransform
import transforms.BasicAugmentation as BasicAugmentation
from torchvision.transforms import ToTensor

transforms_dict = {
    'to_tensor': ToTensor,
    'basic_im': BasicImTransform,
    'basic_small': BasicSmallTransform.BasicSmallTransform,
    'cifar': CifarTransform,
    'basic_aug': BasicAugmentation.BasicAugmentation,
    'affine': BasicSmallTransform.AffineAugmentation,
    'cifar_vit': BasicAugmentation.CifarVITTransform,
    'basic_vit': BasicAugmentation.BasicVITTransform
}