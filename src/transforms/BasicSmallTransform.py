from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Any, Union

class BasicSmallTransform:

    def __init__(self, mean: List[float] = [0.4914, 0.4822, 0.4465], std: List[float] = [0.2023, 0.1994, 0.2010], image_size: List[int] = [32,32], padding: Union[int, Tuple[int,int], List[int]] = 0):
        self.transform = transforms.Compose([
            transforms.Pad(padding),
            lambda x: x.convert('RGB'),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])
    
    def __call__(self, x):
        return self.transform(x)

class AffineAugmentation:

    def __init__(self, angle: int = 0, shift: Tuple[int, int] = (0,0), scale: float = 1.0, shear: Tuple[int, int] = (0,0), mean: List[float] = [0.4914, 0.4822, 0.4465], std: List[float] = [0.2023, 0.1994, 0.2010], image_size: List[int] = [32,32], padding: Union[int, Tuple[int,int], List[int]] = 0) -> None:
        """Initialize shift augmentations

        :param shift: shift in the x and y direction respectively
        :type shift: Tuple[int, int]
        """
        self.transform = transforms.Compose([
            transforms.Pad(padding),
            lambda x: x.convert('RGB'),
            lambda x: TF.affine(x, angle, tuple(shift), scale, shear),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, x) -> Any:
        return self.transform(x)
