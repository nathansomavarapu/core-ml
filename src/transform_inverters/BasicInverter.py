import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Any, Union

class UndoNormalization:

    def __init__(self, mean: List[float], std: List[float]):
        """Initialize a transform to undo normalization on images before saving them.

        :param mean: Original mean used to  normalize images
        :type mean: List[float]
        :param std: Original standard deviation used to normalize images
        :type std: List[float]
        """
        self.mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalization on a torch tensor.

        :param x: Normalized torch image tensor
        :type x: torch.Tensor
        :return: Un-normalized torch image tensor
        :rtype: torch.Tensor
        """
        assert x.dim() == 4

        return (x * self.std) - self.mean

