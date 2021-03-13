import torch
from typing import Tuple, Callable, Any, Optional
from PIL import Image
from torchvision.datasets import ImageFolder

class CueConflictDataset(ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None, shape_texture: str = 'shape') -> None:
        """[Initialize Cue Conflict Dataset, this is a test only dataset

        :param root: Image root for the dataset
        :type root: str
        :param transform: Image transform, defaults to None
        :type transform: Optional[Callable], optional
        :param shape_texture: Indicator for whether to return the shape class or the
        texture class, defaults to 'shape'
        :type shape_texture: str, optional
        """
        super(CueConflictDataset, self).__init__(root, transform=transform)

        self.texture_cl = []

        for sample in self.samples: # type: ignore
            img_path, _ = sample
            shape_cl, texture_cl = img_path[:-4].split('/')[-1].split('-')
            shape_cl = shape_cl[:-1]
            texture_cl = texture_cl[:-1]
            self.texture_cl.append(self.class_to_idx[texture_cl])
        
        self.shape_texture = shape_texture

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a dataset sample

        :param idx: [description]
        :type idx: int
        :return: [description]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        img, shape_cl = super().__getitem__(idx)
        if self.shape_texture == 'shape':
            return (img, shape_cl)
        elif self.shape_texture == 'texture':
            return (img, self.texture_cl[idx])
        else:
            raise NotImplementedError
        
    def set_shape_or_texture(self, shape_or_texture: str) -> None:
        """Set the dataset to return the shape class or the texture class

        :param shape_or_texture: "shape" or "texture" to return the respective
        class
        :type shape_or_texture: str
        """
        assert shape_or_texture == 'shape' or shape_or_texture == 'texture'
        self.shape_texture = shape_or_texture