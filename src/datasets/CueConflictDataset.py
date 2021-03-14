import torch
from typing import Tuple, Callable, Any, Optional
from PIL import Image
from torchvision.datasets import ImageFolder

class CueConflictDataset(ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None, shape_texture: str = 'shape') -> None:
        """Initialize Cue Conflict Dataset, this is a test only dataset

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
        remove_samples_idxs: list = []

        for i, sample in enumerate(self.samples): # type: ignore
            img_path, shape_cl = sample
            _, texture_cl = img_path[:-4].split('/')[-1].split('-')
            texture_cl = texture_cl[:-1]
            texture_cl = self.class_to_idx[texture_cl]
            if texture_cl != shape_cl:
                self.texture_cl.append(texture_cl)
            else:
                remove_samples_idxs.append(i)
        
        remove_samples_idxs = sorted(remove_samples_idxs, reverse=True)
        for i in remove_samples_idxs:
            del self.samples[i]
        
        assert len(self.texture_cl) == len(self.samples)
        
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