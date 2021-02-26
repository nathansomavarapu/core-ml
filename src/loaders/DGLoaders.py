import copy
import glob
import os
import random
from omegaconf import DictConfig
from PIL import Image
from datasets.dg_paths_dict import dg_path_dict
from loaders.dg_paths_dict import stylized_dataset_fp_dict
from abc import ABC, abstractmethod
from typing import Tuple

class StyleLoader(ABC):

    def __init__(self, dataset_name: str, target: str, p: float = 0.1, inter_source=False, intra_source=False) -> None:
        """Initialize StyleLoader with the the path to style images and 
        the probability of converting to a stylized image.

        :param dataset_name: DG dataset name
        :type dataset_name: str
        :param target: Target dataset name
        :type target: str
        :param p: Probability of returning stylized image, defaults to 0.1
        :type p: float, optional
        :param inter_source: Enables inter-source stylization, defaults to False
        :type inter_source: bool, optional
        :param intra_source: Enables intra-source stylization, defaults to False
        :type intra_source: bool, optional
        """
        assert (not intra_source and not inter_source) or (inter_source != intra_source)
        self.style_source = None
        if intra_source:
            self.style_source = 'intra_source'
        elif inter_source:
            self.style_source = 'inter_source'
        else:
            self.style_source = 'painting'

        style_fp_dict = stylized_dataset_fp_dict[self.style_source]
        assert dataset_name in style_fp_dict.keys() and dataset_name in dg_path_dict

        stylize_image_root = style_fp_dict[dataset_name]
        self.stylize_image_root = stylize_image_root

        assert target in dg_path_dict[dataset_name]
        self.other_styles = list(dg_path_dict[dataset_name].keys())
        self.other_styles.remove(target)

        self.replaced_with_style = False

        self.split_point, self.image_fname_len = self.get_split_fname_len(dataset_name)

        self.p = p
    
    def __call__(self, image_path: str) -> Image:
        """This function is called when the object is called,
        it takes in a image path and returns a PIL image or a stylized
        version of a PIL image with some probability. This function assumes
        a particular structure for the organization of the DG dataset and the
        stylization dataset.
        DGDataset:
            Domain1:
                [train/crossval/test] (optional level of nesting)
                    Class1:
                        Image1
                        Image2
                        Image3
                        ...
                    Class2:
                    ...
                    ClassN:
                    ...
            Domain2:
                [train/crossval/test] (optional level of nesting)
                    Class1:
                        ...
                    Class2:
                        ...
                    ...
                    ClassN:
                    ...
            ...
            DomainN:
                ...
        :param image_path: Path to original image
        :type image_path: str
        :return: PIL image.
        :rtype: Image
        """
        draw = random.random()
        self.replaced_with_style = False

        if draw < self.p:
            self.replaced_with_style = True
            image_path_split = image_path.split('/')[-self.split_point:]
            
            # Painting Stylization
            if self.style_source == 'inter_source' or self.style_source == 'intra_source':
                curr_style = image_path_split[0]

                # Inter-source Stylization
                if self.style_source == 'inter_source':
                    styles = copy.copy(self.other_styles)
                    styles.remove(curr_style)
                    new_style = random.choice(styles)
                # Intra-source Stylization
                elif self.style_source == 'intra_source':
                    new_style = curr_style
                
                image_path_split[0] = curr_style + '_as_' + new_style

            image_path_search = '/'.join(image_path_split)[:-self.image_fname_len]
            stylize_search_fp = os.path.join(self.stylize_image_root, image_path_search + '-*')
            image_path_list = glob.glob(stylize_search_fp)
            
            assert len(image_path_list) <= 1 and image_path_list is not None
            if len(image_path_list) == 0:
                print('Missing image for style augmentation at, {}. Using original image.'.format(stylize_search_fp))
            else:
                image_path = image_path_list[0]
        
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
            return img
        
    def was_replaced(self) -> bool:
        """Indicates whether the last image returned was
        stylized or not.

        :return: Replaced last image with stylized version
        :rtype: bool
        """
        return self.replaced_with_style
    
    @abstractmethod
    def get_split_fname_len(self, dataset_name: str) -> Tuple[int,int]:
        """Returns how much of the path should be kept based on the
        current dataset along with the length of '.' + image extension,
        i.e. 4 for .jpg 5 for .jpeg. The first values is based on the amount 
        of nesting in the file structure and is overridden by subclass.

        :param dataset_name: DG dataset name
        :type dataset_name: str
        :return: Number of values that should be kept when the the fp
        is split based on "/" and the filename extension length
        :rtype: Tuple[int, int]
        """
        raise NotImplementedError

class PACSStyleLoader(StyleLoader):

    def get_split_fname_len(self, dataset_name: str) -> Tuple[int, int]:
        return (3,4)

class VLCSStyleLoader(StyleLoader):

    def get_split_fname_len(self, dataset_name: str) -> Tuple[int, int]:
        return (4,4)

class OHStyleLoader(StyleLoader):

    def get_split_fname_len(self, dataset_name: str) -> Tuple[int, int]:
        return (4,4)

class DomainNetStyleLoader(StyleLoader):

    def get_split_fname_len(self, dataset_name: str) -> Tuple[int, int]:
        return (3,4)
