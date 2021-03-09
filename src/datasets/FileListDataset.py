import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, Tuple
from PIL import Image

class FileListDataset(Dataset):

    def __init__(self, data_root: str, data_file: str, transform: Any = None, loader: Callable[[str], Any] = default_loader, class_idx_offset: int = -1) -> None:
        """Initializes a dataset which loads images based on a text file which contains,
        rows of file_name class_index for a particular dataset.

        :param data_root: Root location of data, the files in data_file should be 
        relative to this.
        :type data_root: str
        :param data_file: File containing list of image paths and class_indices, each row
        should contain one image path and the corresponding class index in {0,...,c-1} or 
        {1,...,c} in which case class_idx_offset should be set to -1.
        :type data_file: str
        :param transform: Image transforms for samples, defaults to None
        :type transform: Any, optional
        :param loader: Loader used to load images for dataset, defaults to default_loader
        :type loader: Callable[[str], Any], optional
        :param class_idx_offset: Used when class index is in range [1,c] in order to
        bring the range to [0,c-1] which is expected by PyTorch, defaults to -1
        :type class_idx_offset: int, optional
        """
        self.transform = transform
        self.samples = []
        self.classes = set()
        self.class_to_idx = {}
        self.loader = loader

        with open(data_file, 'r') as data_list:
            for line in data_list:
                img_path, cl = tuple(line.strip().split(' '))
                cl_name = img_path.split('/')[1] if '/' in img_path else cl
                rect_class_idx = int(cl) + class_idx_offset

                self.samples.append((os.path.join(data_root, img_path), rect_class_idx))
                self.classes.add(rect_class_idx)
                self.class_to_idx[rect_class_idx] = cl_name
        
        self.classes = list(self.classes) # type: ignore
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return dataset item by index, applying transformation if they are
        available.

        :param idx: Index of sample to be returned
        :type idx: int
        :return: Tuple of image data and class label as int
        :rtype: Tuple[Any, int]
        """
        img_fp, cl = self.samples[idx]
        img = self.loader(img_fp)

        if self.transform:
            img = self.transform(img)
        
        return img, cl
    
    def __len__(self) -> int:
        """Return length of the dataset.

        :return: dataset length
        :rtype: int
        """
        return len(self.samples)
        