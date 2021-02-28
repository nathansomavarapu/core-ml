import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from typing import Tuple, Callable, Any, Optional
from itertools import product
from PIL import Image
import random

from datasets.ValSplitDataset import ValSplitDataset

class MNISTCifarDataset(Dataset):

    def __init__(self, root: str = 'data', train: bool = True, total_samples: int = 100000, transform: Callable = None, download: bool = False) -> None:
        """Initialize a dataset which comabines MNIST and CIFAR10 images for the simplicity bias
        dataset

        :param root: Path to download the datasets to, defaults to 'data'
        :type root: str, optional
        :param mode: Dataset mode true implies the train dataset, while false implies test, defaults to True
        :type mode: bool, optional
        :param total_samples: Number of samples to use in the dataset, defaults to 50000
        :type total_samples: int, optional
        :param transform: Image transform to be applied to data, defaults to None
        :type transform: Callable, optional
        :param download: Download underlying datasets, defaults to False
        :type download: bool, optional
        """
        cifar10 = CIFAR10(root, train=train, download=download)
        mnist = MNIST(root, train=train, download=download)

        cifar10_truck = [i for i,c in enumerate(cifar10.targets) if c == cifar10.class_to_idx['truck']]
        cifar10_auto = [i for i,c in enumerate(cifar10.targets) if c == cifar10.class_to_idx['automobile']]

        mnist_zero = [i for i,c in enumerate(mnist.targets) if c == 0]
        mnist_one = [i for i,c in enumerate(mnist.targets) if c == 1]

        len_cl_zero = min(len(cifar10_auto) * len(mnist_zero), total_samples//2)
        len_cl_one = min(len(cifar10_truck) * len(mnist_one), total_samples//2)

        total_per_class_samples = min(len_cl_zero, len_cl_one)

        indices_cl_zero = list(product(cifar10_auto, mnist_zero))[:total_per_class_samples] # type: ignore
        indices_cl_one = list(product(cifar10_truck, mnist_one))[:total_per_class_samples]  # type: ignore

        indices_cl_zero = [(ci,mi,0) for (ci,mi) in indices_cl_zero] # type: ignore
        indices_cl_one = [(ci,mi,1) for (ci,mi) in indices_cl_one] # type: ignore

        self.sample_indices = indices_cl_zero + indices_cl_one

        self.mnist_indices = mnist_zero + mnist_one
        self.cifar10_indices = cifar10_truck + cifar10_auto

        self.cifar10 = cifar10
        self.mnist = mnist

        self.train = train

        self.transform = transform
        self.randomize_samples: Optional[str] = None
            
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Return dataset sample given an index.

        :param idx: The index at which to return a sample from
        :type idx: int
        :return: Image and integer that represents class
        :rtype: Tuple[torch.Tensor, int]
        """
        cifar_index, mnist_index, cl = self.sample_indices[index] # type: ignore
        if self.randomize_samples == 'simple':
            mnist_index = random.choice(self.mnist_indices)
        
        if self.randomize_samples == 'complex':
            cifar_idx = random.choice(self.cifar10_indices)
            
        cifar_img = self.cifar10[cifar_index][0] # type: ignore
        mnist_img = self.mnist[mnist_index][0].convert('RGB').resize((32,32)) # type: ignore

        assert cifar_img.width == mnist_img.width

        dst = Image.new('RGB', (cifar_img.width, cifar_img.height + mnist_img.height))
        dst.paste(mnist_img, (0,0))
        dst.paste(cifar_img, (0, mnist_img.height))

        if self.transform:
            dst = self.transform(dst)

        return (dst, cl) # type: ignore

    def __len__(self) -> int:
        """Return the length of the dataset.

        :return: Length of the dataset
        :rtype: int
        """
        return len(self.sample_indices)
    
    def set_randomized_test(self, randomize_samples: str):
        """Set variable that controls sample randomization during test. The options
        for sample randomization are simple, complex or None for no randomization.

        :param randomize_sample: Choose which samples if any to randomize
        :type randomize_sample: str
        """
        assert not self.train
        self.randomize_samples = randomize_samples

        assert self.randomize_samples == None or self.randomize_samples == 'simple' or self.randomize_samples == 'complex'

class MNISTCifar10Wrapper(ValSplitDataset):

    def __init__(self, **kwargs):
        super(MNISTCifar10Wrapper, self).__init__(MNISTCifarDataset, **kwargs)
    
    def set_randomized_test(self, randomize_samples: str):
        self.dataset.set_randomized_test(randomize_samples)
    