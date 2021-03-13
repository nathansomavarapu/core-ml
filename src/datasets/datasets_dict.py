from datasets import ValSplitDataset
from datasets.MNISTCifarDataset import MNISTCifar10Wrapper
from datasets.MNISTM import MNISTM
from datasets.CueConflictDataset import CueConflictDataset

datasets_dict = {
    'cifar10': ValSplitDataset.CIFAR10,
    'mnist_cifar10': MNISTCifar10Wrapper,
    'mnist': ValSplitDataset.MNIST,
    'mnist_m': MNISTM,
    'cue_conflict': CueConflictDataset
}