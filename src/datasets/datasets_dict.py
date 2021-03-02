from datasets import ValSplitDataset
from datasets.MNISTCifarDataset import MNISTCifar10Wrapper

datasets_dict = {
    'cifar10': ValSplitDataset.CIFAR10,
    'mnist_cifar10': MNISTCifar10Wrapper,
    'mnist': ValSplitDataset.MNIST
}