from utils import Mode
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from abc import ABC, abstractmethod

from models.model_dict import models_dict
from optimizers.optimizers_dict import optimizers_dict
from optimizers.schedulers_dict import schedulers_dict
from datasets.datasets_dict import datasets_dict


class BaseMLModule(ABC):

    def __init__(self, conf: dict) -> None:
        self.setup()
        self.mode = Mode.EMPTY

        self.model = self.init_model(conf)
        self.test_model = None
        self.optimizer = self.init_optimizer(conf)
        self.scheduler = self.init_scheduler(conf)
        self.trainset, self.valset, self.testset = self.init_datasets(conf)

        self.config = conf

    def setup(self) -> None:
        """Method to perform any setup needed before
        instantiating other class objects.
        """
        pass

    def init_model(self, conf: dict) -> nn.Module:
        """Initialize model using the config file. The model choices 
        are available in the models directory in models_dict.py.

        :param conf: Configuration file
        :type conf: dict
        :raises NotImplementedError: Raised when model name not supported
        :return: Pytorch Module
        :rtype: nn.Module
        """

        model_conf = conf.model
        model_name = model_conf.name
        model_params = model_conf.copy()
        del model_params['name']

        if model_name not in models_dict.keys():
            raise NotImplementedError

        model_class = models_dict[model_name]
        model = model_class(**model_params)

        return model

    def init_optimizer(self, conf: dict) -> optim.Optimizer:
        """Initialize optimizer using the config file. The optimizer choices 
        are available in the optimizers directory in optimizers_dict.py.

        :param conf: Configuration file
        :type conf: dict
        :return: Pytorch Optimizer
        :rtype: optim.Optimizer
        """

        opt_conf = conf.optimizer
        opt_name = opt_conf.name
        opt_params = opt_conf.copy()
        del opt_params['name']

        if opt_name not in optimizers_dict.keys():
            raise NotImplementedError

        opt_class = optimizers_dict[opt_name]
        opt = opt_class(self.model.parameters(), **opt_params)

        return opt

    def init_scheduler(self, conf: dict) -> optim.lr_scheduler._LRScheduler:
        """Initialize optimizer using the config file. The scheduler choices 
        are available in the optimizers directory in scheduler_dict.py.

        :param conf: Configuration file
        :type conf: dict
        :return: Pytorch learning rate scheduler
        :rtype: optim.lr_scheduler._LRScheduler
        """
        sched_conf = conf.sched
        sched_name = sched_conf.name
        sched_params = sched_conf.copy()
        del sched_params['name']

        if sched_name not in schedulers_dict.keys():
            raise NotImplementedError

        sched_class = schedulers_dict[sched_name]
        sched = sched_class(self.optimizer, **sched_params)
    
    def init_datasets(self, conf: dict) -> tuple[Dataset, Dataset, Dataset]:
        """Initialize train, val and test datasets, the type of the dataset 
        will be checked based on the type of module that is being run. For example 
        a classification problem should use a classification dataset.

        :param conf: Configuration file
        :type conf: dict
        :return: Pytorch Datasets, train, val and test
        :rtype: tuple[Dataset, Dataset, Dataset]
        """
        dataset_conf = conf.dataset
        dataset_name = dataset_conf.name

        train_conf = dataset_conf.train

        val_conf = None
        if 'val' in dataset_conf:
            val_conf = dataset_conf.val
        if 'val_split' in dataset_conf:
            val_conf = 'split_train'

        assert val_conf is not None, "Val configuration not properly specified."

        test_conf = dataset_conf.test

        dataset_class = datasets_dict[dataset_name]

        trainset = dataset_class(**train_conf)
        if val_conf == 'split_train':
            n = len(trainset)
            train_len = int(dataset_conf.val_split.p * n)
            val_len = n - train_len
            trainset, valset = random_split(trainset, [train_len, val_len])
        else:
            valset = dataset_class(**val_conf)
        
        testset = dataset_class(**test_conf)

        return trainset, valset, testset

    # @abstractmethod
    # def init_train_dataset(self, conf: dict) -> Dataset:
    #     """Initialize train dataset, the type of the dataset will be checked
    #     based on the type of module that is being run. For example a classification
    #     problem should use a classification dataset.

    #     :param conf: Configuration file
    #     :type conf: dict
    #     :return: Pytorch Dataset
    #     :rtype: Dataset
    #     """
        

    # @abstractmethod
    # def init_val_dataset(self, conf: dict) -> Dataset:
    #     """Initialize val dataset, the type of the dataset will be checked
    #     based on the type of module that is being run. For example a classification
    #     problem should use a classification dataset.

    #     :param conf: Configuration file
    #     :type conf: dict
    #     :return: Pytorch Dataset
    #     :rtype: Dataset
    #     """
    #     pass

    # @abstractmethod
    # def init_test_dataset(self, conf: dict) -> Dataset:
    #     """Initialize test dataset, the type of the dataset will be checked
    #     based on the type of module that is being run. For example a classification
    #     problem should use a classification dataset.

    #     :param conf: Configuration file
    #     :type conf: dict
    #     :return: Pytorch Dataset
    #     :rtype: Dataset
    #     """
    #     pass

    @abstractmethod
    def train(self) -> None:
        """This method runs one iteration of training, i.e. one full epoch. The method 
        changes the mode to train and sets the model to train. Subclasses will need 
        to override more functionality.
        """
        pass

    @abstractmethod
    def log_train_step(self) -> None:
        """Prints to terminal and/or logs data to mlflow after one iteration of training.
        """
        pass

    @abstractmethod
    def val(self) -> None:
        """This method runs one iteration of validation, i.e. one full epoch. The method 
        changes the mode to train and sets the model to train. Subclasses will need 
        to override more functionality.
        """
        pass

    @abstractmethod
    def log_val_step(self) -> None:
        """Prints to terminal and/or logs data to mlflow after one iteration of validation.
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """This method runs one iteration of testing, i.e. one full epoch. The method 
        changes the mode to train and sets the model to train. The class attribute 
        test_model must be set before testing during the val or train phase. Subclasses 
        will need to override more functionality.
        """
        assert self.test_model is not None, "No test model avaliable, set self.test_model before testing."
        pass

    @abstractmethod
    def log_test_step(self) -> None:
        """Prints to terminal and/or logs data to mlflow after one iteration of testing.
        """
        pass
