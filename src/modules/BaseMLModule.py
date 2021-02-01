from abc import ABC, abstractmethod
from typing import Any, Tuple
from omegaconf import DictConfig
import mlflow

from utils import Mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import Dataset, random_split, DataLoader

from models.models_dict import models_dict
from optimizers.optimizers_dict import optimizers_dict
from optimizers.schedulers_dict import schedulers_dict
from datasets.datasets_dict import datasets_dict
from loss_fn.loss_fn_dict import loss_fn_dict
from transforms.transforms_dict import transforms_dict


class BaseMLModule(ABC):
    """Base module which contains the basic attributes of a ml system
    it contains a model, optimizer, scheduler and dataset attributes.
    Abstract Class to define an interface for ml modules.

    :param ABC: Abstract Basic Class
    :type ABC: Class
    """

    def __init__(self, conf: DictConfig) -> None:
        """Initialize a BaseMLModule.

        :param conf: Configuration file
        :type conf: DictConfig
        """
        self.setup()
        self.mode = Mode.EMPTY

        self.device = self.setup_device()

        self.model = self.init_model(conf)
        self.test_model = None
        self.optimizer = self.init_optimizer(conf)
        self.scheduler = self.init_scheduler(conf)
        self.train_transform, self.test_transform = self.init_transforms(conf)
        self.trainset, self.valset, self.testset = self.init_datasets(conf)
        self.trainloader, self.valloader, self.testloader = self.init_dataloaders(conf)
        self.loss_fn = self.init_loss_fn(conf)

        self.config = conf
    
    def setup_device(self, conf: DictConfig) -> device:
        """Setup the device of the model.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch device
        :rtype: device
        """
        device = torch.device(conf.device) if torch.cuda.is_available() else torch.device('cpu')

        return device

    def setup(self) -> None:
        """Method to perform any setup needed before
        instantiating other class objects.
        """
        self.models_dict = models_dict
        self.optimizers_dict = optimizers_dict
        self.schedulers_dict = schedulers_dict
        self.datasets_dict = datasets_dict
        self.loss_fn_dict = loss_fn_dict
        self.transforms_dict = transforms_dict

    def init_model(self, conf: DictConfig) -> nn.Module:
        """Initialize model using the config file. The model choices 
        are available in the models directory in models_dict.py. This
        function sets up the model's device as well.

        :param conf: Configuration file
        :type conf: DictConfig
        :raises NotImplementedError: Raised when model name not supported
        :return: Pytorch Module
        :rtype: nn.Module
        """

        model_conf = conf.model
        model_name = model_conf.name
        model_params = model_conf.copy()
        del model_params['name']

        if model_name not in self.models_dict:
            raise NotImplementedError

        model_class = self.models_dict[model_name]
        model = model_class(**model_params)

        model.to(self.device)

        return model

    def init_optimizer(self, conf: DictConfig) -> optim.Optimizer:
        """Initialize optimizer using the config file. The optimizer choices 
        are available in the optimizers directory in optimizers_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Optimizer
        :rtype: optim.Optimizer
        """

        opt_conf = conf.optimizer
        opt_name = opt_conf.name
        opt_params = opt_conf.copy()
        del opt_params['name']

        if opt_name not in optimizers_dict:
            raise NotImplementedError

        opt_class = self.optimizers_dict[opt_name]
        opt = opt_class(self.model.parameters(), **opt_params)

        return opt
    
    def init_loss_fn(self, conf: DictConfig) -> nn.Module:
        """Initialize the loss function using the config file. The loss function
        choices are avaliable in the loss function directory in loss_fn_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Loss Function Module
        :rtype: nn.Module
        """
        loss_fn_conf = conf.loss_fn
        loss_fn_name = loss_fn_conf.name
        loss_fn_params = loss_fn_conf.copy()
        del loss_fn_params['name']

        if loss_fn_name not in self.loss_fn_dict:
            raise NotImplementedError
        
        loss_fn_class = self.loss_fn_dict[loss_fn_name]
        loss_fn = loss_fn_class(**loss_fn_params)

        return loss_fn

    def init_scheduler(self, conf: DictConfig) -> optim.lr_scheduler._LRScheduler:
        """Initialize scheduler using the config file. The scheduler choices 
        are available in the optimizers directory in scheduler_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch learning rate scheduler
        :rtype: optim.lr_scheduler._LRScheduler
        """
        if "sched" not in conf:
            return None
        
        sched_conf = conf.sched
        sched_name = sched_conf.name
        sched_params = sched_conf.copy()
        del sched_params['name']

        if sched_name not in self.schedulers_dict:
            raise NotImplementedError

        sched_class = self.schedulers_dict[sched_name]
        sched = sched_class(self.optimizer, **sched_params)

    def init_transforms(self, conf: DictConfig) -> Tuple[Any, Any]:
        """Initialize transform using the config file. The transform choices 
        are available in the transforms directory in transforms_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch image transform, this can be an object or a module
        :rtype: Any
        """
        transform_conf = conf.transforms
        
        train_params = transform_conf.train
        transform_train_name = train_params.name
        del train_params['name']

        test_params = transform_conf.test
        transform_test_name = test_params.name
        del test_params['name']

        if transform_train_name not in self.transforms_dict:
            raise NotImplementedError
        if transform_test_name not in self.transforms_dict:
            raise NotImplementedError
        
        transform_train_class = self.transforms_dict[transform_train_name]
        train_transforms = transform_train_class(**train_params)

        transform_test_class = self.transforms_dict[transform_test_name]
        test_transforms = transform_test_class(**test_params)

        return train_transforms, test_transforms
    
    def init_datasets(self, conf: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
        """Initialize train, val and test datasets, the type of the dataset 
        will be checked based on the type of module that is being run. For example 
        a classification problem should use a classification dataset.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: tuple[Dataset, Dataset, Dataset]
        """
        dataset_conf = conf.dataset
        dataset_name = dataset_conf.name

        if dataset_name not in self.datasets_dict:
            raise NotImplementedError

        train_conf = dataset_conf.train

        assert not ('val' in dataset_conf and 'val_split' in dataset_conf), "Either val or val_split should be specified not both."
        val_conf = None
        if 'val' in dataset_conf:
            val_conf = dataset_conf.val
        if 'val_split' in dataset_conf:
            val_conf = 'split_train'

        assert val_conf is not None, "Val configuration not properly specified."

        test_conf = dataset_conf.test

        dataset_class = self.datasets_dict[dataset_name]

        trainset = dataset_class(**train_conf, transform=self.train_transform)
        if val_conf == 'split_train':
            n = len(trainset)
            val_len = int(dataset_conf.val_split * n)
            train_len = n - val_len
            trainset, valset = random_split(trainset, [train_len, val_len])
        else:
            valset = dataset_class(**val_conf, transform=self.test_transform)
        
        testset = dataset_class(**test_conf, transform=self.test_transform)

        return trainset, valset, testset
    
    def init_dataloaders(self, conf: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Initialize train, val and test dataloaders.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: tuple[DataLoader, DataLoader, DataLoader]
        """
        dataloader_conf = conf.dataloader
        train_dataloader_conf = dataloader_conf.train
        val_dataloader_conf = dataloader_conf.val
        test_dataloader_conf = dataloader_conf.test

        trainloader = DataLoader(self.trainset, **train_dataloader_conf)
        valloader = DataLoader(self.valset, **val_dataloader_conf)
        testloader = DataLoader(self.testset, **test_dataloader_conf)

        return trainloader, valloader, testloader

    @abstractmethod
    def train(self) -> None:
        """This method runs one iteration of training, i.e. one full epoch. Subclasses will need 
        to override more functionality.
        """
        pass

    def print_train_step(self) -> None:
        """Prints to terminal after one iteration of training.
        """
        print_str = 'Epoch {},'.format(self.e)
        for k,v in self.train_log.items():
            print_str = print_str + '{} : {:.4f}, \t'.format(k, v)
            
        print(print_str)

    def log_train_step(self) -> None:
        """Logs data to mlflow after one iteration of training.
        """
        mlflow.log_metrics(self.train_log)

    @abstractmethod
    def val(self) -> None:
        """This method runs one iteration of validation, i.e. one full epoch. Subclasses will need 
        to override more functionality.
        """
        pass

    def print_val_step(self) -> None:
        """Prints to terminal after one iteration of validation.
        """
        print_str = 'Epoch {},'.format(self.e)
        for k,v in self.val_log.items():
            print_str = print_str + '{} : {:.4f}, \t'.format(k, v)
            
        print(print_str)

    def log_val_step(self) -> None:
        """Logs data to mlflow after one iteration of validation.
        """
        mlflow.log_metrics(self.val_log)

    @abstractmethod
    def test(self) -> None:
        """This method runs one iteration of testing, i.e. one full epoch. The class attribute 
        test_model must be set before testing during the val or train phase. Subclasses 
        will need to override more functionality.
        """
        assert self.test_model is not None, "No test model avaliable, set self.test_model before testing."
        pass

    def print_test_step(self) -> None:
        """Prints to terminal after one iteration of testing.
        """
        print_str = 'Epoch {},'.format(self.e)
        for k,v in self.test_log.items():
            print_str = print_str + '{} : {:.4f}, \t'.format(k, v)
            
        print(print_str)

    def log_test_step(self) -> None:
        """Logs data to mlflow after one iteration of testing.
        """
        mlflow.log_metrics(self.test_log)
