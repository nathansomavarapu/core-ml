from abc import ABC, abstractmethod
from typing import Any, Tuple
from omegaconf import DictConfig
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import Dataset, random_split, DataLoader


class BaseMLModule(ABC):
    """Base module which contains the basic attributes of a ml system
    it contains a model, optimizer, scheduler and dataset attributes.
    Abstract Class to define an interface for ml modules.

    :param ABC: Abstract Basic Class
    :type ABC: Class
    """

    def __init__(self, conf: DictConfig, device: device) -> None:
        """Initialize a BaseMLModule.

        :param conf: Configuration file
        :type conf: DictConfig
        :param device: Pytorch device
        :type device: device
        """
        attrs = self.setup()
        for k,v in attrs.items():
            setattr(self, k, v)
        
        self.device = device

        self.model = self.init_model(conf)
        self.test_model = None
        self.optimizer = self.init_optimizer(conf)
        self.scheduler = self.init_scheduler(conf)
        self.train_transform, self.test_transform = self.init_transforms(conf)
        self.trainset, self.valset, self.testset = self.init_datasets(conf)
        self.trainloader, self.valloader, self.testloader = self.init_dataloaders(conf)
        self.loss_fn = self.init_loss_fn(conf)
                
        self.config = conf
    
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
        model_load_path = model_conf.load_path if 'load_path' in model_conf else None

        model_params = dict(model_conf)

        if model_load_path:
            del model_params['load_path']

        if model_name not in self.models_dict:
            raise NotImplementedError

        model_class = self.models_dict[model_name]
        model = model_class(**model_params)

        if model_load_path:
            model.load_state_dict(torch.load(model_load_path, map_location='cpu'))

        model.to(self.device)

        return model

    def init_optimizer(self, conf: DictConfig) -> optim.Optimizer:
        """Initialize optimizer using the config file. The optimizer choices 
        are available in the optimizers directory in optimizers_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :raises NotImplementedError: Raised when optimizer name not supported
        :return: Pytorch Optimizer
        :rtype: optim.Optimizer
        """
        opt_conf = conf.optimizer
        opt_name = opt_conf.name

        opt_params = dict(opt_conf)
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

        loss_fn_params = dict(loss_fn_conf)
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
        if "scheduler" not in conf:
            return None
        
        sched_conf = conf.scheduler
        sched_name = sched_conf.name

        sched_params = dict(sched_conf)
        del sched_params['name']

        if sched_name not in self.schedulers_dict:
            raise NotImplementedError

        sched_class = self.schedulers_dict[sched_name]
        sched = sched_class(self.optimizer, **sched_params)

        return sched

    def init_transforms(self, conf: DictConfig) -> Tuple[Any, Any]:
        """Initialize transform using the config file. The transform choices 
        are available in the transforms directory in transforms_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch image transform, this can be an object or a module
        :rtype: Any
        """
        transform_conf = conf.transforms
        
        transform_train_name = transform_conf.train.name
        train_params = dict(transform_conf.train)
        del train_params['name']

        transform_test_name = transform_conf.test.name
        test_params = dict(transform_conf.test)
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
    
    def init_generic_dataset(self, conf: DictConfig, mode: str) -> Tuple[Dataset, Dict]:
        """Generic function that sets up a dataset. The code to setup
        a basic dataset is the same across train val and test. 

        :param conf: Configuration file
        :type conf: DictConfig
        :param mode: Dataset mode, train, val or test
        :type mode: str
        :return: Tuple of un-instantiated dataset class and a dictionary
        containing arguments for the dataset
        :rtype: Tuple[Dataset, Dict]
        """
        mode_conf = conf[mode]
        name = mode_conf.name if 'name' in mode_conf else conf['name']

        if name not in self.datasets_dict:
            raise NotImplementedError
        
        dataset_class = self.dataset_dict[name]
        dataset_conf = dict(mode_conf)
        dataset_conf.pop('name', None)

        return dataset_class, dataset_conf
    
    def init_trainset(self, conf: DictConfig) -> Dataset:
        """Initialize train dataset based on the configuration file. In the
        base module this function is called by the function init_datasets so
        as to enable spliting the training dataset into training and validation
        datasets on the fly.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, train
        :rtype: Dataset
        """
        trainset_class, train_conf = self.init_generic_dataset(conf.dataset, 'train')
        trainset = trainset_class(**train_conf, transform=self.train_transform)
        return trainset
    
    def init_valset(self, conf: DictConfig) -> Dataset:
        """Initialize val dataset based on the configuration file. In the
        base module this function is called by the function init_datasets so
        as to enable spliting the training dataset into training and validation
        datasets on the fly.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, val
        :rtype: Dataset
        """
        valset_class, val_conf = self.init_generic_dataset(conf.dataset, 'val')
        valset = valset_class(**val_conf, transform=self.test_transform)
        return valset
    
    def init_testset(self, conf: DictConfig) -> Dataset:
        """Initialize test dataset based on the configuration file. In the
        base module this function is called by the function init_datasets so
        as to enable spliting the training dataset into training and validation
        datasets on the fly. The test code is extracted for symmetry and extensibility.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Dataset, test
        :rtype: Dataset
        """
        testset_class, test_conf = self.init_generic_dataset(conf.dataset, 'test')
        testset = testset_class(**test_conf, transform=self.test_transform)
        return testset
    
    def init_datasets(self, conf: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
        """Initialize train, val and test datasets, the type of the dataset 
        will be checked based on the type of module that is being run. For example 
        a classification problem should use a classification dataset.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: tuple[Dataset, Dataset, Dataset]
        """
        trainset = self.init_trainset(conf)

        dataset_conf = conf.dataset
        val = dataset_conf.val
        valset = None
        if val:
            val_split = dataset_conf.val.split if 'split' in dataset_conf.val else None
            
            if val_split:
                n = len(trainset)
                val_len = int(val_split * n)
                train_len = n - val_len
                trainset, valset = random_split(trainset, [train_len, val_len])
            else:
                valset = self.init_valset(conf)
        
        testset = self.init_testset(conf)

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
        valloader = DataLoader(self.valset, **val_dataloader_conf) if self.valset is not None else None
        testloader = DataLoader(self.testset, **test_dataloader_conf)

        return trainloader, valloader, testloader
    
    def scheduler_step(self) -> None:
        """Run one step of the scheduler, meant to be called once per epoch.
        """
        if self.scheduler:
            self.scheduler.step()
    
    def print_val_step(self) -> None:
        """Prints to terminal after one iteration of validation.
        """
        print_str = 'Epoch {}, '.format(self.e)
        for k,v in self.val_log.items():
            print_str = print_str + '{} : {:.4f}, '.format(k, v)
            
        print(print_str[:-1])
    
    def print_test_step(self) -> None:
        """Prints to terminal after one iteration of testing.
        """
        print_str = 'Epoch {}, '.format(self.e)
        for k,v in self.test_log.items():
            print_str = print_str + '{} : {:.4f}, '.format(k, v)
            
        print(print_str[:-1])
    
    @abstractmethod
    def setup(self) -> dict:
        """Method to perform any setup needed before instantiating other 
        class objects. Any variables returned will be assigned to the class. 
        All of the dictionaries used for component initialization should be 
        set this way. This method must be overridden by  subclasses.

        :return: dictionary containing class variables to be set to the
        name specified by the key
        :rtype: dict
        """
    
    @abstractmethod
    def forward_train(self, data: Tuple) -> Any:
        """This method runs one iteration of training. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Any
        """
        pass

    @abstractmethod
    def forward_val(self, data: Tuple) -> Any:
        """This method runs one iteration of validation. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Any
        """
        pass

    @abstractmethod
    def forward_test(self, data: Tuple) -> Any:
        """This method runs one iteration of testing. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Any
        """
        pass
