from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Callable
from omegaconf import DictConfig
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from utils import conf_utils

class BaseMLModule(ABC):
    """Base module which contains the basic attributes of a ml system
    it contains a model, optimizer, scheduler and dataset attributes.
    Abstract Class to define an interface for ml modules.

    The module uses a number of dictionaries for bookeeping to instantiate
    components of the module. The required dictionaries are:

    * models_dict
    * optimizers_dict
    * schedulers_dict
    * datasets_dict
    * loss_fn_dict
    * transforms_dict

    :param ABC: Abstract Basic Class
    :type ABC: Class
    """

    def __init__(self, conf: DictConfig, device: torch.device) -> None:
        """Initialize a BaseMLModule.

        :param conf: Configuration file
        :type conf: DictConfig
        :param device: Pytorch device
        :type torch.device: device
        """
        attrs = self.setup()
        self.models_dict = attrs['models_dict'] if 'models_dict' in attrs else None
        self.optimizers_dict = attrs['optimizers_dict'] if 'optimizers_dict' in attrs else None
        self.schedulers_dict = attrs['schedulers_dict'] if 'schedulers_dict' in  attrs else None
        self.datasets_dict = attrs['datasets_dict'] if 'datasets_dict' in  attrs else None
        self.loss_fn_dict = attrs['loss_fn_dict'] if 'loss_fn_dict' in  attrs else None
        self.transforms_dict = attrs['transforms_dict'] if 'transforms_dict' in  attrs else None
        self.class_mapper_dict = attrs['class_mapper_dict'] if 'class_mapper_dict' in attrs else None
        
        self.device = device

        self.class_mapper = self.init_class_mapper(conf) if self.class_mapper_dict else None
        self.model = self.init_model(conf) if self.models_dict else None
        self.test_model = self.model
        self.optimizer = self.init_optimizer(conf) if self.optimizers_dict else None
        self.scheduler = self.init_scheduler(conf) if self.schedulers_dict else None

        self.train_transform = self.test_transform = None
        if self.datasets_dict:
            self.train_transform, self.test_transform = self.init_transforms(conf)
                
        self.trainset = self.valset = self.testset = None
        if self.datasets_dict:
            self.trainset, self.valset, self.testset = self.init_datasets(conf)
                        
        self.trainloader = self.valloader = self.testloader = None
        if self.datasets_dict:
            self.trainloader, self.valloader, self.testloader = self.init_dataloaders(conf) #type: ignore
        
        self.loss_fn = self.init_loss_fn(conf) if self.loss_fn_dict else None
                
        self.config = conf
        self.e = 0  # NOTE: Set by runner added here to fix linting
        self.val_log: dict = {} # NOTE: Set by runner added here to fix linting
        self.test_log: dict = {} # NOTE: Set by runner added here to fix linting
    
    def init_class_mapper(self, conf: DictConfig) -> Optional[Callable]:
        """Initialize a class mapper which takes classes from a model
        and maps them to a subset for testing. This is used for example
        in the cue-conflict experiments.

        :param conf: Configuration
        :type conf: DictConfig
        :return: Class mapper object
        :rtype: Optional[Callable]
        """

        if 'class_mapper' not in conf:
            return None

        cmapper_conf = conf.class_mapper
        cmapper_name = cmapper_conf._name
        
        if cmapper_name not in self.class_mapper_dict:
            raise NotImplementedError

        mapper_cl = self.class_mapper_dict[cmapper_name]
        cmapper_params = dict(cmapper_conf)
        cmapper_params = self.remove_internal_conf_params(cmapper_params)

        return mapper_cl(**cmapper_params)

    def init_model(self, conf: DictConfig) -> Optional[nn.Module]:
        """Initialize model using the config file. The model choices 
        are available in the models directory in models_dict.py. This
        function sets up the model's device as well.

        :param conf: Configuration file
        :type conf: DictConfig
        :raises NotImplementedError: Raised when model name not supported
        :return: Pytorch Module
        :rtype: Optional[nn.Module]
        """
        if "model" not in conf:
            return None

        model_conf = conf.model
        model_name = model_conf._name

        model_params = dict(model_conf)

        if model_name not in self.models_dict:
            raise NotImplementedError

        model_class = self.models_dict[model_name]
        model_params = self.remove_internal_conf_params(model_params)
        model = model_class(class_mapper=self.class_mapper, **model_params)

        if '_parallel' in model_conf and model_conf._parallel:
            model = nn.DataParallel(model)
        
        model.to(self.device)

        return model

    def init_optimizer(self, conf: DictConfig) -> Optional[optim.Optimizer]:
        """Initialize optimizer using the config file. The optimizer choices 
        are available in the optimizers directory in optimizers_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :raises NotImplementedError: Raised when optimizer name not supported
        :return: Pytorch Optimizer
        :rtype: Optional[optim.Optimizer]
        """
        if "optimizer" not in conf:
            return None
        
        opt_conf = conf.optimizer
        opt_name = opt_conf._name

        opt_params = dict(opt_conf)
        opt_params = self.remove_internal_conf_params(opt_params)

        if opt_name not in self.optimizers_dict:
            raise NotImplementedError

        opt_class = self.optimizers_dict[opt_name]
        opt = opt_class(self.model.parameters(), **opt_params) # type: ignore

        return opt
    
    def init_loss_fn(self, conf: DictConfig) -> Optional[nn.Module]:
        """Initialize the loss function using the config file. The loss function
        choices are avaliable in the loss function directory in loss_fn_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Loss Function Module
        :rtype: Optional[nn.Module]
        """
        if 'loss_fn' not in conf:
            return None
        
        loss_fn_conf = conf.loss_fn
        loss_fn_name = loss_fn_conf._name

        loss_fn_params = dict(loss_fn_conf)
        loss_fn_params = self.remove_internal_conf_params(loss_fn_params)

        if loss_fn_name not in self.loss_fn_dict:
            raise NotImplementedError
        
        loss_fn_class = self.loss_fn_dict[loss_fn_name]
        loss_fn = loss_fn_class(**loss_fn_params)

        return loss_fn

    def init_scheduler(self, conf: DictConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Initialize scheduler using the config file. The scheduler choices 
        are available in the optimizers directory in scheduler_dict.py.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch learning rate scheduler
        :rtype: Union[optim.lr_scheduler._LRScheduler, None]
        """
        if "scheduler" not in conf:
            return None
        
        sched_conf = conf.scheduler
        sched_name = sched_conf._name

        sched_params = dict(sched_conf)
        sched_params = self.remove_internal_conf_params(sched_params)

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
        if 'transforms' not in conf:
            return None, None
        
        transform_conf = conf.transforms

        train_transforms = None
        if "train" in transform_conf:
        
            transform_train_name = transform_conf.train._name
            train_params = dict(transform_conf.train)
            train_params = self.remove_internal_conf_params(train_params)

            if transform_train_name not in self.transforms_dict:
                raise NotImplementedError
            
            transform_train_class = self.transforms_dict[transform_train_name]
            train_transforms = transform_train_class(**train_params)
        
        test_transforms = None
        if 'test' in transform_conf:

            transform_test_name = transform_conf.test._name if 'test' in transform_conf else None
            test_params = dict(transform_conf.test)
            test_params = self.remove_internal_conf_params(test_params)

        
            if transform_test_name not in self.transforms_dict:
                raise NotImplementedError
        
            transform_test_class = self.transforms_dict[transform_test_name]
            test_transforms = transform_test_class(**test_params)

        return train_transforms, test_transforms
    
    def init_generic_dataset(self, conf: DictConfig, mode: str) -> Tuple[Dataset, dict]:
        """Generic function that sets up a dataset. The code to setup
        a basic dataset is the same across train val and test.

        :param conf: Configuration file at the point of the dataset
        config, i.e. conf.dataset is passed in
        :type conf: DictConfig
        :param mode: Dataset mode, train, val or test
        :type mode: str
        :return: Tuple of un-instantiated dataset class and a dictionary
        containing arguments for the dataset
        :rtype: Tuple[Dataset, Dict]
        """
        mode_conf = conf[mode]
        name = mode_conf._name if '_name' in mode_conf else conf['_name']

        if name not in self.datasets_dict:
            raise NotImplementedError
        
        dataset_class = self.datasets_dict[name]
        dataset_conf = dict(mode_conf)
        dataset_conf = self.remove_internal_conf_params(dataset_conf)

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
        trainset_class, train_conf = self.init_generic_dataset(conf, 'train')
        trainset = trainset_class(**train_conf, transform=self.train_transform) # type: ignore
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
        valset_class, val_conf = self.init_generic_dataset(conf, 'val')
        valset = valset_class(**val_conf, transform=self.test_transform) # type: ignore
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
        testset_class, test_conf = self.init_generic_dataset(conf, 'test')
        testset = testset_class(**test_conf, transform=self.test_transform) # type: ignore
        return testset
    
    def init_datasets(self, conf: DictConfig) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Initialize train, val and test datasets, the type of the dataset 
        will be checked based on the type of module that is being run. For example 
        a classification problem should use a classification dataset.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: Tuple[Dataset, Union[Dataset, None], Dataset]
        """
        dataset_conf = conf.dataset

        train = dataset_conf.train if 'train' in dataset_conf else None
        trainset = None
        if train:
            trainset = self.init_trainset(dataset_conf)
        
        val = dataset_conf.val if 'val' in dataset_conf else None
        valset = None
        if val:
            valset = self.init_valset(dataset_conf) # type: ignore
        
        test = dataset_conf.test if 'test' in dataset_conf else None
        testset = None
        if test:
            testset = self.init_testset(dataset_conf)

        return trainset, valset, testset
    
    def init_dataloaders(self, conf: DictConfig) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """Initialize train, val and test dataloaders.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: Tuple[Dataset, Union[Dataset, None], Dataset]
        """
        dataloader_conf = conf.dataloader
        train_dataloader_conf = dataloader_conf.train
        val_dataloader_conf = dataloader_conf.val
        test_dataloader_conf = dataloader_conf.test

        trainloader = DataLoader(self.trainset, **train_dataloader_conf) if self.trainset else None # type: ignore
        valloader = DataLoader(self.valset, **val_dataloader_conf) if self.valset else None # type: ignore
        testloader = DataLoader(self.testset, **test_dataloader_conf) if self.testset else None # type: ignore

        return trainloader, valloader, testloader # type: ignore
    
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
    
    def remove_internal_conf_params(self, conf: dict) -> dict:
        """Remove all parameters with an underscore preceding them,
        calls a utility function to do this in order to share functionality.

        :param conf: Configuration
        :type conf: DictConfig
        :return: Dict with internal parameters removed
        :rtype: dict
        """
        return conf_utils.remove_internal_conf_params(conf)
    
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
    def forward_train(self, data: Tuple) -> Optional[Any]:
        """This method runs one iteration of training. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Optional[Any]
        """
        pass

    @abstractmethod
    def forward_val(self, data: Tuple) -> Optional[Any]:
        """This method runs one iteration of validation. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Optional[Any]
        """
        pass

    @abstractmethod
    def forward_test(self, data: Tuple) -> Optional[Any]:
        """This method runs one iteration of testing. Subclasses will need 
        to override functionality

        :param data: Data tuple of inputs to module
        :type data: Tuple
        :return: Tensor and logs used for optimization
        :rtype: Optional[Any]
        """
        pass
