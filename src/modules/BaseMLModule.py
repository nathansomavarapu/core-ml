from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from omegaconf import DictConfig
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader

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
        self.models_dict = attrs['models_dict']
        self.optimizers_dict = attrs['optimizers_dict']
        self.schedulers_dict = attrs['schedulers_dict']
        self.datasets_dict = attrs['datasets_dict']
        self.loss_fn_dict = attrs['loss_fn_dict']
        self.transforms_dict = attrs['transforms_dict']
        
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
        self.e = 0  # NOTE: Set by runner added here to fix linting
        self.val_log: dict = {} # NOTE: Set by runner added here to fix linting
        self.test_log: dict = {} # NOTE: Set by runner added here to fix linting
    
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
        model_name = model_conf._name

        model_params = dict(model_conf)

        if model_name not in self.models_dict:
            raise NotImplementedError

        model_class = self.models_dict[model_name]
        model_params = self.remove_internal_conf_params(model_params)
        model = model_class(**model_params)

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
        opt_name = opt_conf._name

        opt_params = dict(opt_conf)
        opt_params = self.remove_internal_conf_params(opt_params)

        if opt_name not in self.optimizers_dict:
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
        transform_conf = conf.transforms
        
        transform_train_name = transform_conf.train._name
        train_params = dict(transform_conf.train)
        train_params = self.remove_internal_conf_params(train_params)

        transform_test_name = transform_conf.test._name
        test_params = dict(transform_conf.test)
        test_params = self.remove_internal_conf_params(test_params)

        if transform_train_name not in self.transforms_dict:
            raise NotImplementedError
        if transform_test_name not in self.transforms_dict:
            raise NotImplementedError
        
        transform_train_class = self.transforms_dict[transform_train_name]
        train_transforms = transform_train_class(**train_params)

        transform_test_class = self.transforms_dict[transform_test_name]
        test_transforms = transform_test_class(**test_params)

        return train_transforms, test_transforms
    
    def init_generic_dataset(self, conf: DictConfig, mode: str) -> Tuple[Dataset, dict]:
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
        conf = conf.dataset
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
    
    def init_datasets(self, conf: DictConfig) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        """Initialize train, val and test datasets, the type of the dataset 
        will be checked based on the type of module that is being run. For example 
        a classification problem should use a classification dataset.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch Datasets, train, val and test
        :rtype: Tuple[Dataset, Union[Dataset, None], Dataset]
        """
        trainset = self.init_trainset(conf)

        dataset_conf = conf.dataset
        val = dataset_conf.val
        valset = None
        if val:
            valset = self.init_valset(conf) # type: ignore
        
        testset = self.init_testset(conf)

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

        trainloader = DataLoader(self.trainset, **train_dataloader_conf)
        valloader = DataLoader(self.valset, **val_dataloader_conf) if self.valset is not None else None
        testloader = DataLoader(self.testset, **test_dataloader_conf)

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
        """Remove all parameters from a dictionary with _ in front
        of the name. These are used to index into internal dictionaries for
        different module options.

        :param conf: Configuration dictionary
        :type conf: dict
        :return: Configuration dictionary with internal params removed
        :rtype: dict
        """
        if len(conf) == 0:
            return conf
        
        delete_list = []
        for k,v in conf.items():
            if k[0] == '_':
                delete_list.append(k)
            elif isinstance(conf[k], dict):
                conf[k] = self.remove_internal_conf_params(conf[k])
        
        for k in delete_list:
            del conf[k]
        
        return conf
    
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
