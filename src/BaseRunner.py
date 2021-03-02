import os
import mlflow
import torch
import copy
import tqdm
import numpy as np
import random

from torch import device

from omegaconf import DictConfig
import hydra.utils as hutils
from abc import ABC, abstractmethod
from modules.BaseMLModule import BaseMLModule
from typing import Union, Tuple

class BaseRunner(ABC):
    """Abstract class for running modules, this takes care of logging
    and the high level flow of running the system.

    :param ABC: Abstract Basic Class
    :type ABC: Class
    """

    def __init__(self, conf: DictConfig) -> None:
        """Initializes the runner with options from the configuration
        file.

        :param conf: Configuration file
        :type conf: DictConfig
        """
        self.seed = self.init_seed_cudnn(conf)
        self.device = self.setup_device(conf)
        self.epochs = conf.runner.epochs if 'epochs' in conf.runner else 0

        self.progress, self.print_to_term, self.log_mlflow = self.init_output_options(conf)

        self.module = self.setup_module(conf)

        self.val_acc = float('-inf')
        self.do_val = bool(conf.runner.val)

        self.test_every_epoch = conf.runner.test_every_epoch if 'test_every_epoch' in conf.runner else False
        self.dry_run = conf.runner.dry_run if 'dry_run' in conf.runner else False
        if self.dry_run:
            self.epochs = 1

        self.config = conf
        self.e = 0 # Set for linter, updated over multiple iterations
    
    def setup_device(self, conf: DictConfig) -> device:
        """Setup the device of the model.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Pytorch device
        :rtype: device
        """
        device = torch.device(conf.runner.device) if torch.cuda.is_available() else torch.device('cpu')

        return device
    
    def init_output_options(self, conf: DictConfig) -> Tuple[bool, bool, bool]:
        """Initialize options for printing progress bar, printing to terminal and logging to mlflow.

        :param conf: [description]
        :type conf: DictConfig
        :return: Outputs bool for progress bar, print to terminal and logging to mlflow respectively
        :rtype: Tuple[bool, bool, bool]
        """
        runner_conf = conf.runner
        progress = runner_conf.progress if 'progress' in runner_conf else False
        print_to_term = runner_conf.print_to_term if 'print_to_term' in runner_conf else False
        log_mlflow = runner_conf.log_mlflow if 'log_mlflow' in runner_conf else False

        return progress, print_to_term, log_mlflow
    
    def init_seed_cudnn(self, conf: DictConfig) -> Union[int, None]:
        """Initialize runner seed, sets seed for torch, numpy and random along with
        cudnn parameters.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: seed value, None if no seed
        :rtype: Union[int, None]
        """
        runner_conf = conf.runner
        seed = runner_conf.seed if 'seed' in runner_conf else None
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            # torch.set_deterministic(True)
        # torch.backends.cudnn.benchmark = False
        
        return seed
    
    def setup(self) -> None:
        """Setup function called before main starts, a teardown
        function is called at the end of the run function.

        :param conf: [description]
        :type conf: DictConfig
        """
        mlflow.set_tracking_uri('file://' + hutils.get_original_cwd() + '/mlruns')
        if self.log_mlflow:
            mlflow.set_experiment(self.config.runner.exp_name)
        
        if self.log_mlflow:
            self.log_parameters(self.config)
            mlflow.log_param('node', os.uname()[1])
    
    def move_data_to_device(self, data: Tuple) -> Tuple:
        """Moves the data tensors to the device.

        :param data: Data tensors
        :type data: Tuple
        :return: Data tensors moved to device
        :rtype: Tuple
        """
        tmp = []
        for dv in data:
            tmp.append(dv.to(self.device))
        
        return tuple(tmp)
    
    def teardown(self) -> None:
        """Teardown function called after runner ends, meant to
        cleanup setup function.
        """
        pass
    
    def log_parameters(self, conf: DictConfig, parent_name: str = '') -> None:
        """Logs parameters to mlflow using the configuration file.

        :param conf: Configuration file
        :type conf: DictConfig
        """
        for k,v in conf.items():
            param_name = parent_name + '.' + k if parent_name else k
            if isinstance(v, DictConfig):
                self.log_parameters(v, parent_name=param_name)
            else:
                mlflow.log_param(param_name, v)

    def generate_train_string(self, train_log: dict, step: Union[int,None] = None) -> str:
        """Generates a string to print train metrics.

        :param train_log: Dictionary of train metrics
        :type train_log: dict
        :param step: Step to log, defaults to None
        :type step: Union[int,None], optional
        :return: String generated from log
        :rtype: str
        """
        print_str = 'Epoch {}, '.format(step) if step else ''
        for k,v in train_log.items():
            print_str = print_str + '{} : {:.4f}, '.format(k, v)
            
        return print_str[:-1]

    def generate_val_string(self, val_log: dict, step: Union[int,None] = None) -> str:
        """Generates a string to print val metrics.

        :param train_log: Dictionary of train metrics
        :type train_log: dict
        :param step: Step to log, defaults to None
        :type step: Union[int,None], optional
        :return: String generated from log
        :rtype: str
        """
        print_str = 'Epoch {}, '.format(step) if step else ''
        for k,v in val_log.items():
            print_str = print_str + '{} : {:.4f}, '.format(k, v)
            
        return print_str[:-1]

    def generate_test_string(self, test_log: dict, step: Union[int,None] = None) -> str:
        """Generates a string to print val metrics.

        :param train_log: Dictionary of train metrics
        :type train_log: dict
        :param step: Step to log, defaults to None
        :type step: Union[int,None], optional
        :return: String generated from log
        :rtype: str
        """
        print_str = 'Epoch {}, '.format(step) if step else ''
        for k,v in test_log.items():
            print_str = print_str + '{} : {:.4f}, '.format(k, v)
            
        return print_str[:-1]
    
    def log_train_step(self, train_log: dict, step: Union[int,None] = None) -> None:
        """Logs train metric to mlflow. Keys in dictionary are the 
        parameters to be logged the values are the values to be logged.

        :param train_log: Training log with key, value pairs to be logged
        :type train_log: dict
        :param step: Step/iteration of log, defaults to None
        :type step: Union[int,None], optional
        """
        if self.log_mlflow:
            mlflow.log_metrics(train_log, step=step)
    
    def log_val_step(self, val_log: dict, step: Union[int,None] = None) -> None:
        """Logs val metric to mlflow. Keys in dictionary are the 
        parameters to be logged the values are the values to be logged.

        :param train_log: Validation log with key, value pairs to be logged
        :type train_log: dict
        :param step: Step/iteration of log, defaults to None
        :type step: Union[int,None], optional
        """
        if self.log_mlflow:
            mlflow.log_metrics(val_log, step=self.e)
    
    def log_test_step(self, test_log: dict, step: Union[int,None] = None) -> None:
        """Logs test metric to mlflow. Keys in dictionary are the 
        parameters to be logged the values are the values to be logged.

        :param train_log: Testing log with key, value pairs to be logged
        :type train_log: dict
        :param step: Step/iteration of log, defaults to None
        :type step: Union[int,None], optional
        """
        if self.log_mlflow:
            mlflow.log_metrics(test_log, step=self.e)
    
    def update_dict(self, total_dict: dict, iter_dict: dict) -> dict:
        """Updates a dictionary with metrics from a single iteration.

        :param total_dict: Dictionary which updated to incorporate the iteration
        metric
        :type total_dict: dict
        :param iter_dict: Metrics from one iteration of training, validation or testing
        :type iter_dict: dict
        :return: Updated dictionary
        :rtype: dict
        """
        for k in iter_dict:
            if k not in total_dict:
                total_dict[k] = iter_dict[k]
            else:
                total_dict[k] += iter_dict[k]
        
        return total_dict
    
    def update_test_model(self, val_log: dict) -> None:
        """Update the model to be used for testing
        based on current val loss or accuracy. This method is responsible for
        model selection.

        :param val_log: Log from one iteration of validation.
        :type val_log: dict
        """
        if val_log['val_acc'] >= self.val_acc:
            self.module.test_model = copy.deepcopy(self.module.model)
            self.val_acc = val_log['val_acc']

            self.module.test_model.save_model()

    @abstractmethod
    def setup_module(self, conf: DictConfig) -> BaseMLModule:
        """Initialize module to be used as part of run logic.
        Abstract method to be implemented.

        :param conf: Configuration file
        :type conf: DictConfig
        :return: Module which is to be used in the runner
        :rtype: BaseMLModule
        """
        pass
    
    @abstractmethod
    def train(self) -> dict:
        """Run one iteration of training, this is an abstract method and needs to
        be defined by the child class. This method also contains all logging logic.

        :return: Dictionary used for logging
        :rtype: dict
        """
        pass

    @abstractmethod
    def val(self) -> dict:
        """Run one iteration of validation, this is an abstract method and needs to
        be defined by the child class. This method also contains all logging logic.

        :return: Dictionary used for logging
        :rtype: dict
        """
        pass
    
    @abstractmethod
    def test(self) -> dict:
        """Run one iteration of testing, this is an abstract method and needs to
        be defined by the child class. This method also contains all logging logic.

        :return: Dictionary used for logging
        :rtype: dict
        """
        pass

    @abstractmethod
    def main(self) -> None:
        """Main runner, this is the function that should be called by
        by main.
        """
        pass
    