import os
import mlflow
import torch
import copy
import tqdm

from omegaconf import DictConfig
import hydra.utils as hutils
from abc import ABC, abstractmethod
from modules.BaseMLModule import BaseMLModule

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
        runner_conf = conf.runner
        self.epochs = runner_conf.epochs

        self.progress = runner_conf.progress if 'progress' in runner_conf else False
        self.print_to_term = runner_conf.print_to_term if 'print_to_term' in runner_conf else False
        self.log_mlflow = runner_conf.log_mlflow if 'log_mlflow' in runner_conf else False
        
        self.module = self.setup_module(conf)
        
        self.config = conf
    
    def setup(self) -> None:
        """Setup function called before runner starts, a teardown
        function is called at the end of the run function.

        :param conf: [description]
        :type conf: DictConfig
        """
        mlflow.set_tracking_uri('file://' + hutils.get_original_cwd() + '/mlruns')
        if self.log_mlflow:
            mlflow.set_experiment(self.config.runner.exp_name)
    
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
    
    def train(self) -> None:
        """Run the module's train method and put model in train mode.
        """
        self.module.model.train()
        self.module.train()

        if self.log_mlflow:
            self.module.log_train_step()
        if self.print_to_term:
            self.module.print_train_step()

    def val(self) -> None:
        """Run the module's validation method and put model in eval mode
        and turn off gradients. Save the model to the test model for testing 
        based on val acc.
        """
        with torch.no_grad():
            self.module.model.eval()
            self.module.val()
        
        if self.log_mlflow:
            self.module.log_val_step()
        if self.print_to_term:
            self.module.print_val_step()
        
        if not hasattr(self.module, 'val_acc') or self.module.val_log['val_acc'] >= self.module.val_acc:
            self.test_model = copy.deepcopy(self.module.model)
            self.module.val_acc = self.module.val_log['val_acc']
    
    def test(self) -> None:
        """Run the module's validation method and put model in eval mode
        and turn off gradients. Save the model to the test model for testing 
        based on val acc.
        """
        with torch.no_grad():
            self.module.test_model.eval()
            self.module.test()
        
        if self.log_mlflow:
            self.module.log_test_step()
        if self.print_to_term:
            self.module.print_test_step()

    def main(self) -> None:
        """Main runner, this is the function that should be called by
        by main.
        """
        self.setup()
        if self.log_mlflow:
            self.log_parameters(self.config)
            mlflow.log_param('node', os.uname()[1])
        
        for e in tqdm.tqdm(range(self.epochs), disable=(not self.progress)):
            self.module.e = e

            self.train()
            self.val()
        
        self.test()

        self.teardown()
    
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