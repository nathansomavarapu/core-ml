from omegaconf import DictConfig
import hydra
import tqdm
import torch
import copy
from typing import Tuple, Callable, Any

from BaseRunner import BaseRunner
from modules.VisualClassificationModule import VisualClassificationModule

class VisualClassificationRunner(BaseRunner):  

    def __init__(self, conf: DictConfig) -> None:
        super(VisualClassificationRunner, self).__init__(conf)

        self.test_grad = conf.runner.test_grad if 'test_grad' in conf.runner else False
        self.val_grad = conf.runner.val_grad if 'val_grad' in conf.runner else False

    def setup_module(self, conf: DictConfig) -> VisualClassificationModule:
        """Initializes the visual classification module.

        :param conf: Config file
        :type conf: DictConfig
        :return: Visual classification module
        :rtype: VisualClassificationModule
        """
        return VisualClassificationModule(conf, self.device)
    
    def train(self) -> Any:
        self.module.model.train()

        total_log_dict: dict = {}
        train_batches = len(self.module.trainloader)

        train_pbar_iter = tqdm.tqdm(self.module.trainloader, disable=(not self.progress))

        for data in train_pbar_iter:
            data = self.move_data_to_device(data)
            self.module.optimizer.zero_grad()
            log_dict = self.module.forward_train(data)
            self.module.optimizer.step()

            total_log_dict = self.update_dict(total_log_dict, log_dict)

            log_dict['acc'] = (log_dict['correct'] / log_dict['total']) * 100.0
            log_dict.pop('correct')
            log_dict.pop('total')

            if self.print_to_term:
                train_pbar_iter.set_description('Training: ' + self.generate_train_string(log_dict))

            if self.dry_run:
                break
        
        total_log_dict['train_batches'] = train_batches
        train_log = self.extract_train_log(total_log_dict)

        self.log_train_step(train_log, step=self.e)
        
        return train_log
    
    def val(self) -> Any:
        self.module.model.eval()

        total_log_dict: dict = {}
        val_batches = len(self.module.valloader)

        with torch.set_grad_enabled(self.val_grad):
            val_pbar_iter = tqdm.tqdm(self.module.valloader, disable=(not self.progress))
            for data in val_pbar_iter:
                data = self.move_data_to_device(data)
                log_dict = self.module.forward_val(data)

                total_log_dict = self.update_dict(total_log_dict, log_dict)

                log_dict['acc'] = (log_dict['correct'] / log_dict['total']) * 100.0
                log_dict.pop('correct')
                log_dict.pop('total')

                if self.print_to_term:
                    val_pbar_iter.set_description('Validation: ' + self.generate_test_string(log_dict))

                if self.dry_run:
                    break
        
        total_log_dict['val_batches'] = val_batches
        val_log = self.extract_val_log(total_log_dict)

        self.log_val_step(val_log, step=self.e)

        self.update_test_model(val_log)
        
        return val_log
        
    def test(self) -> Any:
        self.module.test_model.eval()

        total_log_dict: dict = {}
        log_dict = {}
        test_batches = len(self.module.testloader)

        with torch.set_grad_enabled(self.test_grad):
            test_pbar_iter = tqdm.tqdm(self.module.testloader, disable=(not self.progress))
            for data in test_pbar_iter:
                data = self.move_data_to_device(data)
                log_dict = self.module.forward_test(data)

                total_log_dict = self.update_dict(total_log_dict, log_dict)

                log_dict['acc'] = (log_dict['correct'] / log_dict['total']) * 100.0
                log_dict.pop('correct')
                log_dict.pop('total')

                if self.print_to_term:
                    test_pbar_iter.set_description('Testing: ' + self.generate_test_string(log_dict))

                if self.dry_run:
                    break

        total_log_dict['test_batches'] = test_batches
        test_log = self.extract_test_log(total_log_dict)
        
        self.log_test_step(test_log, step=self.e)

        return test_log
    
    def extract_generic_log(self, log: dict, mode: str) -> dict:
        """Extract generic log.

        :param log: Accumulated log
        :type log: dict
        :param mode: train, val or test used to prepend to metrics
        :type mode: str
        :return: Dictionary with relevant metrics
        :rtype: dict
        """
        metric_log = {
            mode + '_acc': (log['correct'] / log['total']) * 100.0,
            mode + '_loss': log['loss'] / log[mode + '_batches']
        }
        return metric_log
    
    def extract_train_log(self, log: dict) -> dict:
        """Extract relevant statistics from train log. 

        :param log: Accumulated train log
        :type log: dict
        :return: Dictionary with relevant metrics
        :rtype: dict
        """
        return self.extract_generic_log(log, 'train')
    
    def extract_val_log(self, log: dict) -> dict:
        """Extract relevant statistics from train log. 

        :param log: Accumulated train log
        :type log: dict
        :return: Dictionary with relevant metrics
        :rtype: dict
        """
        return self.extract_generic_log(log, 'val')
    
    def extract_test_log(self, log: dict) -> dict:
        """Extract relevant statistics from train log. 

        :param log: Accumulated train log
        :type log: dict
        :return: Dictionary with relevant metrics
        :rtype: dict
        """
        return self.extract_generic_log(log, 'test')
    
    def main(self) -> None:
        self.setup()

        epoch_pbar_iter = tqdm.tqdm(range(self.epochs), disable=(not self.progress))
        epoch_stats = ''
        for e in epoch_pbar_iter:
            self.e = e

            train_log = self.train()

            epoch_stats = self.generate_train_string(train_log, step=e)

            if self.do_val:
                val_log = self.val()
                epoch_stats += ' ' + self.generate_val_string(val_log)
            else:
                self.module.test_model = self.module.model
            
            self.module.scheduler_step()

            if self.test_every_epoch:
                test_log = self.test()
                epoch_stats += ' ' + self.generate_test_string(test_log)
            
            if self.print_to_term:
                epoch_pbar_iter.set_description(epoch_stats)
        
        test_log = self.test()
        if self.print_to_term:
            print(self.generate_test_string(test_log))

        self.teardown()

@hydra.main(config_path="../conf", config_name="visual_classification.yaml")
def main(conf: DictConfig):
    runner = VisualClassificationRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()