from omegaconf import DictConfig
import hydra
import tqdm
import torch
import copy

from BaseRunner import BaseRunner
from modules.VisualClassificationModule import VisualClassificationModule

class VisualClassificationRunner(BaseRunner):        

    def setup_module(self, conf: DictConfig) -> VisualClassificationModule:
        """Initializes the visual classification module.

        :param conf: Config file
        :type conf: DictConfig
        :return: Visual classification module
        :rtype: VisualClassificationModule
        """
        return VisualClassificationModule(conf, self.device)
    
    def train(self) -> dict:
        self.module.model.train()

        total_log_dict: dict = {}
        train_batches = len(self.module.trainloader)

        train_pbar_iter = tqdm.tqdm(self.module.trainloader, disable=(not self.progress))

        for data in train_pbar_iter:
            data = self.move_data_to_device(data)
            loss, log_dict = self.module.forward_train(data)
            
            self.module.optimizer.zero_grad()
            loss.backward()
            self.module.optimizer.step()

            total_log_dict = self.update_dict(total_log_dict, log_dict)

            log_dict['acc'] = (log_dict['correct'] / log_dict['total']) * 100.0
            log_dict.pop('correct')
            log_dict.pop('total')

            if self.print_to_term:
                train_pbar_iter.set_description('Training: ' + self.generate_train_string(log_dict))

            if self.dry_run:
                break
        
        train_log = {
            'train_acc': (total_log_dict['correct'] / total_log_dict['total']) * 100.0,
            'train_loss': total_log_dict['loss'] / train_batches
        }

        self.log_train_step(train_log, step=self.e)
        
        return train_log
    
    def val(self) -> dict:
        self.module.model.eval()

        total_log_dict: dict = {}
        val_batches = len(self.module.valloader)

        with torch.no_grad():
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
        
        val_log = {
            'val_acc': (total_log_dict['correct'] / total_log_dict['total']) * 100.0,
            'val_loss': total_log_dict['loss'] / val_batches
        }

        self.log_val_step(val_log, step=self.e)

        self.update_test_model(val_log)
        
        return val_log
        
    def test(self) -> dict:
        self.module.test_model.eval()

        total_log_dict: dict = {}
        log_dict = {}
        test_batches = len(self.module.testloader)

        with torch.no_grad():
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

        test_log = {
            'test_acc': (total_log_dict['correct'] / total_log_dict['total']) * 100.0,
            'test_loss': total_log_dict['loss'] / test_batches
        }
        
        self.log_test_step(test_log, step=self.e)

        return test_log
    
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

@hydra.main(config_path="../conf", config_name="cifar10.yaml")
def main(conf: DictConfig):
    runner = VisualClassificationRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()