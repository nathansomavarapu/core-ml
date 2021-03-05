from omegaconf import DictConfig
import hydra
import tqdm
import torch
import copy
from typing import Tuple, Callable
from transform_inverters.inverter_dict import inverter_dict

from BaseRunner import BaseRunner
from modules.VisualClassificationModule import VisualClassificationModule
from utils.conf_utils import remove_internal_conf_params
from torchvision.utils import save_image

class VisualClassificationRunner(BaseRunner):  

    def __init__(self, conf: DictConfig) -> None:
        self.inverter_dict = self.setup_dictionaries()

        super(VisualClassificationRunner, self).__init__(conf)

        self.save_imgs = None if 'save_images' not in conf.runner else conf.runner.save_images
        self.im_transform_inverter = self.init_im_transform_inverter(conf)
    
    def setup_dictionaries(self) -> dict:
        """Setup inverter dictionary.

        :return: Inverter Dictionary
        :rtype: dict
        """
        return inverter_dict

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
                
            if self.save_imgs:
                self.save_images(data)

        test_log = {
            'test_acc': (total_log_dict['correct'] / total_log_dict['total']) * 100.0,
            'test_loss': total_log_dict['loss'] / test_batches
        }
        
        self.log_test_step(test_log, step=self.e)

        return test_log
    
    def save_images(self, data: Tuple[torch.Tensor,...], epoch: int = None) -> None:
        """Takes in a batch samples and saves a number of images if the runner
        conf variable runner.save_images = <num_images> is set.

        :param data: Tuple of image data and labels
        :type data: Tuple[torch.Tensor,...]
        :param epoch: Epoch when images were saved, defaults to None
        :type epoch: int, optional
        """
        images, _ = data
        images = images[:self.save_imgs]

        with torch.no_grad():
            pred = self.module.test_model(images).cpu()
        _, labels = pred.max(dim=1)

        images = images.cpu()

        label_to_save = [str(x.item()) for x in labels] # type: ignore
        label_str = '_'.join(label_to_save)

        if self.im_transform_inverter:
            im_to_save = self.im_transform_inverter(images)
        
        save_image(im_to_save, label_str + '.png')
    
    def init_im_transform_inverter(self, conf: DictConfig) -> Callable:
        """Initialize an optional componenet that will invert image
        transformations before saving images.

        :param conf: [description]
        :type conf: DictConfig
        :return: [description]
        :rtype: Callable
        """
        inverter_conf = conf.runner.inverter
        inv_name = inverter_conf._name

        if inv_name not in self.inverter_dict:
            raise NotImplementedError
        
        inverter_conf = dict(inverter_conf)
        inv_params = remove_internal_conf_params(inverter_conf)
        inverter_class = self.inverter_dict[inv_name]
        
        return inverter_class(**inv_params)
    
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