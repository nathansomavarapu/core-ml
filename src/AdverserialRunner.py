from omegaconf import DictConfig
import hydra

from VisualClassificationRunner import VisualClassificationRunner
from modules.AdversarialModule import AdversarialModule

class AdversarialRunner(VisualClassificationRunner):

    def setup_module(self, conf: DictConfig) -> AdversarialModule:
        """Initializes the visual classification module.

        :param conf: Config file
        :type conf: DictConfig
        :return: Adversarial module
        :rtype: AdversarialModule
        """
        return AdversarialModule(conf, self.device)

    def train(self) -> None:
        None

    def val(self) -> None:
        None
    
    def main(self) -> None:
        self.setup()

        test_log = self.test()
        if self.print_to_term:
            print(self.generate_test_string(test_log))
        
        self.teardown()
    
    def extract_test_log(self, log: dict) -> dict:
        """Extract relevant statistics from train log. 

        :param log: Accumulated train log
        :type log: dict
        :return: Dictionary with relevant metrics
        :rtype: dict
        """
        test_log = self.extract_generic_log(log, 'test')
        test_log['adv_acc'] = (log['correct_adv'] / log['total']) * 100.0

        return test_log

@hydra.main(config_path="../conf", config_name="adv.yaml")
def main(conf: DictConfig):
    runner = AdversarialRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()