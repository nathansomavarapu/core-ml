from omegaconf import DictConfig
import hydra

from modules.InvModule import InvModule
from VisualClassificationRunner import VisualClassificationRunner

class InvRunner(VisualClassificationRunner):

    def setup_module(self, conf: DictConfig) -> InvModule:
        """Returns Invariance Module.

        :param conf: Configuration
        :type conf: DictConfig
        :return: Invariance Module
        :rtype: InvModule
        """
        return InvModule(conf, self.device)
    
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
            mode + '_acc': (log['correct'] / log['total']) * 100.0
        }
        return metric_log

    def main(self) -> None:
        self.setup()

        test_log = self.test()
        if self.print_to_term:
            print(self.generate_test_string(test_log))
        
        self.teardown()

@hydra.main(config_path="../conf", config_name="visual_classification.yaml")
def main(conf: DictConfig):
    assert 'load_path' in conf.model
    runner = InvRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()
