from omegaconf import DictConfig
import hydra

from BaseRunner import BaseRunner
from modules.VisualClassificationModule import VisualClassificationModule

class VisualClassificationRunner(BaseRunner):

    def setup_module(self, conf: DictConfig):
        return VisualClassificationModule(conf)

@hydra.main(config_path="../conf", config_name="cifar10.yaml")
def main(conf: DictConfig):
    runner = VisualClassificationRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()