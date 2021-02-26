from omegaconf import DictConfig
import hydra
import tqdm
import torch
import copy

from VisualClassificationRunner import VisualClassificationRunner
from modules.DGModule import DGModule

class DGRunner(VisualClassificationRunner):

    def setup_module(self, conf: DictConfig) -> DGModule:
        """Initializes the DG classification module.

        :param conf: Config file
        :type conf: DictConfig
        :return: DG classification module
        :rtype: DGModule
        """
        return DGModule(conf, self.device)

@hydra.main(config_path="../conf", config_name="dg.yaml")
def main(conf: DictConfig):
    runner = DGRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()