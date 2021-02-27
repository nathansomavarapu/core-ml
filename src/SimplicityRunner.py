from omegaconf import DictConfig
import hydra
import tqdm
import torch
import copy

from VisualClassificationRunner import VisualClassificationRunner

class SimplicityRunner(VisualClassificationRunner):        
        
    def test(self) -> dict:
        """Test the model on the standard dataset and the randomized datasets.

        :return: Dictionary with randomized test added
        :rtype: dict
        """
        self.module.testset.set_randomized_test(None)
        test_log = super().test()

        self.module.testset.set_randomized_test('simple')
        random_test_log_simple = super().test()
        random_test_log_simple = {'randomized_simple_' + k: v for k,v in random_test_log_simple.items()}

        self.module.testset.set_randomized_test('complex')
        random_test_log_complex = super().test()
        random_test_log_complex = {'randomized_complex_' + k: v for k,v in random_test_log_complex.items()}

        joined_log = {**test_log,**random_test_log_simple,**random_test_log_complex}
        out_log = {k:v for k,v in joined_log.items() if 'loss' not in k}

        return out_log


@hydra.main(config_path="../conf", config_name="simplicity.yaml")
def main(conf: DictConfig):
    runner = SimplicityRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()