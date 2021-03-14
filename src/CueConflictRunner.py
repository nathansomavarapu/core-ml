from omegaconf import DictConfig
import hydra

from VisualClassificationRunner import VisualClassificationRunner

class CueConflictRunner(VisualClassificationRunner):      
        
    def test(self) -> dict:
        """Test the model on the standard dataset and the randomized datasets.

        :return: Dictionary with randomized test added
        :rtype: dict
        """
        self.module.testset.set_shape_or_texture('shape')
        shape_test_log = super().test()

        self.module.testset.set_shape_or_texture('texture')
        texture_test_log = super().test()

        return {'shape_bias': 100.0 * shape_test_log['correct'] / (shape_test_log['correct'] + texture_test_log['correct'])}
    
    def extract_test_log(self, log: dict) -> dict:

        out_log = {
            'correct': log['correct']
        }

        return out_log

@hydra.main(config_path="../conf", config_name="visual_classification.yaml")
def main(conf: DictConfig):
    runner = CueConflictRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()