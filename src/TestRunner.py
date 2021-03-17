from omegaconf import DictConfig
import hydra

from VisualClassificationRunner import VisualClassificationRunner

class TestRunner(VisualClassificationRunner):

    def main(self) -> None:
        self.setup()

        test_log = self.test()
        if self.print_to_term:
            print(self.generate_test_string(test_log))
        
        self.teardown()

@hydra.main(config_path="../conf", config_name="visual_classification.yaml")
def main(conf: DictConfig):
    assert 'load_path' in conf.model
    runner = TestRunner(conf)
    runner.main()

if __name__ == '__main__':
    main()
