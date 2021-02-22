from modules.VisualClassificationModule import VisualClassificationModule

from datasets.datasets_dict import dg_datasets_dict

class DGModule(VisualClassificationModule):

    def setup(self) -> dict:
        """Overrides parent setup to replace the previous dataset dictionary
        with the dg dataset dictionary.

        :return: Dictionary of class attributes to be set
        :rtype: dict
        """
        attrs = super().setup()
        attrs['datasets_dict'] = dg_datasets_dict

        return attrs

    def init_generic_dataset(self, conf: DictConfig, mode: str) -> Tuple[Dataset, Dict]:
        """Overrides the parent dataset initialization function to take the dataset name
        and target dataset config values in differently.

        :param conf: Configuration file
        :type conf: DictConfig
        :param mode: train, val or test
        :type mode: str
        :return: Pytorch dataset and configuration dictionary
        :rtype: Tuple[Dataset, Dict]
        """
        dataset_class, dataset_conf = super().init_generic_dataset(conf, mode)
        dataset_conf['target'] = conf['target']

        return dataset_class, dataset_conf