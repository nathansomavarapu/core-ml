from datasets.FileListDataset import FileListDataset
from datasets.ValSplitDataset import ValSplitDataset

from typing import Callable, Optional
from datasets.mnistm_paths_dict import mnistm_paths_dict, mnistm_roots_dict

class MNISTM(ValSplitDataset):

    def __init__(self, mode: str = 'train', transform: Optional[Callable] = None, split=0.2, **kwargs) -> None:
        """Initialize MNISTM dataset.

        :param mode: Dataset mode must be train, val or test, defaults to 'train'
        :type mode: str, optional
        :param transform: Image tranforms, defaults to None
        :type transform: Optional[Callable], optional
        :param split: Validation set split, defaults to 0.2
        :type split: float, optional
        """
        dataset = FileListDataset(mnistm_roots_dict[mode], mnistm_paths_dict[mode], class_idx_offset=0, **kwargs)
        super(MNISTM, self).__init__(dataset, mode=mode, split=split, transform=transform)