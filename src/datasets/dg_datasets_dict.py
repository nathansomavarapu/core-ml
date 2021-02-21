from DGDatasets import PACSDataset, VLCSDataset, OHDataset, DomainNetDataset

dg_root_dict = {
    'pacs': '/srv/share3/nsomavarapu3/datasets/DG/kfold/',
    'domainnet': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/'
}

dg_path_dict = {
    'pacs': {
        'photo': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_train_kfold.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_crossval_kfold.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_painting_test_kfold.txt'
            },
        'art_painting': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/kfold/art_painting_train_kfold.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/kfold/art_painting_crossval_kfold.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/kfold/art_painting_test_kfold.txt'
            },
        'cartoon': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/kfold/cartoon_train_kfold.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/kfold/cartoon_crossval_kfold.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/kfold/cartoon_test_kfold.txt'
        },
        'sketch': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/kfold/sketch_train_kfold.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/kfold/sketch_crossval_kfold.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/kfold/sketch_test_kfold.txt'
        }
    },

    'vlcs': {
        'caltech': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/CALTECH/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/CALTECH/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/CALTECH/full/'
        },
        'labelme': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/LABELME/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/LABELME/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/LABELME/full/'
        },
        'pascal': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/PASCAL/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/PASCAL/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/PASCAL/full/'
        },
        'sun': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/SUN/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/SUN/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/SUN/full/'
        }
    },

    'oh': {
        'art': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Art/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Art/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Art/full'
        },
        'clipart': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Clipart/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Clipart/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Clipart/full'
        },
        'product': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Product/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Product/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Product/full'
        },
        'real': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Real/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Real/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/Real/full'
        }
    },

    'domainnet': {
        'clipart': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/clipart_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/clipart_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/clipart_test.txt'
        },
        'infograph': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/infograph_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/infograph_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/infograph_test.txt'
        },
        'painting': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/painting_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/painting_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/painting_test.txt'
        },
        'quickdraw': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/quickdraw_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/quickdraw_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/quickdraw_test.txt'
        },
        'real': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/real_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/real_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/real_test.txt'
        },
        'sketch': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/sketch_train.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/sketch_val.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/sketch_test.txt'
        }
    }
}

dg_dataset_dict = {
    'pacs': PACSDataset,
    'vlcs': VLCSDataset,
    'oh': OHDataset,
    'domainnet': DomainNetDataset
}