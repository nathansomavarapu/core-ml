dg_root_dict = {
    'pacs': '/srv/share3/nsomavarapu3/datasets/DG/kfold/',
    'domainnet': '/srv/share3/nsomavarapu3/datasets/DG/domainnet/'
}

dg_path_dict = {
    'pacs': {
        'photo': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_train_kfold.txt',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_crossval_kfold.txt',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/kfold/photo_test_kfold.txt'
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
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/caltech/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/caltech/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/caltech/full/'
        },
        'labelme': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/labelme/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/labelme/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/labelme/full/'
        },
        'pascal': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/pascal/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/pascal/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/pascal/full/'
        },
        'sun': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/sun/train/',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/sun/crossval/',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/VLCS/sun/full/'
        }
    },

    'oh': {
        'art': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/art/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/art/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/art/full'
        },
        'clipart': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/clipart/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/clipart/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/clipart/full'
        },
        'product': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/product/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/product/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/product/full'
        },
        'real': {
            'train': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/real/train',
            'val': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/real/test',
            'test': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split/real/full'
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