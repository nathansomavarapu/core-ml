# NOTE inter/intra-source stylization is not available for domainnet currently.
stylized_dataset_fp_dict = {
    'painting': {
        'pacs': '/srv/share3/nsomavarapu3/datasets/DG/kfold_stylized',
        'vlcs': '/srv/share3/nsomavarapu3/datasets/DG/VLCS_stylized',
        'oh': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split_stylized/',
        'domainnet': '/srv/share3/nsomavarapu3/datasets/DG/domainnet_stylized/'
    },
    'inter_source': {
        'pacs': '/srv/share3/nsomavarapu3/datasets/DG/kfold_stylized_pair',
        'vlcs': '/srv/share3/nsomavarapu3/datasets/DG/VLCS_stylized_pair',
        'oh': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split_stylize_pair/'
        # 'domainnet': ''
    },
    'intra_source': {
        'pacs': '/srv/share3/nsomavarapu3/datasets/DG/kfold_stylized_pair',
        'vlcs': '/srv/share3/nsomavarapu3/datasets/DG/VLCS_stylized_pair',
        'oh': '/srv/share3/nsomavarapu3/datasets/DG/OfficeHomeDataset_split_stylize_pair/'
        # 'domainnet': ''
    }
}