from datasets.DGDatasets import PACSDataset, VLCSDataset, OHDataset, DomainNetDataset

dg_datasets_dict = {
    'pacs': PACSDataset,
    'vlcs': VLCSDataset,
    'oh': OHDataset,
    'domainnet': DomainNetDataset
}