import torch.optim as optim

schedulers_dict = {
    'step': optim.lr_scheduler.StepLR,
    'cosine': optim.lr_scheduler.CosineAnnealingLR
}
