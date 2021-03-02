import torch.optim as optim
import schedulers.CustomSchedulers as CustomSchedulers

schedulers_dict = {
    'step': optim.lr_scheduler.StepLR,
    'cosine': optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warmup': CustomSchedulers.WarmupCosineSchedule
}
