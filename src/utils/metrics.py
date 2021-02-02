import torch

def correct(preds, labels, n=1):
    
    _, predicted = preds.detach().topk(n, 1)
    
    labels = labels.view(-1, 1).expand_as(predicted)
    correct = (predicted[:,:n] == labels).sum().sum().item()
    return correct