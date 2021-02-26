import torch

def correct(preds: torch.Tensor, labels: torch.Tensor, n: int = 1) -> int:
    """Compute the number of correctly predicted samples from the preds
    tensor using the correct labels.

    :param preds: Torch tensor of size [batch_size, num_classes] with logits
    :type preds: torch.Tensor
    :param labels: Torch tensor of size [batch_size] with the correct class
    for each sample
    :type labels: torch.Tensor
    :param n: n parameter for topn computation, defaults to 1
    :type n: int, optional
    :return: Number of correctly predicted samples
    :rtype: int
    """
    _, predicted = preds.detach().topk(n, 1)
    
    labels = labels.view(-1, 1).expand_as(predicted)
    correct = (predicted[:,:n] == labels).sum().sum().item()
    return correct