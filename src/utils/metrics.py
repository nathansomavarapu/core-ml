import torch

def get_class_prediction(preds: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Take a tensor of probabilistic predictions and return the predicted class
    for each sample.

    :param preds: Tensor of [Batch Size, C] with the predicted probability of each class
    per sample 
    :type preds: Torch.Tensor
    :param n: [Top n to consider when getting predicted class, defaults to 1
    :type n: int, optional
    :return: Predicted class based on probabilities per class
    :rtype: torch.Tensor
    """
    _, predicted = preds.detach().topk(n, 1)
    return predicted

def correct_indices(preds: torch.Tensor, labels: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Compute the number of correctly predicted samples from the preds
    tensor using the correct labels.

    :param preds: Torch tensor of size [batch_size, num_classes] with logits
    :type preds: torch.Tensor
    :param labels: Torch tensor of size [batch_size] with the correct class
    for each sample
    :type labels: torch.Tensor
    :param n: n parameter for topn computation, defaults to 1
    :type n: int, optional
    :return: Torch tensor which records which samples the model predicted correctly,
    the tensor is of size [batch_size, 1]
    :rtype: torch.Tensor
    """
    predicted = get_class_prediction(preds, n=n)
    
    labels = labels.view(-1, 1).expand_as(predicted)
    correct = predicted[:,:n] == labels
    return correct.squeeze()

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
    correct_idxs = correct_indices(preds, labels, n=n)

    return correct_idxs.sum().item() # type: ignore