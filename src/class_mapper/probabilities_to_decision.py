# CREDIT: https://github.com/rgeirhos/texture-vs-shape

import torch
from abc import ABC, abstractmethod

import class_mapper.human_categories as hc


class ProbabilitiesToDecisionMapping(ABC):

    @abstractmethod
    def __call__(self, probabilities):
        pass


    def check_input(self, probabilities):
        """Run assert checks on probabilities.

        Keyword arguments:
        probabilities -- a np.ndarray of length 1000
                         (softmax output: all values should be
                         within [0,1])
        """
        assert type(probabilities) is torch.Tensor

class ImageNetProbabilitiesTo16ClassesMapping(ProbabilitiesToDecisionMapping):
    """Return the entry-level category decision for probabilities."""

    def __init__(self, aggregation_function=torch.mean):

        self.aggregation_function = aggregation_function


    def __call__(self, probabilities):
        """Return 16 category vector of probabilities.

        Keyword arguments:
        probabilities -- a torch tensor of length 1000, pre-softmax
        """
        self.check_input(probabilities)
        assert probabilities.size(-1) == 1000
        probabilities = torch.softmax(probabilities, -1)

        agg_prob = torch.zeros((probabilities.size(0), 16))
        c = hc.HumanCategories()
        for i, category in enumerate(hc.get_human_object_recognition_categories()):
            indices = c.get_imagenet_indices_for_category(category)
            values = probabilities[:,indices]
            aggregated_value = self.aggregation_function(values, dim=-1)
            agg_prob[:,i] = aggregated_value
   
        return agg_prob.to(probabilities.device)

