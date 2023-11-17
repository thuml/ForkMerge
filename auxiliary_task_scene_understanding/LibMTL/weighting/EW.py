import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstract_weighting import AbsWeighting


class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """

    def __init__(self):
        super(EW, self).__init__()

    def backward(self, losses, **kwargs):
        weights = torch.FloatTensor(kwargs['weights']).to(self.device)
        if weights.shape[0] != losses.shape[0]:
            weights = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, weights).to(self.device).sum()
        loss.backward()
        return np.ones(self.task_num)
