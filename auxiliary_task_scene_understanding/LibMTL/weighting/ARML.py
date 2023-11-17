import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstract_weighting import AbsWeighting


class ARML(AbsWeighting):

    def __init__(self):
        super(ARML, self).__init__()
        self.print_interval = 20
        self.cnt = 0

    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([1.0] * self.task_num, device=self.device))

    def backward(self, losses, **kwargs):
        pri_idx = kwargs['pri_idx']

        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

        main_task_grad = grads[pri_idx]
        aux_task_grad = None
        for i in range(self.task_num):
            if i != pri_idx:
                if aux_task_grad is None:
                    aux_task_grad = self.loss_scale[i] * grads[i]
                else:
                    aux_task_grad = aux_task_grad + self.loss_scale[i] * grads[i]
        L_grad = ((main_task_grad - aux_task_grad) ** 2).sum()
        L_grad.backward()

        # self.loss_scale.data = self.loss_scale.data + (self.task_num - self.loss_scale.data.sum()) / self.task_num
        self.loss_scale.data[pri_idx] = 1.
        loss_weight = self.loss_scale.detach().clone()

        if self.rep_grad:
            self._backward_new_grads(loss_weight, per_grads=per_grads)
        else:
            self._backward_new_grads(loss_weight, grads=grads)

        if self.cnt % self.print_interval == 0:
            print(loss_weight.cpu().numpy())
        self.cnt += 1
        return loss_weight.cpu().numpy()
