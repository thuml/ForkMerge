import torch
import numpy as np

from .abstract_weighting import AbsWeighting


class OLAUX(AbsWeighting):

    def __init__(self):
        super(OLAUX, self).__init__()
        self.print_interval = 100
        self.cnt = 0

        # for simplicity use fixed task_num 3
        self.weights = torch.ones(3)
        self.task_accumulate_cos_sim = {}
        for i in range(3):
            self.task_accumulate_cos_sim[i] = 0
        self.update_interval = 5

    def backward(self, losses, **kwargs):
        beta = kwargs['beta']
        pri_idx = kwargs['pri_idx']
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]
        pc_grads = grads.clone()

        for tn_i in range(self.task_num):
            if tn_i == pri_idx:
                continue
            cos_sim = torch.dot(pc_grads[tn_i], pc_grads[pri_idx]) / pc_grads[tn_i].norm() / pc_grads[pri_idx].norm()
            self.task_accumulate_cos_sim[tn_i] += (beta * cos_sim).detach().cpu().item()

        if (self.cnt + 1) % self.update_interval == 0:
            for i in range(3):
                self.weights[i] += self.task_accumulate_cos_sim[i]
                self.task_accumulate_cos_sim[i] = 0

        self._backward_new_grads(self.weights.to(losses.device), grads=grads)
        if self.cnt % self.print_interval == 0:
            print(self.weights.numpy())
        self.cnt += 1
        return self.weights.numpy()
