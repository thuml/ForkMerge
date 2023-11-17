import torch
import random
import numpy as np

from .abstract_weighting import AbsWeighting


class PriGradVac(AbsWeighting):

    def __init__(self):
        super(PriGradVac, self).__init__()
        self.print_interval = 100
        self.cnt = 0

    def init_param(self):
        self.rho_T = torch.zeros(self.task_num, self.task_num).to(self.device)

    def backward(self, losses, **kwargs):
        pri_idx = kwargs['pri_idx']
        beta = kwargs['beta']

        if self.rep_grad:
            raise ValueError('No support method GradVac with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]

        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            if tn_i == pri_idx:
                continue
            rho_ij = torch.dot(pc_grads[tn_i], grads[pri_idx]) / (pc_grads[tn_i].norm() * grads[pri_idx].norm())
            if rho_ij < self.rho_T[tn_i, pri_idx]:
                w = pc_grads[tn_i].norm() * (self.rho_T[tn_i, pri_idx] * (1 - rho_ij ** 2).sqrt() - rho_ij * (
                        1 - self.rho_T[tn_i, pri_idx] ** 2).sqrt()) / (
                            grads[pri_idx].norm() * (1 - self.rho_T[tn_i, pri_idx] ** 2).sqrt())
                pc_grads[tn_i] += grads[pri_idx] * w
                batch_weight[pri_idx] += w.item()
            # Fix bug of LibMTL
            self.rho_T[tn_i, pri_idx] = (1 - beta) * self.rho_T[tn_i, pri_idx] + beta * rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)

        if self.cnt % self.print_interval == 0:
            print(batch_weight)
        self.cnt += 1
        return batch_weight
