import torch
import numpy as np

from .abstract_weighting import AbsWeighting


class PriPCGrad(AbsWeighting):

    def __init__(self):
        super(PriPCGrad, self).__init__()
        self.print_interval = 100
        self.cnt = 0

    def backward(self, losses, **kwargs):
        pri_idx = kwargs['pri_idx']
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]
        pc_grads = grads.clone()

        for tn_i in range(self.task_num):
            if tn_i == pri_idx:
                continue
            g = torch.dot(pc_grads[tn_i], pc_grads[pri_idx])
            if g < 0:
                pc_grads[tn_i] -= g * pc_grads[pri_idx] / (pc_grads[pri_idx].norm().pow(2))
                batch_weight[pri_idx] -= (g / (pc_grads[pri_idx].norm().pow(2))).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        if self.cnt % self.print_interval == 0:
            print(batch_weight)
        self.cnt += 1
        return batch_weight
