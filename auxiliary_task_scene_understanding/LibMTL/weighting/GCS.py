import numpy as np
import torch

from .abstract_weighting import AbsWeighting


class GCS(AbsWeighting):

    def __init__(self):
        super(GCS, self).__init__()

    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        pri_idx = np.where(np.array(kwargs['pri_tasks']) == 1)[0][0]
        if self.rep_grad:
            raise ValueError('No support method GCS with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]
        gcs_grads = grads.clone()
        for i in range(self.task_num):
            if i == pri_idx:
                # primary task
                batch_weight[i] = 1
                gcs_grads[i] = grads[i]
            else:
                # auxiliary task
                cos_sim = torch.dot(gcs_grads[i], grads[pri_idx]) / (
                            torch.norm(gcs_grads[i]) * torch.norm(grads[pri_idx]))
                if kwargs['gcs_method'] == 'unweighted':
                    gcs_grads[i] = gcs_grads[i] * (cos_sim > 0).float()
                    batch_weight[i] = (cos_sim > 0).float().item()
                else:
                    gcs_grads[i] = gcs_grads[i] * max(0, cos_sim)
                    batch_weight[i] = max(0, cos_sim.item())
        new_grads = gcs_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
