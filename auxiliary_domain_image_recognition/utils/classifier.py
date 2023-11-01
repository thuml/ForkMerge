import torch.nn as nn
import torch
from typing import Optional, List, Dict


class MultiOutputClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1,
                 finetune=True, pool_layer=None):
        super(MultiOutputClassifier, self).__init__()
        self.backbone = backbone
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        self.heads = heads
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor, dataset_name):
        """"""
        f = self.backbone(x)
        f = self.pool_layer(f)
        f = self.bottleneck(f)
        return self.heads[dataset_name](f)

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.heads.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
