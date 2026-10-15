import torch
from torch import nn


class IDSNet(nn.Module):
    """MLP classifier over flow/packet features.

    LayerNorm instead of BatchNorm: BatchNorm's running statistics get
    averaged across non-IID clients and the global model drifts.
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
