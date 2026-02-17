from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import timm


class TimmEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet18", *, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.out_dim = int(getattr(self.model, "num_features", 0))
        if self.out_dim <= 0:
            raise RuntimeError(f"Could not infer feature dim for timm model: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet18Encoder(TimmEncoder):
    def __init__(self, *, pretrained: bool = False):
        super().__init__("resnet18", pretrained=pretrained)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2, bn_last: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            self.net = nn.Linear(in_dim, out_dim, bias=True)
            return

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=not bn_last))
        if bn_last:
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@dataclass(frozen=True)
class SwAVConfig:
    feat_dim: int = 128
    n_prototypes: int = 3000
    temperature: float = 0.1
    epsilon: float = 0.05
    sinkhorn_iterations: int = 3
    crops_for_assign: tuple[int, ...] = (0, 1)
