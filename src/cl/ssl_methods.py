from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl_models import MLP, SwAVConfig, l2_normalize


class SSLMethod(nn.Module):
    def __init__(self, *, encoder: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(feature_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SimCLR(SSLMethod):
    def __init__(self, *, encoder: nn.Module, feature_dim: int, proj_dim: int = 128, hidden_dim: int = 2048, temperature: float = 0.2):
        super().__init__(encoder=encoder, feature_dim=feature_dim)
        self.projector = MLP(feature_dim, hidden_dim, proj_dim, num_layers=2)
        self.temperature = float(temperature)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = views
        z1 = l2_normalize(self.projector(self.encoder(x1)))
        z2 = l2_normalize(self.projector(self.encoder(x2)))
        z = torch.cat([z1, z2], dim=0)

        sim = z @ z.t()
        sim.fill_diagonal_(-9e15)

        b = z1.size(0)
        labels = torch.arange(b, device=z.device)
        labels = torch.cat([labels + b, labels], dim=0)
        logits = sim / self.temperature
        return F.cross_entropy(logits, labels)


class VICReg(SSLMethod):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        feature_dim: int,
        proj_dim: int = 8192,
        hidden_dim: int = 8192,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
    ):
        super().__init__(encoder=encoder, feature_dim=feature_dim)
        self.projector = MLP(feature_dim, hidden_dim, proj_dim, num_layers=3)
        self.sim_coeff = float(sim_coeff)
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = views
        x = self.projector(self.encoder(x))
        y = self.projector(self.encoder(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = (F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean()) / 2

        cov_x = (x.t() @ x) / (x.size(0) - 1)
        cov_y = (y.t() @ y) / (y.size(0) - 1)
        cov_loss = (self._off_diag(cov_x).pow(2).sum() + self._off_diag(cov_y).pow(2).sum()) / x.size(1)

        return self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

    @staticmethod
    def _off_diag(m: torch.Tensor) -> torch.Tensor:
        n = m.size(0)
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(SSLMethod):
    def __init__(self, *, encoder: nn.Module, feature_dim: int, proj_dim: int = 8192, hidden_dim: int = 8192, lambd: float = 0.005):
        super().__init__(encoder=encoder, feature_dim=feature_dim)
        self.projector = MLP(feature_dim, hidden_dim, proj_dim, num_layers=3)
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)
        self.lambd = float(lambd)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        y1, y2 = views
        z1 = self.bn(self.projector(self.encoder(y1)))
        z2 = self.bn(self.projector(self.encoder(y2)))

        n = z1.size(0)
        c = (z1.t() @ z2) / n

        on = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = self._off_diag(c).pow_(2).sum()
        return on + self.lambd * off

    @staticmethod
    def _off_diag(m: torch.Tensor) -> torch.Tensor:
        n = m.size(0)
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SwAV(SSLMethod):
    def __init__(self, *, encoder: nn.Module, feature_dim: int, cfg: SwAVConfig = SwAVConfig(), hidden_dim: int = 2048):
        super().__init__(encoder=encoder, feature_dim=feature_dim)
        self.cfg = cfg
        self.projector = MLP(feature_dim, hidden_dim, cfg.feat_dim, num_layers=2)
        self.prototypes = nn.Linear(cfg.feat_dim, cfg.n_prototypes, bias=False)

    def forward(self, crops: Sequence[torch.Tensor]) -> torch.Tensor:
        z = [l2_normalize(self.projector(self.encoder(x))) for x in crops]
        out = [self.prototypes(v) for v in z]

        loss = 0.0
        n_assign = len(self.cfg.crops_for_assign)
        for i, crop_id in enumerate(self.cfg.crops_for_assign):
            with torch.no_grad():
                q = self._sinkhorn(out[crop_id])
            subloss = 0.0
            for v, o in enumerate(out):
                if v == crop_id:
                    continue
                p = F.log_softmax(o / self.cfg.temperature, dim=1)
                subloss += -(q * p).sum(dim=1).mean()
            loss += subloss / (len(out) - 1)
        return loss / max(1, n_assign)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        q = torch.exp(logits / self.cfg.epsilon).t()  # (K,B)
        q /= q.sum()

        k, b = q.shape
        for _ in range(self.cfg.sinkhorn_iterations):
            q /= q.sum(dim=1, keepdim=True)
            q /= k
            q /= q.sum(dim=0, keepdim=True)
            q /= b
        q *= b
        return q.t().detach()
