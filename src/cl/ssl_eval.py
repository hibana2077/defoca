from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

from ..pipeline import NormalizeConfig, normalize_tensor


@dataclass(frozen=True)
class EvalConfig:
    linear_epochs: int = 20
    linear_lr: float = 1e-2
    knn_k: int = 20
    knn_temperature: float = 0.1
    max_knn_train: int = 50000


class PretrainEvaluator:
    def __init__(self, *, encoder: nn.Module, device: str, norm_cfg: NormalizeConfig = NormalizeConfig()):
        self.encoder = encoder
        self.device = torch.device(device)
        self.norm_cfg = norm_cfg
        self.encoder.to(self.device)
        self._feature_dim: Optional[int] = None
        out_dim = getattr(encoder, "out_dim", None)
        if isinstance(out_dim, int) and out_dim > 0:
            self._feature_dim = int(out_dim)

    @torch.no_grad()
    def _features(self, loader: DataLoader, *, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        feats, ys = [], []
        seen = 0
        for x, y in loader:
            x = normalize_tensor(x.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            f = self.encoder(x)
            feats.append(f.detach().cpu())
            ys.append(y.detach().cpu())
            seen += int(y.size(0))
            if max_samples is not None and seen >= int(max_samples):
                break
        return torch.cat(feats, dim=0), torch.cat(ys, dim=0)

    def linear_eval(self, *, train_loader: DataLoader, val_loader: DataLoader, num_classes: int, cfg: EvalConfig) -> Dict[str, float]:
        self.encoder.eval()
        feat_dim = self._feature_dim if self._feature_dim is not None else self._infer_dim(train_loader)
        head = nn.Linear(int(feat_dim), num_classes).to(self.device)
        opt = torch.optim.SGD(head.parameters(), lr=cfg.linear_lr, momentum=0.9)

        for _ in range(int(cfg.linear_epochs)):
            head.train()
            for x, y in train_loader:
                x = normalize_tensor(x.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
                y = y.to(self.device, non_blocking=True)
                with torch.no_grad():
                    f = self.encoder(x)
                logits = head(f)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = normalize_tensor(x.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
                y = y.to(self.device, non_blocking=True)
                f = self.encoder(x)
                pred = head(f).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        return {"linear_acc": correct / max(1, total)}

    def knn_eval(self, *, train_loader: DataLoader, val_loader: DataLoader, cfg: EvalConfig) -> Dict[str, float]:
        train_f, train_y = self._features(train_loader, max_samples=int(cfg.max_knn_train))

        train_f = F.normalize(train_f, dim=1)
        train_y = train_y.long()

        correct = total = 0
        self.encoder.eval()
        for x, y in val_loader:
            x = normalize_tensor(x.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            y = y.to(self.device, non_blocking=True).long()
            with torch.no_grad():
                f = F.normalize(self.encoder(x), dim=1).cpu()

            sims = f @ train_f.t()
            vals, idx = sims.topk(k=int(cfg.knn_k), dim=1)
            votes = train_y[idx]  # (B,K)
            weights = torch.exp(vals / float(cfg.knn_temperature))

            num_classes = int(train_y.max().item()) + 1
            scores = torch.zeros(f.size(0), num_classes)
            scores.scatter_add_(1, votes, weights)
            pred = scores.argmax(dim=1)

            correct += int((pred == y.cpu()).sum().item())
            total += int(y.size(0))

        return {"knn_acc": correct / max(1, total)}

    def clustering_eval(self, *, loader: DataLoader, num_classes: int) -> Dict[str, float]:
        f, y = self._features(loader)
        f_np = f.numpy().astype(np.float32)
        y_np = y.numpy().astype(np.int64)
        km = KMeans(n_clusters=int(num_classes), n_init="auto", random_state=0)
        pred = km.fit_predict(f_np)
        return {
            "nmi": float(normalized_mutual_info_score(y_np, pred)),
            "ari": float(adjusted_rand_score(y_np, pred)),
        }

    @torch.no_grad()
    def _infer_dim(self, loader: DataLoader) -> int:
        x, _ = next(iter(loader))
        x = normalize_tensor(x.to(self.device), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
        return int(self.encoder(x).shape[1])
