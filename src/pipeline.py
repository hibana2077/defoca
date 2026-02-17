from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .defoca import DEFOCA


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50
    flatten_views: bool = True  # (B,V,...) -> (B*V,...)


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50


@dataclass(frozen=True)
class DefocaConfig:
    enabled: bool = True
    P: int = 4
    ratio: float = 0.25
    sigma: float = 1.0
    strategy: str = "contiguous"  # random|contiguous|dispersed
    V: int = 8
    max_attempts: int = 10
    ensure_unique: bool = True


@dataclass(frozen=True)
class NormalizeConfig:
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def normalize_tensor(x: torch.Tensor, *, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    """Normalize tensor with broadcasting.

    Supports shapes:
      - (C,H,W)
      - (B,C,H,W)
      - (B,V,C,H,W)
    """
    if x.dim() < 3:
        raise ValueError(f"Expected image tensor with >=3 dims, got {tuple(x.shape)}")
    c = x.shape[-3]
    if c != len(mean) or c != len(std):
        raise ValueError(f"Channel mismatch: C={c}, mean/std={len(mean)}")

    device = x.device
    dtype = x.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype)
    std_t = torch.tensor(std, device=device, dtype=dtype)

    shape = [1] * x.dim()
    shape[-3] = c
    mean_t = mean_t.view(*shape)
    std_t = std_t.view(*shape)
    return (x - mean_t) / std_t


class ClsPipeline:
    """Classification pipeline that applies DEFOCA only during training."""

    def __init__(
        self,
        *,
        model: nn.Module,
        num_classes: int,
        train_cfg: TrainConfig,
        defoca_cfg: DefocaConfig,
        norm_cfg: NormalizeConfig = NormalizeConfig(),
    ):
        self.model = model
        self.num_classes = int(num_classes)
        self.train_cfg = train_cfg
        self.defoca_cfg = defoca_cfg
        self.norm_cfg = norm_cfg

        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.defoca: Optional[DEFOCA]
        if defoca_cfg.enabled:
            self.defoca = DEFOCA(
                P=defoca_cfg.P,
                ratio=defoca_cfg.ratio,
                sigma=defoca_cfg.sigma,
                strategy=defoca_cfg.strategy,  # type: ignore[arg-type]
                V=defoca_cfg.V,
                max_attempts=defoca_cfg.max_attempts,
                ensure_unique=defoca_cfg.ensure_unique,
            )
        else:
            self.defoca = None

    def _prepare_train_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if self.defoca is not None:
            views = self.defoca(images, generator=generator)  # (B,V,C,H,W)
            views = normalize_tensor(views, mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            if self.train_cfg.flatten_views:
                b, v, c, h, w = views.shape
                views = views.reshape(b * v, c, h, w)
                labels = labels.repeat_interleave(v)
            return views, labels

        images = normalize_tensor(images, mean=self.norm_cfg.mean, std=self.norm_cfg.std)
        return images, labels

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            images = normalize_tensor(images, mean=self.norm_cfg.mean, std=self.norm_cfg.std)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)

            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

        return {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total),
        }

    def train_one_epoch(self, loader: DataLoader, *, epoch: int, generator: Optional[torch.Generator] = None) -> Dict[str, float]:
        self.model.train()
        total = 0
        correct = 0
        total_loss = 0.0

        for step, (images, labels) in enumerate(loader):
            x, y = self._prepare_train_batch(images, labels, generator=generator)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            bs = int(y.size(0))
            total_loss += float(loss.item()) * bs
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += bs

            if self.train_cfg.log_every > 0 and (step + 1) % self.train_cfg.log_every == 0:
                print(
                    f"epoch={epoch} step={step+1}/{len(loader)} "
                    f"loss={total_loss/max(1,total):.4f} acc={correct/max(1,total):.4f}"
                )

        return {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total),
        }


Views = Union[tuple[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]]


class PretrainPipeline:
    def __init__(
        self,
        *,
        method: nn.Module,
        pretrain_cfg: PretrainConfig,
        norm_cfg: NormalizeConfig = NormalizeConfig(),
    ):
        self.method = method
        self.pretrain_cfg = pretrain_cfg
        self.norm_cfg = norm_cfg

        self.device = torch.device(pretrain_cfg.device)
        self.method.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.method.parameters(),
            lr=pretrain_cfg.lr,
            weight_decay=pretrain_cfg.weight_decay,
        )

    def _to_device(self, views: Views) -> Views:
        if isinstance(views, tuple):
            x1, x2 = views
            x1 = normalize_tensor(x1.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            x2 = normalize_tensor(x2.to(self.device, non_blocking=True), mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            return (x1, x2)

        out: list[torch.Tensor] = []
        for x in views:
            x = x.to(self.device, non_blocking=True)
            out.append(normalize_tensor(x, mean=self.norm_cfg.mean, std=self.norm_cfg.std))
        return out

    def train_one_epoch(self, loader: DataLoader, *, epoch: int) -> Dict[str, float]:
        self.method.train()
        total = 0
        total_loss = 0.0

        for step, (views, _) in enumerate(loader):
            views = self._to_device(views)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.method(views)
            loss.backward()
            self.optimizer.step()

            bs = int(views[0].size(0) if isinstance(views, tuple) else views[0].size(0))
            total += bs
            total_loss += float(loss.item()) * bs
            if self.pretrain_cfg.log_every > 0 and (step + 1) % self.pretrain_cfg.log_every == 0:
                print(f"epoch={epoch} step={step+1}/{len(loader)} loss={total_loss/max(1,total):.4f}")

        return {"loss": total_loss / max(1, total)}
