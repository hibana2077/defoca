from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class CeaConfig:
        """Consensus Evidence Alignment (CEA).

        Loss-only steering version from docs/TIPver.md.

        Notes:
            - Expects DEFOCA views (V>1) for meaningful consensus.
            - Evidence is extracted from model internal features when possible.
        """

        enabled: bool = False
        # Evidence extraction
        signal: str = "gradcam"  # gradcam | feat_norm | input_grad
        P: int = 4  # evidence grid size (P×P). Usually matches DefocaConfig.P
        tau: float = 0.2  # softmax temperature for q^{(v)}
        gamma: float = 1.0  # stability penalty exp(-gamma * var)
        topk: int = 4  # anchor set size (k)

        # Loss weights
        lambda_align: float = 1.0  # global multiplier for alignment
        lambda_js: float = 1.0
        lambda_iou: float = 1.0

        # Numerics
        eps: float = 1e-8


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
        cea_cfg: CeaConfig = CeaConfig(),
        norm_cfg: NormalizeConfig = NormalizeConfig(),
    ):
        self.model = model
        self.num_classes = int(num_classes)
        self.train_cfg = train_cfg
        self.defoca_cfg = defoca_cfg
        self.cea_cfg = cea_cfg
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

        if self.cea_cfg.enabled and self.defoca is None:
            raise ValueError("CEA is enabled but DEFOCA is disabled. Enable --defoca or disable --cea.")

    def _make_views(
        self,
        images: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        if self.defoca is None:
            views = images.unsqueeze(1)  # (B,1,C,H,W)
        else:
            views = self.defoca(images, generator=generator)  # (B,V,C,H,W)
        return normalize_tensor(views, mean=self.norm_cfg.mean, std=self.norm_cfg.std)

    def _timm_forward_features_and_logits(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Best-effort extraction of internal features + logits for timm models."""
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            feats_out = self.model.forward_features(x)  # type: ignore[attr-defined]
            # timm forward_head sometimes accepts pre_logits kwarg.
            try:
                logits = self.model.forward_head(feats_out, pre_logits=False)  # type: ignore[attr-defined]
            except TypeError:
                logits = self.model.forward_head(feats_out)  # type: ignore[attr-defined]

            # Pick a tensor for evidence extraction.
            feats_tensor: Optional[torch.Tensor]
            if isinstance(feats_out, torch.Tensor):
                feats_tensor = feats_out
            elif isinstance(feats_out, (tuple, list)) and len(feats_out) > 0 and isinstance(feats_out[-1], torch.Tensor):
                feats_tensor = feats_out[-1]
            else:
                feats_tensor = None

            return feats_tensor, logits

        # Fallback: no features.
        return None, self.model(x)

    def _maybe_drop_cls_token(self, t: torch.Tensor) -> torch.Tensor:
        """Drop first token if it looks like a cls token.

        Heuristic:
          - if N is square -> keep
          - else if N-1 is square -> drop first
        """
        if t.dim() != 2:
            raise ValueError(f"Expected (B,N) token map, got {tuple(t.shape)}")
        n = int(t.size(1))
        side = int(round(n**0.5))
        if side * side == n:
            return t
        side2 = int(round((n - 1) ** 0.5))
        if n > 1 and side2 * side2 == (n - 1):
            return t[:, 1:]
        return t

    def _pool_to_grid(self, s: torch.Tensor, *, P: int) -> torch.Tensor:
        """Pool a saliency/evidence map to (P,P).

        Accepts:
          - (B,H,W)
          - (B,N) where N is a square
        Returns (B,P,P)
        """
        P = int(P)
        if P <= 0:
            raise ValueError(f"P must be >= 1, got {P}")

        if s.dim() == 3:
            return F.adaptive_avg_pool2d(s.unsqueeze(1), output_size=(P, P)).squeeze(1)

        if s.dim() == 2:
            n = int(s.size(1))
            side = int(round(n**0.5))
            if side * side != n:
                raise ValueError(f"Cannot reshape token map with N={n} to square grid")
            s2 = s.view(s.size(0), 1, side, side)
            return F.adaptive_avg_pool2d(s2, output_size=(P, P)).squeeze(1)

        raise ValueError(f"Expected (B,H,W) or (B,N), got {tuple(s.shape)}")

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor, *, eps: float) -> torch.Tensor:
        """Jensen-Shannon divergence between distributions.

        p: (...,N) prob, q: (...,N) prob
        returns: (...) scalar per leading dims
        """
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum(dim=-1)
        kl_qm = (q * (q / m).log()).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def _topk_iou(self, a: torch.Tensor, b: torch.Tensor, *, k: int) -> torch.Tensor:
        """IoU between top-k index sets of distributions.

        a: (B,N), b: (B,N)
        returns: (B,) IoU
        """
        k = int(k)
        if k <= 0:
            raise ValueError(f"topk must be >= 1, got {k}")
        bsz, n = a.shape
        k = min(k, n)

        a_idx = a.topk(k, dim=-1).indices
        b_idx = b.topk(k, dim=-1).indices
        a_mask = torch.zeros((bsz, n), dtype=torch.bool, device=a.device)
        b_mask = torch.zeros((bsz, n), dtype=torch.bool, device=a.device)
        a_mask.scatter_(1, a_idx, True)
        b_mask.scatter_(1, b_idx, True)

        inter = (a_mask & b_mask).sum(dim=1).float()
        union = (a_mask | b_mask).sum(dim=1).float().clamp_min(1.0)
        return inter / union

    def _compute_cea_losses(
        self,
        *,
        views: torch.Tensor,
        labels: torch.Tensor,
        feats: Optional[torch.Tensor],
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute alignment loss from DEFOCA views.

        views: (B,V,C,H,W) normalized
        labels: (B,) long

        Returns:
          align_loss (scalar), plus logging dict.
        """
        cfg = self.cea_cfg
        if not cfg.enabled:
            zero = views.new_zeros(())
            return zero, {"align": zero, "js": zero, "iou": zero}

        b, v, c, h, w = views.shape
        x = views.reshape(b * v, c, h, w)
        y = labels.repeat_interleave(v)
        # Evidence source selection
        signal = str(cfg.signal)
        eps = float(cfg.eps)

        if signal == "input_grad" or feats is None:
            x2 = x.detach().requires_grad_(True)
            _feats2, logits2 = self._timm_forward_features_and_logits(x2)
            scores = logits2.gather(1, y.view(-1, 1)).squeeze(1)
            grad_x = torch.autograd.grad(scores.sum(), x2, retain_graph=True, create_graph=False)[0].detach()
            s = grad_x.abs().mean(dim=1)  # (B*V,H,W)
            grid = self._pool_to_grid(s, P=cfg.P)  # (B*V,P,P)
        else:
            # Feature-based evidence
            if signal == "gradcam":
                scores = logits.gather(1, y.view(-1, 1)).squeeze(1)
                grads = torch.autograd.grad(scores.sum(), feats, retain_graph=True, create_graph=False)[0].detach()
                if feats.dim() == 4:
                    s = (grads * feats).sum(dim=1).relu()  # (B*V,Hf,Wf)
                    grid = self._pool_to_grid(s, P=cfg.P)
                elif feats.dim() == 3:
                    # (B*V,N,C) tokens; drop cls token when present.
                    token_rel = (grads * feats).sum(dim=2).relu()  # (B*V,N)
                    token_rel = self._maybe_drop_cls_token(token_rel)
                    grid = self._pool_to_grid(token_rel, P=cfg.P)
                else:
                    raise ValueError(f"Unsupported feature shape for gradcam: {tuple(feats.shape)}")
            elif signal == "feat_norm":
                if feats.dim() == 4:
                    s = feats.pow(2).sum(dim=1).sqrt()  # (B*V,Hf,Wf)
                    grid = self._pool_to_grid(s, P=cfg.P)
                elif feats.dim() == 3:
                    token_norm = feats.pow(2).sum(dim=2).sqrt()  # (B*V,N)
                    token_norm = self._maybe_drop_cls_token(token_norm)
                    grid = self._pool_to_grid(token_norm, P=cfg.P)
                else:
                    raise ValueError(f"Unsupported feature shape for feat_norm: {tuple(feats.shape)}")
            else:
                raise ValueError(f"Unknown CEA signal: {signal} (expected gradcam|feat_norm|input_grad)")

        n = int(cfg.P) * int(cfg.P)
        r = grid.view(b, v, n)  # (B,V,N)
        q = F.softmax(r / float(cfg.tau), dim=-1)
        mu = q.mean(dim=1)
        var = q.var(dim=1, unbiased=False)
        c_score = mu * torch.exp(-float(cfg.gamma) * var)
        q_star = c_score / c_score.sum(dim=-1, keepdim=True).clamp_min(eps)

        # JS divergence (B,V)
        js = self._js_divergence(q, q_star.unsqueeze(1).expand_as(q), eps=eps).mean(dim=1)  # (B,)
        js_mean = js.mean()

        # IoU over top-k anchor sets (B,V)
        ious = []
        for vi in range(v):
            ious.append(self._topk_iou(q[:, vi, :], q_star, k=cfg.topk))
        iou = torch.stack(ious, dim=1).mean(dim=1)  # (B,)
        iou_mean = iou.mean()
        d_iou = 1.0 - iou_mean

        align = float(cfg.lambda_js) * js_mean + float(cfg.lambda_iou) * d_iou
        align = float(cfg.lambda_align) * align
        return align, {"align": align.detach(), "js": js_mean.detach(), "iou": iou_mean.detach()}

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
        total_align = 0.0

        for step, (images, labels) in enumerate(loader):
            if self.cea_cfg.enabled:
                # CEA path: keep (B,V,...) grouping for alignment.
                views = self._make_views(images, generator=generator)  # (B,V,C,H,W)
                labels = labels.to(self.device, non_blocking=True)
                b, v, c, h, w = views.shape
                x = views.view(b * v, c, h, w)
                y = labels.repeat_interleave(v)

                self.optimizer.zero_grad(set_to_none=True)
                feats, logits = self._timm_forward_features_and_logits(x)
                cls_loss = self.criterion(logits, y)
                align_loss, cea_logs = self._compute_cea_losses(views=views, labels=labels, feats=feats, logits=logits)
                loss = cls_loss + align_loss
                loss.backward()
                self.optimizer.step()

                bs = int(y.size(0))
                total_loss += float(loss.item()) * bs
                total_align += float(cea_logs["align"].item()) * bs
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += bs
            else:
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
            "align": total_align / max(1, total) if self.cea_cfg.enabled else 0.0,
        }
