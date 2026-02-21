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

        # Version B: evidence-guided gating (computation steering)
        gate_enabled: bool = False
        gate_alpha: float = 1.0  # strength; 0 disables effect even if enabled
        gate_target: str = "auto"  # auto | vit | cnn
        vit_block: int = -1  # -1 = middle block
        cnn_stage: str = "layer3"  # e.g., layer2/layer3/layer4 for ResNet

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

        self._gate_handles = []
        self._gate_active = False
        self._gate_mask_tokens: Optional[torch.Tensor] = None  # (B*V,Npatch)
        self._gate_mask_map: Optional[torch.Tensor] = None  # (B*V,1,H,W) (lazy resized in hook)
        self._gate_is_vit = False
        self._gate_is_cnn = False

        if self.cea_cfg.enabled and self.cea_cfg.gate_enabled:
            self._init_gating_hooks()

    def _init_gating_hooks(self) -> None:
        cfg = self.cea_cfg
        target = str(cfg.gate_target)

        is_vit = hasattr(self.model, "blocks") and isinstance(getattr(self.model, "blocks"), (list, nn.ModuleList, nn.Sequential))
        
        try:
            cnn_stage_module = self.model.get_submodule(str(cfg.cnn_stage))
            is_cnn = True
        except AttributeError:
            cnn_stage_module = None
            is_cnn = False

        if target == "auto":
            if is_vit:
                target = "vit"
            elif is_cnn:
                target = "cnn"
            elif hasattr(self.model, "stages"):
                # For models like TinyViT, Swin, ConvNeXt which have 'stages'
                target = "cnn"
                # Try to find the last stage
                stages = getattr(self.model, "stages")
                if isinstance(stages, (list, nn.ModuleList, nn.Sequential)):
                    import dataclasses
                    cfg = dataclasses.replace(cfg, cnn_stage=f"stages.{len(stages)-1}")
                    self.cea_cfg = cfg
                    cnn_stage_module = self.model.get_submodule(cfg.cnn_stage)
                    is_cnn = True
                else:
                    raise ValueError("CEA gating target auto-detection failed for this model (stages is not a list/Sequential)")
            else:
                raise ValueError("CEA gating target auto-detection failed for this model")

        if target == "vit":
            if not is_vit:
                raise ValueError("CEA gating target 'vit' requested but model has no .blocks")
            blocks = getattr(self.model, "blocks")
            n_blocks = len(blocks)
            idx = int(cfg.vit_block)
            if idx < 0:
                idx = n_blocks // 2
            if idx >= n_blocks:
                raise ValueError(f"vit_block index out of range: {idx} (n_blocks={n_blocks})")

            def _hook(_module, _inp, out):
                if (not self._gate_active) or (self._gate_mask_tokens is None):
                    return out
                if not isinstance(out, torch.Tensor) or out.dim() != 3:
                    return out
                # out: (B*V, N, C)
                n = int(out.size(1))
                mask = self._gate_mask_tokens
                if n <= 0:
                    return out

                # Support both layouts:
                #   - with cls token: patch tokens are [1:], mask length == N-1
                #   - without cls token: patch tokens are [:], mask length == N
                if mask.size(1) == n - 1 and n > 1:
                    patch_slice = slice(1, None)
                elif mask.size(1) == n:
                    patch_slice = slice(0, None)
                else:
                    # Token count mismatch: skip gating to avoid shape errors.
                    return out
                alpha = float(cfg.gate_alpha)
                if alpha == 0.0:
                    return out

                m = mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)  # (B*V,Npatch,1)
                scale = 1.0 + alpha * (m - 1.0)
                out2 = out
                out2 = out2.clone()
                out2[:, patch_slice, :] = out2[:, patch_slice, :] * scale
                return out2

            handle = blocks[idx].register_forward_hook(_hook)
            self._gate_handles.append(handle)
            self._gate_is_vit = True
            return

        if target == "cnn":
            stage_name = str(cfg.cnn_stage)
            if cnn_stage_module is None:
                raise ValueError(f"CEA gating cnn_stage not found: {stage_name}")
            stage = cnn_stage_module

            def _hook(_module, _inp, out):
                if (not self._gate_active) or (self._gate_mask_map is None):
                    return out
                if not isinstance(out, torch.Tensor) or out.dim() != 4:
                    return out
                alpha = float(cfg.gate_alpha)
                if alpha == 0.0:
                    return out
                bsz, ch, hh, ww = out.shape
                mask = self._gate_mask_map
                if mask.dim() != 4 or mask.size(0) != bsz:
                    return out
                if mask.size(-2) != hh or mask.size(-1) != ww:
                    mask = F.interpolate(mask, size=(hh, ww), mode="bilinear", align_corners=False)
                mask = mask.to(device=out.device, dtype=out.dtype)
                scale = 1.0 + alpha * (mask - 1.0)
                return out * scale

            handle = stage.register_forward_hook(_hook)
            self._gate_handles.append(handle)
            self._gate_is_cnn = True
            return

        raise ValueError(f"Unknown gate_target: {target}")

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

    def _upsample_qstar_to_token_mask(self, q_star: torch.Tensor, *, n_patch: int) -> torch.Tensor:
        """Convert q* (B,P*P) into a (B,n_patch) mask with mean=1."""
        cfg = self.cea_cfg
        eps = float(cfg.eps)
        bsz, n = q_star.shape
        P = int(cfg.P)
        if n != P * P:
            raise ValueError(f"q_star length mismatch: got {n}, expected {P*P}")
        side = int(round(n_patch**0.5))
        if side * side != n_patch:
            raise ValueError(f"Cannot upsample q* to non-square patch count n_patch={n_patch}")
        q2 = q_star.view(bsz, 1, P, P)
        up = F.interpolate(q2, size=(side, side), mode="bilinear", align_corners=False).view(bsz, n_patch)
        up = up / up.mean(dim=-1, keepdim=True).clamp_min(eps)
        return up

    def _qstar_to_feature_mask(self, q_star: torch.Tensor, *, size_hw: Tuple[int, int]) -> torch.Tensor:
        """Convert q* (B,P*P) into a (B,1,H,W) mask with mean=1."""
        cfg = self.cea_cfg
        eps = float(cfg.eps)
        bsz, n = q_star.shape
        P = int(cfg.P)
        if n != P * P:
            raise ValueError(f"q_star length mismatch: got {n}, expected {P*P}")
        q2 = q_star.view(bsz, 1, P, P)
        h, w = int(size_hw[0]), int(size_hw[1])
        up = F.interpolate(q2, size=(h, w), mode="bilinear", align_corners=False)
        up = up / up.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return up

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

    def _compute_q_and_qstar(
        self,
        *,
        views: torch.Tensor,
        labels: torch.Tensor,
        feats: Optional[torch.Tensor],
        logits: torch.Tensor,
        retain_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-view evidence distributions q and consensus q*.

        Returns:
          q: (B,V,N)
          q_star: (B,N)
        """
        cfg = self.cea_cfg
        b, v, c, h, w = views.shape
        y = labels.repeat_interleave(v)

        signal = str(cfg.signal)
        eps = float(cfg.eps)

        if signal == "input_grad" or feats is None:
            x = views.reshape(b * v, c, h, w)
            x2 = x.detach().requires_grad_(True)
            _feats2, logits2 = self._timm_forward_features_and_logits(x2)
            scores = logits2.gather(1, y.view(-1, 1)).squeeze(1)
            grad_x = torch.autograd.grad(
                scores.sum(),
                x2,
                retain_graph=bool(retain_graph),
                create_graph=False,
            )[0].detach()
            s = grad_x.abs().mean(dim=1)  # (B*V,H,W)
            grid = self._pool_to_grid(s, P=cfg.P)  # (B*V,P,P)
        else:
            if signal == "gradcam":
                scores = logits.gather(1, y.view(-1, 1)).squeeze(1)
                try:
                    grads = torch.autograd.grad(
                        scores.sum(),
                        feats,
                        retain_graph=bool(retain_graph),
                        create_graph=False,
                    )[0].detach()
                except RuntimeError:
                    signal = "feat_norm"
                    grads = None

                if feats.dim() == 4:
                    if signal == "feat_norm" or grads is None:
                        s = feats.pow(2).sum(dim=1).sqrt()
                    else:
                        s = (grads * feats).sum(dim=1).relu()
                        if s.abs().sum() < 1e-6:
                            s = feats.pow(2).sum(dim=1).sqrt()
                    grid = self._pool_to_grid(s, P=cfg.P)
                elif feats.dim() == 3:
                    if signal == "feat_norm" or grads is None:
                        token_rel = feats.pow(2).sum(dim=2).sqrt()
                    else:
                        token_rel = (grads * feats).sum(dim=2).relu()
                        if token_rel[:, 1:].abs().sum() < 1e-6:
                            token_rel = feats.pow(2).sum(dim=2).sqrt()
                    token_rel = self._maybe_drop_cls_token(token_rel)
                    grid = self._pool_to_grid(token_rel, P=cfg.P)
                else:
                    raise ValueError(f"Unsupported feature shape for evidence: {tuple(feats.shape)}")
            elif signal == "feat_norm":
                if feats.dim() == 4:
                    s = feats.pow(2).sum(dim=1).sqrt()
                    grid = self._pool_to_grid(s, P=cfg.P)
                elif feats.dim() == 3:
                    token_norm = feats.pow(2).sum(dim=2).sqrt()
                    token_norm = self._maybe_drop_cls_token(token_norm)
                    grid = self._pool_to_grid(token_norm, P=cfg.P)
                else:
                    raise ValueError(f"Unsupported feature shape for feat_norm: {tuple(feats.shape)}")
            else:
                raise ValueError(f"Unknown CEA signal: {signal} (expected gradcam|feat_norm|input_grad)")

        n = int(cfg.P) * int(cfg.P)
        r = grid.view(b, v, n)
        q = F.softmax(r / float(cfg.tau), dim=-1)
        mu = q.mean(dim=1)
        var = q.var(dim=1, unbiased=False)
        c_score = mu * torch.exp(-float(cfg.gamma) * var)
        q_star = c_score / c_score.sum(dim=-1, keepdim=True).clamp_min(eps)
        return q, q_star

    def _alignment_losses_from_q(
        self,
        *,
        q: torch.Tensor,
        q_star: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute JS + top-k IoU alignment terms between q (per-view) and target q*."""
        cfg = self.cea_cfg
        eps = float(cfg.eps)
        b, v, n = q.shape
        q_star_bv = q_star.unsqueeze(1).expand(b, v, n)

        js = self._js_divergence(q, q_star_bv, eps=eps).mean(dim=1)  # (B,)
        js_mean = js.mean()

        ious = []
        for vi in range(v):
            ious.append(self._topk_iou(q[:, vi, :], q_star, k=cfg.topk))
        iou = torch.stack(ious, dim=1).mean(dim=1)
        iou_mean = iou.mean()
        d_iou = 1.0 - iou_mean

        align = float(cfg.lambda_js) * js_mean + float(cfg.lambda_iou) * d_iou
        align = float(cfg.lambda_align) * align
        return align, {"align": align.detach(), "js": js_mean.detach(), "iou": iou_mean.detach()}

    def _compute_cea_losses_singlepass(
        self,
        *,
        views: torch.Tensor,
        labels: torch.Tensor,
        feats: Optional[torch.Tensor],
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cea_cfg
        if not cfg.enabled:
            zero = views.new_zeros(())
            return zero, {"align": zero, "js": zero, "iou": zero}
        q, q_star = self._compute_q_and_qstar(views=views, labels=labels, feats=feats, logits=logits)
        return self._alignment_losses_from_q(q=q, q_star=q_star)

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

        total_align = 0.0
        total_js = 0.0
        total_iou = 0.0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Standard classification eval on clean images.
            images_n = normalize_tensor(images, mean=self.norm_cfg.mean, std=self.norm_cfg.std)
            logits = self.model(images_n)
            loss = self.criterion(logits, labels)
            bs = int(labels.size(0))
            total_loss += float(loss.item()) * bs

            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += bs

            # Optional: also report CEA alignment metrics on DEFOCA views.
            if self.cea_cfg.enabled:
                if self.defoca is None:
                    raise RuntimeError("CEA enabled but DEFOCA is missing")
                with torch.enable_grad():
                    views = self._make_views(images)  # (B,V,C,H,W) normalized
                    b, v, c, h, w = views.shape
                    x = views.view(b * v, c, h, w)
                    feats, logits_v = self._timm_forward_features_and_logits(x)
                    q, q_star = self._compute_q_and_qstar(
                        views=views,
                        labels=labels,
                        feats=feats,
                        logits=logits_v,
                        retain_graph=False,
                    )
                    _align_loss, cea_logs = self._alignment_losses_from_q(q=q, q_star=q_star)
                    total_align += float(cea_logs["align"].item()) * bs
                    total_js += float(cea_logs["js"].item()) * bs
                    total_iou += float(cea_logs["iou"].item()) * bs

        out = {
            "loss": total_loss / max(1, total),
            "acc": correct / max(1, total),
        }
        if self.cea_cfg.enabled:
            out.update(
                {
                    "align": total_align / max(1, total),
                    "js": total_js / max(1, total),
                    "iou": total_iou / max(1, total),
                }
            )
        return out

    def train_one_epoch(self, loader: DataLoader, *, epoch: int, generator: Optional[torch.Generator] = None) -> Dict[str, float]:
        self.model.train()
        total = 0
        correct = 0
        total_loss = 0.0
        total_align = 0.0
        total_js = 0.0
        total_iou = 0.0

        for step, (images, labels) in enumerate(loader):
            if self.cea_cfg.enabled:
                # CEA path: keep (B,V,...) grouping for alignment.
                views = self._make_views(images, generator=generator)  # (B,V,C,H,W)
                labels = labels.to(self.device, non_blocking=True)
                b, v, c, h, w = views.shape
                x = views.view(b * v, c, h, w)
                y = labels.repeat_interleave(v)

                self.optimizer.zero_grad(set_to_none=True)
                cfg = self.cea_cfg
                if cfg.gate_enabled and float(cfg.gate_alpha) != 0.0:
                    # Two-pass gating: pass1 builds stop-grad q*, pass2 applies gating.
                    self._gate_active = False
                    self._gate_mask_tokens = None
                    self._gate_mask_map = None

                    feats1, logits1 = self._timm_forward_features_and_logits(x)
                    q1, q_star1 = self._compute_q_and_qstar(
                        views=views,
                        labels=labels,
                        feats=feats1,
                        logits=logits1,
                        retain_graph=False,
                    )
                    q_star1 = q_star1.detach()

                    # Build gating masks and activate hooks for pass2.
                    self._gate_active = True
                    if self._gate_is_vit:
                        # Need patch token count; infer from a cheap forward output shape.
                        # We'll infer from feats1 if token-like; else from model output on pass2 hook.
                        # For safety, use a forward_features token count when available.
                        n_patch = None
                        if feats1 is not None and isinstance(feats1, torch.Tensor) and feats1.dim() == 3:
                            tok_rel = feats1
                            n_tok = int(tok_rel.size(1))
                            # If has cls token, use n-1.
                            n_patch = int(self._maybe_drop_cls_token(torch.zeros((1, n_tok), device=tok_rel.device)).size(1))
                        if n_patch is None:
                            # Fallback: assume square of timm default patch grid inferred from input size.
                            n_patch = 196

                        tok_mask_b = self._upsample_qstar_to_token_mask(q_star1, n_patch=n_patch)  # (B,Npatch)
                        self._gate_mask_tokens = tok_mask_b.repeat_interleave(v, dim=0)
                    if self._gate_is_cnn:
                        # Provide a coarse mask; hook will resize to stage output.
                        m = self._qstar_to_feature_mask(q_star1, size_hw=(int(cfg.P), int(cfg.P)))
                        self._gate_mask_map = m.repeat_interleave(v, dim=0)

                    feats2, logits = self._timm_forward_features_and_logits(x)
                    cls_loss = self.criterion(logits, y)
                    q2, _q_star2 = self._compute_q_and_qstar(views=views, labels=labels, feats=feats2, logits=logits, retain_graph=True)
                    align_loss, cea_logs = self._alignment_losses_from_q(q=q2, q_star=q_star1)
                    loss = cls_loss + align_loss

                    # Disable gating after the step to avoid leaking state.
                    self._gate_active = False
                    self._gate_mask_tokens = None
                    self._gate_mask_map = None
                else:
                    feats, logits = self._timm_forward_features_and_logits(x)
                    cls_loss = self.criterion(logits, y)
                    align_loss, cea_logs = self._compute_cea_losses_singlepass(views=views, labels=labels, feats=feats, logits=logits)
                    loss = cls_loss + align_loss
                loss.backward()
                self.optimizer.step()

                bs = int(y.size(0))
                total_loss += float(loss.item()) * bs
                total_align += float(cea_logs["align"].item()) * bs
                total_js += float(cea_logs["js"].item()) * bs
                total_iou += float(cea_logs["iou"].item()) * bs
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
            "js": total_js / max(1, total) if self.cea_cfg.enabled else 0.0,
            "iou": total_iou / max(1, total) if self.cea_cfg.enabled else 0.0,
        }
