from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import json
import math

import numpy as np
import torch
import torch.nn as nn

from .defoca import PatchGrid, gaussian_blur_patch
from .pipeline import NormalizeConfig, normalize_tensor


EnergyMode = Literal["total", "highfreq"]
SaliencyTarget = Literal["predicted", "label", "index"]


@dataclass(frozen=True)
class SNRConfig:
    img_size: int = 224
    P: int = 4
    topk_ratio: float = 0.25
    sigma: float = 1.0
    energy: EnergyMode = "total"
    cutoff_ratio: float = 0.5
    eps: float = 1e-8
    saliency_target: SaliencyTarget = "predicted"
    target_index: int = 0
    norm: NormalizeConfig = NormalizeConfig()


def _to_scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def compute_saliency_absgrad(
    *,
    model: nn.Module,
    image: torch.Tensor,
    label: Optional[int],
    cfg: SNRConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Vanilla saliency map using absolute input gradient.

    Returns:
      saliency: (H,W) float32 on CPU
      target: int target class used

    Notes:
      - `image` expected in [0,1] range, shape (C,H,W)
      - Saliency is computed w.r.t. the *unnormalized* image tensor, but the
        model sees the normalized tensor.
    """

    if image.dim() != 3:
        raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")

    x = image.to(device=device, dtype=torch.float32).requires_grad_(True)
    x_norm = normalize_tensor(x, mean=cfg.norm.mean, std=cfg.norm.std)

    model.zero_grad(set_to_none=True)
    logits = model(x_norm.unsqueeze(0))  # (1,K)
    if logits.dim() != 2 or logits.size(0) != 1:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    if cfg.saliency_target == "predicted":
        target = int(torch.argmax(logits[0]).item())
    elif cfg.saliency_target == "label":
        if label is None:
            raise ValueError("label is required when saliency_target='label'")
        target = int(label)
    else:
        target = int(cfg.target_index)

    if target < 0 or target >= int(logits.size(1)):
        raise ValueError(
            f"Target index {target} out of range for logits K={int(logits.size(1))}. "
            "Use --saliency-target predicted, or provide a compatible checkpoint."
        )

    score = logits[0, target]
    score.backward()

    if x.grad is None:
        raise RuntimeError("No gradient computed for input")

    sal = x.grad.detach().abs().sum(dim=0)  # (H,W)
    sal = sal / (sal.max().clamp_min(cfg.eps))
    return sal.cpu(), target


def saliency_to_patch_set(*, saliency_hw: torch.Tensor, P: int, topk_ratio: float) -> Set[int]:
    """Aggregate saliency into P×P grid and take top-k% patches."""

    if saliency_hw.dim() != 2:
        raise ValueError(f"Expected saliency (H,W), got {tuple(saliency_hw.shape)}")

    H, W = int(saliency_hw.size(0)), int(saliency_hw.size(1))
    grid = PatchGrid(H=H, W=W, P=int(P))
    grid.validate()

    patch_scores: List[float] = []
    for idx in range(grid.N):
        y0, y1, x0, x1 = grid.bbox(idx)
        score = saliency_hw[y0:y1, x0:x1].mean()
        patch_scores.append(float(score.item()))

    k = int(round(float(topk_ratio) * grid.N))
    k = max(1, min(k, grid.N))

    order = np.argsort(np.asarray(patch_scores))[::-1]
    selected = set(int(i) for i in order[:k].tolist())
    return selected


def _fft_power(patch: torch.Tensor) -> torch.Tensor:
    # patch: (C,H,W)
    fft = torch.fft.fft2(patch, norm="ortho")
    power = torch.abs(fft) ** 2
    return power


def patch_energy_total(patch: torch.Tensor) -> torch.Tensor:
    return _fft_power(patch).sum()


def patch_energy_highfreq(patch: torch.Tensor, *, cutoff_ratio: float) -> torch.Tensor:
    power = _fft_power(patch)  # (C,H,W)
    _, H, W = power.shape

    # Shift so DC is centered.
    power = torch.fft.fftshift(power, dim=(-2, -1))

    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=power.device),
        torch.arange(W, device=power.device),
        indexing="ij",
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).float()
    cutoff = float(cutoff_ratio) * float(max(H, W))
    mask = dist > cutoff

    # mask: (H,W), power: (C,H,W)
    return power[:, mask].sum()


def patch_energy(
    patch: torch.Tensor,
    *,
    mode: EnergyMode,
    cutoff_ratio: float,
) -> torch.Tensor:
    if mode == "total":
        return patch_energy_total(patch)
    if mode == "highfreq":
        return patch_energy_highfreq(patch, cutoff_ratio=cutoff_ratio)
    raise ValueError(f"Unknown energy mode: {mode}")


def compute_snr_from_image(
    *,
    image: torch.Tensor,
    S: Set[int],
    cfg: SNRConfig,
) -> Dict[str, float]:
    """Compute baseline SNR (Eq.8 numerator/denominator from original image)."""

    if image.dim() != 3:
        raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")

    C, H, W = image.shape
    grid = PatchGrid(H=int(H), W=int(W), P=int(cfg.P))
    grid.validate()

    signal = image.new_zeros(())
    noise = image.new_zeros(())

    for idx in range(grid.N):
        y0, y1, x0, x1 = grid.bbox(idx)
        p = image[:, y0:y1, x0:x1]
        e = patch_energy(p, mode=cfg.energy, cutoff_ratio=cfg.cutoff_ratio)
        if idx in S:
            signal = signal + e
        else:
            noise = noise + e

    snr = signal / (noise + float(cfg.eps))
    return {
        "signal": _to_scalar(signal),
        "noise": _to_scalar(noise),
        "snr": _to_scalar(snr),
    }


def blur_non_discriminative_patches(
    *,
    image: torch.Tensor,
    S: Set[int],
    cfg: SNRConfig,
) -> torch.Tensor:
    """Blur patches NOT in S using the same Gaussian as DEFOCA."""

    if image.dim() != 3:
        raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")

    out = image.clone()
    C, H, W = out.shape
    grid = PatchGrid(H=int(H), W=int(W), P=int(cfg.P))
    grid.validate()

    for idx in range(grid.N):
        if idx in S:
            continue
        y0, y1, x0, x1 = grid.bbox(idx)
        patch = out[:, y0:y1, x0:x1]
        out[:, y0:y1, x0:x1] = gaussian_blur_patch(patch, cfg.sigma)

    return out


def compute_snr_prime(
    *,
    image: torch.Tensor,
    S: Set[int],
    cfg: SNRConfig,
) -> Dict[str, float]:
    """Compute SNR' (Eq.8 denominator after low-pass filtering non-discriminative patches)."""

    if image.dim() != 3:
        raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")

    C, H, W = image.shape
    grid = PatchGrid(H=int(H), W=int(W), P=int(cfg.P))
    grid.validate()

    # Numerator uses original discriminative patch energies.
    signal = image.new_zeros(())
    for idx in S:
        y0, y1, x0, x1 = grid.bbox(int(idx))
        p = image[:, y0:y1, x0:x1]
        signal = signal + patch_energy(p, mode=cfg.energy, cutoff_ratio=cfg.cutoff_ratio)

    # Denominator uses blurred non-discriminative patch energies.
    x_blur = blur_non_discriminative_patches(image=image, S=S, cfg=cfg)
    noise = image.new_zeros(())
    for idx in range(grid.N):
        if idx in S:
            continue
        y0, y1, x0, x1 = grid.bbox(idx)
        p = x_blur[:, y0:y1, x0:x1]
        noise = noise + patch_energy(p, mode=cfg.energy, cutoff_ratio=cfg.cutoff_ratio)

    snr_prime = signal / (noise + float(cfg.eps))
    return {
        "signal": _to_scalar(signal),
        "noise_blur": _to_scalar(noise),
        "snr_prime": _to_scalar(snr_prime),
    }


@dataclass
class SNRResult:
    snr: float
    snr_prime: float
    delta: float
    ratio: float
    target: int


def aggregate_results(results: Sequence[SNRResult]) -> Dict[str, Any]:
    snr = np.asarray([r.snr for r in results], dtype=np.float64)
    snr_p = np.asarray([r.snr_prime for r in results], dtype=np.float64)
    delta = snr_p - snr
    ratio = snr_p / np.maximum(snr, 1e-12)

    out: Dict[str, Any] = {
        "n": int(len(results)),
        "snr": {"mean": float(snr.mean()), "std": float(snr.std(ddof=1) if len(snr) > 1 else 0.0)},
        "snr_prime": {"mean": float(snr_p.mean()), "std": float(snr_p.std(ddof=1) if len(snr_p) > 1 else 0.0)},
        "delta": {"mean": float(delta.mean()), "std": float(delta.std(ddof=1) if len(delta) > 1 else 0.0)},
        "ratio": {"mean": float(ratio.mean()), "std": float(ratio.std(ddof=1) if len(ratio) > 1 else 0.0)},
    }

    try:
        from scipy import stats  # type: ignore

        if len(delta) >= 2:
            t = stats.ttest_rel(snr_p, snr)
            out["ttest_rel"] = {"statistic": float(t.statistic), "pvalue": float(t.pvalue)}
    except Exception:
        # SciPy may be missing in minimal environments; stats are optional.
        pass

    return out


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
