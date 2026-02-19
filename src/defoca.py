from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Sequence, Set, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


StrategyName = Literal["random", "contiguous", "dispersed"]


@dataclass(frozen=True)
class PatchGrid:
    """Non-overlapping P×P grid with exact coverage.

    Uses floor(W/P), floor(H/P) for regular patches; last row/col absorb remainder.
    """

    H: int
    W: int
    P: int

    @property
    def N(self) -> int:
        return self.P * self.P

    @property
    def ph(self) -> int:
        return self.H // self.P

    @property
    def pw(self) -> int:
        return self.W // self.P

    def validate(self) -> None:
        if self.P <= 0:
            raise ValueError(f"P must be >= 1, got {self.P}")
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"Invalid image size HxW={self.H}x{self.W}")
        if self.ph <= 0 or self.pw <= 0:
            raise ValueError(
                f"P={self.P} is too large for HxW={self.H}x{self.W} (ph={self.ph}, pw={self.pw})"
            )

    def rc_to_idx(self, r: int, c: int) -> int:
        return r * self.P + c

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        return idx // self.P, idx % self.P

    def bbox(self, idx: int) -> Tuple[int, int, int, int]:
        """Return (y0, y1, x0, x1) in pixel coordinates."""
        r, c = self.idx_to_rc(idx)
        y0 = r * self.ph
        y1 = (r + 1) * self.ph
        x0 = c * self.pw
        x1 = (c + 1) * self.pw

        if r == self.P - 1:
            y1 = self.H
        if c == self.P - 1:
            x1 = self.W
        return y0, y1, x0, x1

    def neighbors4(self, idx: int) -> List[int]:
        r, c = self.idx_to_rc(idx)
        out: List[int] = []
        if r - 1 >= 0:
            out.append(self.rc_to_idx(r - 1, c))
        if r + 1 < self.P:
            out.append(self.rc_to_idx(r + 1, c))
        if c - 1 >= 0:
            out.append(self.rc_to_idx(r, c - 1))
        if c + 1 < self.P:
            out.append(self.rc_to_idx(r, c + 1))
        return out


class PatchSelectionStrategy:
    def select(self, grid: PatchGrid, n: int, *, generator: Optional[torch.Generator] = None) -> List[int]:
        raise NotImplementedError


class RandomSelection(PatchSelectionStrategy):
    def select(self, grid: PatchGrid, n: int, *, generator: Optional[torch.Generator] = None) -> List[int]:
        n = int(n)
        if n <= 0:
            return []
        n = min(n, grid.N)
        perm = torch.randperm(grid.N, generator=generator)
        return perm[:n].tolist()


class ContiguousSelection(PatchSelectionStrategy):
    """Seed patch + 4-neighborhood expansion; fallback random fill if stuck."""

    def select(self, grid: PatchGrid, n: int, *, generator: Optional[torch.Generator] = None) -> List[int]:
        n = int(n)
        if n <= 0:
            return []
        n = min(n, grid.N)

        seed = int(torch.randint(0, grid.N, (1,), generator=generator).item())
        selected: Set[int] = {seed}
        q = deque([seed])

        while q and len(selected) < n:
            cur = q.popleft()
            for nb in grid.neighbors4(cur):
                if nb in selected:
                    continue
                selected.add(nb)
                q.append(nb)
                if len(selected) >= n:
                    break

        if len(selected) < n:
            remaining = torch.tensor(
                [i for i in range(grid.N) if i not in selected],
                dtype=torch.long,
            )
            if remaining.numel() > 0:
                k = min(n - len(selected), int(remaining.numel()))
                perm = remaining[torch.randperm(remaining.numel(), generator=generator)[:k]]
                for idx in perm.tolist():
                    selected.add(int(idx))

        return list(selected)


class DispersedSelection(PatchSelectionStrategy):
    """Farthest-point heuristic (Θ(nN)) to spread patches across the grid."""

    def select(self, grid: PatchGrid, n: int, *, generator: Optional[torch.Generator] = None) -> List[int]:
        n = int(n)
        if n <= 0:
            return []
        n = min(n, grid.N)

        # Coordinates (N, 2) as float for distance computation.
        coords = torch.stack(
            [
                torch.arange(grid.N) // grid.P,
                torch.arange(grid.N) % grid.P,
            ],
            dim=1,
        ).float()

        seed = int(torch.randint(0, grid.N, (1,), generator=generator).item())
        selected = [seed]
        selected_mask = torch.zeros(grid.N, dtype=torch.bool)
        selected_mask[seed] = True

        # min_dist2[i] = min squared distance from i to any selected point
        seed_xy = coords[seed : seed + 1]
        min_dist2 = ((coords - seed_xy) ** 2).sum(dim=1)
        min_dist2[selected_mask] = -1.0

        for _ in range(1, n):
            idx = int(torch.argmax(min_dist2).item())
            selected.append(idx)
            selected_mask[idx] = True
            xy = coords[idx : idx + 1]
            dist2 = ((coords - xy) ** 2).sum(dim=1)
            min_dist2 = torch.minimum(min_dist2, dist2)
            min_dist2[selected_mask] = -1.0

        return selected


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    radius = int(torch.ceil(torch.tensor(3.0 * sigma)).item())
    k = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_patch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise separable Gaussian blur.

    Supports x with shape (C,H,W) or (B,C,H,W).
    Padding is reflect to avoid dark borders and avoids reading outside the patch.
    """

    if x.dim() == 3:
        x4 = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 4:
        x4 = x
        squeeze = False
    else:
        raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got shape={tuple(x.shape)}")

    b, c, h, w = x4.shape
    device, dtype = x4.device, x4.dtype
    k1d = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
    k = int(k1d.numel())
    pad = k // 2

    # Horizontal
    weight_x = k1d.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    x_pad = F.pad(x4, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(x_pad, weight_x, bias=None, stride=1, padding=0, groups=c)

    # Vertical
    weight_y = k1d.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    y_pad = F.pad(y, (0, 0, pad, pad), mode="reflect")
    y2 = F.conv2d(y_pad, weight_y, bias=None, stride=1, padding=0, groups=c)

    return y2.squeeze(0) if squeeze else y2


class DEFOCA(nn.Module):
    """Blur-to-Focus Attention Layer (training-time transform).

    - Apply *after* global augmentations (crop/flip/jitter).
    - Produces V views by selecting n patches and blurring them.
    - Do not use at test time.
    """

    def __init__(
        self,
        *,
        P: int = 4,
        ratio: float = 0.25,
        sigma: float = 1.0,
        strategy: StrategyName = "contiguous",
        V: int = 8,
        max_attempts: int = 10,
        ensure_unique: bool = True,
    ):
        super().__init__()
        self.P = int(P)
        self.ratio = float(ratio)
        self.sigma = float(sigma)
        self.strategy = strategy
        self.V = int(V)
        self.max_attempts = int(max_attempts)
        self.ensure_unique = bool(ensure_unique)

        if self.V <= 0:
            raise ValueError(f"V must be >= 1, got {self.V}")
        if not (0.0 <= self.ratio <= 1.0):
            raise ValueError(f"ratio must be in [0,1], got {self.ratio}")
        if self.max_attempts <= 0:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.strategy not in ("random", "contiguous", "dispersed"):
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if self.strategy == "random":
            self._selector: PatchSelectionStrategy = RandomSelection()
        elif self.strategy == "contiguous":
            self._selector = ContiguousSelection()
        else:
            self._selector = DispersedSelection()

    def _n_from_grid(self, grid: PatchGrid) -> int:
        n = int(round(self.ratio * grid.N))
        return max(0, min(n, grid.N))

    def _apply_single(self, x: torch.Tensor, *, generator: Optional[torch.Generator]) -> torch.Tensor:
        # x: (C,H,W)
        c, h, w = x.shape
        grid = PatchGrid(H=h, W=w, P=self.P)
        grid.validate()
        n = self._n_from_grid(grid)
        idxs = self._selector.select(grid, n, generator=generator)

        out = x.clone()
        for idx in idxs:
            y0, y1, x0, x1 = grid.bbox(int(idx))
            patch = out[:, y0:y1, x0:x1]
            out[:, y0:y1, x0:x1] = gaussian_blur_patch(patch, self.sigma)
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return views.

        Input:
          - (C,H,W) -> (V,C,H,W)
          - (B,C,H,W) -> (B,V,C,H,W)
        """

        if x.dim() == 3:
            c, h, w = x.shape
            grid = PatchGrid(H=h, W=w, P=self.P)
            grid.validate()
            used: Set[Tuple[int, ...]] = set()
            views: List[torch.Tensor] = []
            for _ in range(self.V):
                # Enforce uniqueness by controlling the selection attempts.
                if self.ensure_unique:
                    for _attempt in range(self.max_attempts):
                        n = self._n_from_grid(grid)
                        idxs = self._selector.select(grid, n, generator=generator)
                        key = tuple(sorted(int(i) for i in idxs))
                        if key not in used:
                            used.add(key)
                            break
                    # Apply using the last sampled idxs (even if duplicated after max_attempts)
                    out = x.clone()
                    for idx in idxs:
                        y0, y1, x0, x1 = grid.bbox(int(idx))
                        patch = out[:, y0:y1, x0:x1]
                        out[:, y0:y1, x0:x1] = gaussian_blur_patch(patch, self.sigma)
                    views.append(out)
                else:
                    views.append(self._apply_single(x, generator=generator))
            return torch.stack(views, dim=0)

        if x.dim() == 4:
            b, c, h, w = x.shape
            grid = PatchGrid(H=h, W=w, P=self.P)
            grid.validate()
            out = []
            for i in range(b):
                out.append(self.forward(x[i], generator=generator))
            return torch.stack(out, dim=0)

        raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got shape={tuple(x.shape)}")

    def apply_one_view(
        self,
        x: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate a single DEFOCA view: (C,H,W) -> (C,H,W).

        Uses Route A from ssl_defoca.md §3 (recommended for SSL):
          1. Blur the *whole* image once (one separable-conv call).
          2. Build a binary pixel mask from the selected patch grid.
          3. Alpha-blend: out = mask * x + (1 - mask) * x_blur.

        Compared to calling forward() and discarding V-1 views:
          - Avoids generating V views when only one is needed (V× saving).
          - Replaces n per-patch blur calls with a single full-image blur.
        """
        if x.dim() != 3:
            raise ValueError(f"apply_one_view expects (C,H,W), got {tuple(x.shape)}")

        c, h, w = x.shape
        grid = PatchGrid(H=h, W=w, P=self.P)
        grid.validate()
        n = self._n_from_grid(grid)
        if n == 0:
            return x.clone()

        idxs = self._selector.select(grid, n, generator=generator)

        # Route A: one full-image blur, then patch-mask blend.
        x_blur = gaussian_blur_patch(x, self.sigma)  # (C,H,W)

        # keep_mask: 1 = keep sharp, 0 = replace with blur.
        keep_mask = x.new_ones(1, h, w)
        for idx in idxs:
            y0, y1, x0, x1 = grid.bbox(int(idx))
            keep_mask[:, y0:y1, x0:x1] = 0.0

        return keep_mask * x + (1.0 - keep_mask) * x_blur
