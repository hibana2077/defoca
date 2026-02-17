from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch.utils.data import Dataset


class TwoCropDataset(Dataset):
    def __init__(self, base: Dataset, *, t1: Callable, t2: Callable):
        self.base = base
        self.t1 = t1
        self.t2 = t2

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        return (self.t1(img), self.t2(img)), y


class MultiCropDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        transforms: Sequence[Callable],
        nmb_crops: Sequence[int],
    ):
        if len(transforms) != len(nmb_crops):
            raise ValueError("transforms and nmb_crops must have same length")
        self.base = base
        self.transforms = list(transforms)
        self.nmb_crops = [int(n) for n in nmb_crops]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        crops = []
        for t, n in zip(self.transforms, self.nmb_crops):
            crops.extend([t(img) for _ in range(n)])
        return crops, y


class DefocaPickView:
    def __init__(self, defoca_module, *, view_index: Optional[int] = None):
        self.defoca = defoca_module
        self.view_index = view_index

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        views = self.defoca(x.unsqueeze(0)).squeeze(0)  # (V,C,H,W)
        if self.view_index is not None:
            return views[int(self.view_index)]
        j = int(torch.randint(0, views.size(0), (1,)).item())
        return views[j]
