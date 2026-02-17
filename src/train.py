from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from .pipeline import ClsPipeline, DefocaConfig, TrainConfig
from .ufgvc import UFGVCDataset


def build_transforms(img_size: int):
    # Global augmentations first (operate on PIL), then ToTensor.
    train_t = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_t, val_t


def try_build_split(
    *,
    dataset_name: str,
    root: str,
    split: str,
    transform,
    download: bool,
):
    try:
        return UFGVCDataset(
            dataset_name=dataset_name,
            root=root,
            split=split,
            transform=transform,
            download=download,
        )
    except ValueError:
        return None


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="UFGVC classification with DEFOCA")
    p.add_argument("--dataset", type=str, default="soybean")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=42)

    # DEFOCA
    p.add_argument("--defoca", action="store_true", help="Enable DEFOCA during training")
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--strategy", type=str, default="contiguous", choices=["random", "contiguous", "dispersed"])
    p.add_argument("--V", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--no-unique", action="store_true", help="Allow duplicate patch sets across views")

    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    train_t, val_t = build_transforms(args.img_size)
    train_ds = try_build_split(
        dataset_name=args.dataset,
        root=args.root,
        split="train",
        transform=train_t,
        download=True,
    )
    if train_ds is None:
        raise RuntimeError(f"Dataset {args.dataset} has no 'train' split")

    val_ds = try_build_split(
        dataset_name=args.dataset,
        root=args.root,
        split="val",
        transform=val_t,
        download=True,
    )
    if val_ds is None:
        val_ds = try_build_split(
            dataset_name=args.dataset,
            root=args.root,
            split="test",
            transform=val_t,
            download=True,
        )

    num_classes = len(train_ds.classes)
    model = resnet18(weights=None, num_classes=num_classes)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    defoca_cfg = DefocaConfig(
        enabled=bool(args.defoca),
        P=args.P,
        ratio=args.ratio,
        sigma=args.sigma,
        strategy=args.strategy,
        V=args.V,
        max_attempts=args.max_attempts,
        ensure_unique=not args.no_unique,
    )

    print("TrainConfig:", asdict(train_cfg))
    print("DefocaConfig:", asdict(defoca_cfg))
    print(f"num_classes={num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    pipeline = ClsPipeline(model=model, num_classes=num_classes, train_cfg=train_cfg, defoca_cfg=defoca_cfg)

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = pipeline.train_one_epoch(train_loader, epoch=epoch, generator=gen)
        print(f"epoch={epoch} train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")

        if val_loader is not None:
            val_metrics = pipeline.evaluate(val_loader)
            print(f"epoch={epoch} val   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
