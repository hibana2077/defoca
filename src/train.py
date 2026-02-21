from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Optional

import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .pipeline import ClsPipeline, CeaConfig, DefocaConfig, TrainConfig
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
    except (ValueError, RuntimeError):
        return None


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="UFGVC classification with DEFOCA")
    p.add_argument(
        "--task",
        type=str,
        default="supervised",
        choices=["supervised"],
        help="Only supervised training is supported (SSL/pretrain removed).",
    )
    p.add_argument("--arch", type=str, default="resnet18", help="timm backbone name")
    p.add_argument("--dataset", type=str, default="soybean")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor (only used when num_workers > 0)",
    )
    pw_group = p.add_mutually_exclusive_group()
    pw_group.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Enable DataLoader persistent_workers when num_workers > 0",
    )
    pw_group.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable DataLoader persistent_workers",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=42)

    # DEFOCA
    p.add_argument("--defoca", action="store_true", help="Enable DEFOCA during supervised training")
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--strategy", type=str, default="contiguous", choices=["random", "contiguous", "dispersed"])
    p.add_argument("--V", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--no-unique", action="store_true", help="Allow duplicate patch sets across views")

    # CEA (TIP version: evidence alignment)
    p.add_argument("--cea", action="store_true", help="Enable CEA (Consensus Evidence Alignment) loss")
    p.add_argument(
        "--cea-signal",
        type=str,
        default="gradcam",
        choices=["gradcam", "feat_norm", "input_grad"],
        help="Evidence signal source (gradcam is usually best trade-off)",
    )
    p.add_argument(
        "--cea-P",
        type=int,
        default=0,
        help="CEA evidence grid size P (0 = follow --P)",
    )
    p.add_argument("--cea-tau", type=float, default=0.2)
    p.add_argument("--cea-gamma", type=float, default=1.0)
    p.add_argument("--cea-topk", type=int, default=4)
    p.add_argument("--cea-lambda-align", type=float, default=1.0)
    p.add_argument("--cea-lambda-js", type=float, default=1.0)
    p.add_argument("--cea-lambda-iou", type=float, default=1.0)

    # CEA Version B: evidence-guided gating
    p.add_argument("--cea-gate", action="store_true", help="Enable evidence-guided gating (Version B)")
    p.add_argument("--cea-gate-alpha", type=float, default=1.0, help="Gating strength (0 disables effect)")
    p.add_argument(
        "--cea-gate-target",
        type=str,
        default="auto",
        choices=["auto", "vit", "cnn"],
        help="Where to apply gating (auto picks based on model)",
    )
    p.add_argument(
        "--cea-vit-block",
        type=int,
        default=-1,
        help="ViT block index for gating (-1 = middle)",
    )
    p.add_argument(
        "--cea-cnn-stage",
        type=str,
        default="layer3",
        help="CNN stage name for gating (e.g., layer2/layer3/layer4 for ResNet)",
    )

    args = p.parse_args(argv)

    # DataLoader behavior
    prefetch_factor = int(args.prefetch_factor)
    if prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be >= 1")

    if args.no_persistent_workers:
        persistent_workers = False
    elif args.persistent_workers:
        persistent_workers = args.num_workers > 0
    else:
        # preserve current behavior by default
        persistent_workers = args.num_workers > 0

    torch.manual_seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    train_t, val_t = build_transforms(args.img_size)
    train_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="train", transform=train_t, download=True)
    if train_ds is None:
        raise RuntimeError(f"Dataset {args.dataset} has no 'train' split")

    print(f"Found {len(train_ds)} training samples")

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

    cea_P = int(args.cea_P) if int(args.cea_P) > 0 else int(args.P)
    cea_cfg = CeaConfig(
        enabled=bool(args.cea),
        signal=str(args.cea_signal),
        P=cea_P,
        tau=float(args.cea_tau),
        gamma=float(args.cea_gamma),
        topk=int(args.cea_topk),
        lambda_align=float(args.cea_lambda_align),
        lambda_js=float(args.cea_lambda_js),
        lambda_iou=float(args.cea_lambda_iou),
        gate_enabled=bool(args.cea_gate),
        gate_alpha=float(args.cea_gate_alpha),
        gate_target=str(args.cea_gate_target),
        vit_block=int(args.cea_vit_block),
        cnn_stage=str(args.cea_cnn_stage),
    )

    print("TrainConfig:", asdict(train_cfg))
    print("DefocaConfig:", asdict(defoca_cfg))
    print("CeaConfig:", asdict(cea_cfg))
    print(f"num_classes={num_classes}")

    # ---- GPU / env debug info ----
    print(f"[DEBUG] Python={sys.version}", flush=True)
    print(f"[DEBUG] PyTorch={torch.__version__}", flush=True)
    print(f"[DEBUG] CUDA available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[DEBUG] GPU {i}: {props.name}  total_mem={props.total_memory/1024**3:.1f} GB", flush=True)
    # --------------------------------

    model = timm.create_model(args.arch, pretrained=True, num_classes=num_classes)
    pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if train_cfg.num_workers > 0 else None,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if train_cfg.num_workers > 0 else None,
        )

    pipeline = ClsPipeline(model=model, num_classes=num_classes, train_cfg=train_cfg, defoca_cfg=defoca_cfg, cea_cfg=cea_cfg)
    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = pipeline.train_one_epoch(train_loader, epoch=epoch, generator=gen)
        if bool(args.cea):
            print(
                f"epoch={epoch} train loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} align={train_metrics['align']:.4f} "
                f"js={train_metrics['js']:.4f} iou={train_metrics['iou']:.4f}"
            )
        else:
            print(f"epoch={epoch} train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")
        if val_loader is not None:
            val_metrics = pipeline.evaluate(val_loader)
            print(f"epoch={epoch} val   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
