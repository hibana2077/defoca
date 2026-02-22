from __future__ import annotations

import argparse
import json
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        help="Only supervised training is supported.",
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

    p.add_argument(
        "--metrics-json",
        type=str,
        default="best_test.json",
        help="Write best-test summary JSON to this path (only when test split exists).",
    )

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
    p.add_argument("--cea-k", type=float, default=1.0, help="CEA overall strength (multiplies alignment loss)")

    # CEA Version B: evidence-guided gating
    p.add_argument("--cea-gate", action="store_true", help="Enable evidence-guided gating (Version B; implies --cea)")
    p.add_argument("--ceag-k", type=float, default=1.0, help="CEAG gating strength (0 disables effect)")

    args = p.parse_args(argv)

    if float(args.cea_k) < 0.0:
        raise ValueError("--cea-k must be >= 0")
    if float(args.ceag_k) < 0.0:
        raise ValueError("--ceag-k must be >= 0")

    # CEAG uses the same evidence signal as CEA, so gating implies CEA.
    if bool(args.cea_gate) and (not bool(args.cea)):
        args.cea = True

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

    val_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="val", transform=val_t, download=True)
    test_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="test", transform=val_t, download=True)

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

    # ---- CEA/CEAG slim interface: fixed defaults + derived values ----
    # Evidence signal + grid are fixed to match paper-friendly defaults.
    cea_P = int(args.P)
    cea_topk = max(1, int(round((cea_P * cea_P) * float(args.ratio))))
    cea_cfg = CeaConfig(
        enabled=bool(args.cea),
        signal="gradcam",
        P=cea_P,
        tau=0.2,
        gamma=1.0,
        topk=int(cea_topk),
        # One knob: multiply overall alignment loss.
        lambda_align=float(args.cea_k),
        lambda_js=1.0,
        lambda_iou=1.0,
        gate_enabled=bool(args.cea_gate),
        gate_alpha=float(args.ceag_k),
        gate_target="auto",
        vit_block=-1,
        cnn_stage="layer3",
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

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if train_cfg.num_workers > 0 else None,
        )

    # Per-epoch eval set: prefer test, fallback to val.
    eval_name = None
    eval_loader = None
    if test_loader is not None:
        eval_name = "test"
        eval_loader = test_loader
    elif val_loader is not None:
        eval_name = "val"
        eval_loader = val_loader

    pipeline = ClsPipeline(model=model, num_classes=num_classes, train_cfg=train_cfg, defoca_cfg=defoca_cfg, cea_cfg=cea_cfg)

    best_test_acc = float("-inf")
    best_test_epoch = -1
    best_test_metrics = None

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

        if eval_loader is not None and eval_name is not None:
            eval_metrics = pipeline.evaluate(eval_loader)
            if bool(args.cea):
                print(
                    f"epoch={epoch} {eval_name:4s} loss={eval_metrics['loss']:.4f} acc={eval_metrics['acc']:.4f} "
                    f"align={eval_metrics['align']:.4f} js={eval_metrics['js']:.4f} iou={eval_metrics['iou']:.4f}"
                )
            else:
                print(f"epoch={epoch} {eval_name:4s} loss={eval_metrics['loss']:.4f} acc={eval_metrics['acc']:.4f}")

            # Track best test metrics (only meaningful when test split exists).
            if test_loader is not None and eval_name == "test":
                acc = float(eval_metrics["acc"])
                if acc > best_test_acc:
                    best_test_acc = acc
                    best_test_epoch = int(epoch)
                    best_test_metrics = dict(eval_metrics)

                if best_test_metrics is not None:
                    if bool(args.cea):
                        print(
                            f"best_test@epoch={best_test_epoch} acc={best_test_metrics['acc']:.4f} "
                            f"loss={best_test_metrics['loss']:.4f} align={best_test_metrics['align']:.4f} "
                            f"js={best_test_metrics['js']:.4f} iou={best_test_metrics['iou']:.4f}"
                        )
                    else:
                        print(
                            f"best_test@epoch={best_test_epoch} acc={best_test_metrics['acc']:.4f} "
                            f"loss={best_test_metrics['loss']:.4f}"
                        )

    # ---- end-of-run summary + JSON ----
    if test_loader is not None and best_test_metrics is not None:
        if bool(args.cea):
            print(
                f"[SUMMARY] best_test@epoch={best_test_epoch} "
                f"acc={best_test_metrics['acc']:.4f} loss={best_test_metrics['loss']:.4f} "
                f"align={best_test_metrics['align']:.4f} js={best_test_metrics['js']:.4f} iou={best_test_metrics['iou']:.4f}"
            )
        else:
            print(
                f"[SUMMARY] best_test@epoch={best_test_epoch} "
                f"acc={best_test_metrics['acc']:.4f} loss={best_test_metrics['loss']:.4f}"
            )

        payload = {
            "best_test_epoch": int(best_test_epoch),
            "best_test": {k: float(v) for k, v in best_test_metrics.items()},
            "arch": str(args.arch),
            "dataset": str(args.dataset),
            "num_classes": int(num_classes),
            "train_cfg": asdict(train_cfg),
            "defoca_cfg": asdict(defoca_cfg),
            "cea_cfg": asdict(cea_cfg),
        }

        out_path = str(args.metrics_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] wrote {out_path}")
    else:
        print("[SUMMARY] no test split found; skip best_test JSON")


if __name__ == "__main__":
    main()
