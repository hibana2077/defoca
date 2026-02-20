from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

from .defoca import DEFOCA
from .pipeline import ClsPipeline, DefocaConfig, PretrainConfig, TrainConfig
from .cl.ssl_datasets import DefocaPickView, MultiCropDataset, TwoCropDataset
from .cl.ssl_eval import EvalConfig, PretrainEvaluator
from .cl.ssl_methods import BarlowTwins, SimCLR, SwAV, VICReg
from .cl.ssl_models import ResNet18Encoder, SwAVConfig, TimmEncoder
from .pipeline import PretrainPipeline
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


def build_ssl_transforms(img_size: int, *, defoca_cfg: DefocaConfig | None, view_index: int | None = None):
    base = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
        ]
    )

    print(f"[DEBUG build_ssl_transforms] defoca_cfg={defoca_cfg}", flush=True)
    if defoca_cfg is None or not defoca_cfg.enabled:
        print("[DEBUG build_ssl_transforms] DEFOCA disabled -> returning base transform only", flush=True)
        return base

    print("[DEBUG build_ssl_transforms] DEFOCA ENABLED -> appending DefocaPickView", flush=True)
    defoca = DEFOCA(
        P=defoca_cfg.P,
        ratio=defoca_cfg.ratio,
        sigma=defoca_cfg.sigma,
        strategy=defoca_cfg.strategy,  # type: ignore[arg-type]
        V=defoca_cfg.V,
        max_attempts=defoca_cfg.max_attempts,
        ensure_unique=defoca_cfg.ensure_unique,
    )
    pick = DefocaPickView(defoca, view_index=view_index)
    return transforms.Compose([base, pick])


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
    p.add_argument("--task", type=str, default="supervised", choices=["supervised", "pretrain"])
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
    p.add_argument(
        "--defoca-ssl",
        action="store_true",
        help="Enable DEFOCA inside SSL augmentations (uses random patch selection)",
    )
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--strategy", type=str, default="contiguous", choices=["random", "contiguous", "dispersed"])
    p.add_argument("--V", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--no-unique", action="store_true", help="Allow duplicate patch sets across views")

    # CL pretrain
    p.add_argument("--ssl-method", type=str, default="simclr", choices=["simclr", "vicreg", "barlow", "swav"])
    p.add_argument("--ssl-proj-dim", type=int, default=128)
    p.add_argument("--ssl-hidden-dim", type=int, default=2048)
    p.add_argument("--ssl-temperature", type=float, default=0.2)
    p.add_argument("--ssl-lambd", type=float, default=0.005)
    p.add_argument("--ssl-sim", type=float, default=25.0)
    p.add_argument("--ssl-std", type=float, default=25.0)
    p.add_argument("--ssl-cov", type=float, default=1.0)

    # SwAV (minimal multi-crop)
    p.add_argument("--swav-feat-dim", type=int, default=128)
    p.add_argument("--swav-n-prototypes", type=int, default=3000)
    p.add_argument("--swav-temperature", type=float, default=0.1)
    p.add_argument("--swav-epsilon", type=float, default=0.05)
    p.add_argument("--swav-sinkhorn-iters", type=int, default=3)
    p.add_argument("--swav-crops-for-assign", type=int, nargs="+", default=[0, 1])
    p.add_argument("--swav-nmb-crops", type=int, nargs="+", default=[2])
    p.add_argument("--swav-size-crops", type=int, nargs="+", default=[224])
    p.add_argument("--swav-min-scale-crops", type=float, nargs="+", default=[0.14])
    p.add_argument("--swav-max-scale-crops", type=float, nargs="+", default=[1.0])

    # Pretrain eval
    p.add_argument("--linear-epochs", type=int, default=20)
    p.add_argument("--linear-lr", type=float, default=1e-2)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-t", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=5,
                   help="Run linear/kNN/clustering eval every N epochs during SSL pretraining (0 = end only)")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for periodic SSL evaluation (default: same as --batch-size)",
    )
    p.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="Num workers for periodic SSL evaluation (default: 0 to reduce RAM spikes)",
    )
    p.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=2,
        help="Prefetch factor for periodic SSL evaluation (only used when eval-num-workers > 0)",
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
    defoca_ssl_cfg = DefocaConfig(
        enabled=bool(args.defoca_ssl),
        P=args.P,
        ratio=args.ratio,
        sigma=args.sigma,
        strategy="random",
        V=args.V,
        max_attempts=args.max_attempts,
        ensure_unique=not args.no_unique,
    )

    print("TrainConfig:", asdict(train_cfg))
    print("DefocaConfig:", asdict(defoca_cfg))
    print("DefocaSSLConfig:", asdict(defoca_ssl_cfg))
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

    if args.task == "supervised":
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

        pipeline = ClsPipeline(model=model, num_classes=num_classes, train_cfg=train_cfg, defoca_cfg=defoca_cfg)
        for epoch in range(1, train_cfg.epochs + 1):
            train_metrics = pipeline.train_one_epoch(train_loader, epoch=epoch, generator=gen)
            print(f"epoch={epoch} train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")
            if val_loader is not None:
                val_metrics = pipeline.evaluate(val_loader)
                print(f"epoch={epoch} val   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")
        return

    # ============ self-supervised pretraining ============
    pre_cfg = PretrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    base_ssl_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="train", transform=None, download=True)
    if base_ssl_ds is None:
        raise RuntimeError(f"Dataset {args.dataset} has no 'train' split")

    print(f"[DEBUG] Building encoder: {args.arch}", flush=True)
    encoder = TimmEncoder(args.arch, pretrained=False)
    print(f"[DEBUG] encoder.out_dim={encoder.out_dim}", flush=True)
    if args.ssl_method == "simclr":
        print("[DEBUG] building SimCLR transforms (t1, t2)...", flush=True)
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        print(f"[DEBUG] t1 pipeline:\n{t1}", flush=True)
        print(f"[DEBUG] t2 pipeline:\n{t2}", flush=True)
        ssl_ds = TwoCropDataset(base_ssl_ds, t1=t1, t2=t2)
        method = SimCLR(
            encoder=encoder,
            feature_dim=encoder.out_dim,
            proj_dim=args.ssl_proj_dim,
            hidden_dim=args.ssl_hidden_dim,
            temperature=args.ssl_temperature,
        )
    elif args.ssl_method == "vicreg":
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        ssl_ds = TwoCropDataset(base_ssl_ds, t1=t1, t2=t2)
        method = VICReg(
            encoder=encoder,
            feature_dim=encoder.out_dim,
            proj_dim=max(256, args.ssl_proj_dim),
            hidden_dim=max(256, args.ssl_hidden_dim),
            sim_coeff=args.ssl_sim,
            std_coeff=args.ssl_std,
            cov_coeff=args.ssl_cov,
        )
    elif args.ssl_method == "barlow":
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_ssl_cfg, view_index=None)
        ssl_ds = TwoCropDataset(base_ssl_ds, t1=t1, t2=t2)
        method = BarlowTwins(
            encoder=encoder,
            feature_dim=encoder.out_dim,
            proj_dim=max(256, args.ssl_proj_dim),
            hidden_dim=max(256, args.ssl_hidden_dim),
            lambd=args.ssl_lambd,
        )
    else:
        if not (len(args.swav_nmb_crops) == len(args.swav_size_crops) == len(args.swav_min_scale_crops) == len(args.swav_max_scale_crops)):
            raise ValueError("SwAV crop args must have same length")

        pick = None
        if defoca_ssl_cfg.enabled:
            defoca = DEFOCA(
                P=defoca_ssl_cfg.P,
                ratio=defoca_ssl_cfg.ratio,
                sigma=defoca_ssl_cfg.sigma,
                strategy=defoca_ssl_cfg.strategy,  # type: ignore[arg-type]
                V=defoca_ssl_cfg.V,
                max_attempts=defoca_ssl_cfg.max_attempts,
                ensure_unique=defoca_ssl_cfg.ensure_unique,
            )
            pick = DefocaPickView(defoca, view_index=None)

        tx = []
        for size, min_s, max_s in zip(args.swav_size_crops, args.swav_min_scale_crops, args.swav_max_scale_crops):
            t_list = [
                transforms.RandomResizedCrop(int(size), scale=(float(min_s), float(max_s))),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
            ]
            if pick is not None:
                t_list.append(pick)
            tx.append(
                transforms.Compose(t_list)
            )
        ssl_ds = MultiCropDataset(base_ssl_ds, transforms=tx, nmb_crops=args.swav_nmb_crops)
        swcfg = SwAVConfig(
            feat_dim=args.swav_feat_dim,
            n_prototypes=args.swav_n_prototypes,
            temperature=args.swav_temperature,
            epsilon=args.swav_epsilon,
            sinkhorn_iterations=args.swav_sinkhorn_iters,
            crops_for_assign=tuple(args.swav_crops_for_assign),
        )
        method = SwAV(encoder=encoder, feature_dim=encoder.out_dim, cfg=swcfg, hidden_dim=args.ssl_hidden_dim)

    print(f"[DEBUG] ssl_ds type={type(ssl_ds).__name__}  len={len(ssl_ds)}", flush=True)
    print(
        f"[DEBUG] DataLoader: batch_size={pre_cfg.batch_size}  num_workers={pre_cfg.num_workers}  "
        f"prefetch_factor={prefetch_factor if pre_cfg.num_workers > 0 else None}  "
        f"persistent_workers={persistent_workers}",
        flush=True,
    )
    pin_memory = bool(torch.cuda.is_available())
    pre_loader = DataLoader(
        ssl_ds,
        batch_size=pre_cfg.batch_size,
        shuffle=True,
        num_workers=pre_cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if pre_cfg.num_workers > 0 else None,
    )

    # ---- DataLoader timing benchmark (first 5 batches) ----
    print("[DEBUG] Benchmarking DataLoader: timing first 5 batches...", flush=True)
    _t0 = time.perf_counter()
    for _i, (_views, _) in enumerate(pre_loader):
        _t1 = time.perf_counter()
        _v0 = _views[0] if isinstance(_views, (list, tuple)) else _views
        print(
            f"[DEBUG] batch {_i}: load_time={_t1 - _t0:.3f}s  "
            f"view0_shape={tuple(_v0.shape)}  "
            f"view0_dtype={_v0.dtype}",
            flush=True,
        )
        _t0 = _t1
        if _i >= 4:
            break
    print("[DEBUG] DataLoader benchmark done.", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU mem after data loading: "
              f"allocated={torch.cuda.memory_allocated()/1024**2:.1f} MB  "
              f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB", flush=True)
    # -------------------------------------------------------

    # ============ evaluation setup (built once, reused every eval_interval epochs) ============
    eval_cfg = EvalConfig(
        linear_epochs=args.linear_epochs,
        linear_lr=args.linear_lr,
        knn_k=args.knn_k,
        knn_temperature=args.knn_t,
    )
    eval_train_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="train", transform=val_t, download=True)
    eval_val_ds = val_ds
    if eval_val_ds is None:
        eval_val_ds = eval_train_ds

    eval_batch_size = int(args.batch_size if args.eval_batch_size is None else args.eval_batch_size)
    eval_num_workers = int(args.eval_num_workers)
    eval_prefetch_factor = int(args.eval_prefetch_factor)
    eval_pin_memory = bool(torch.cuda.is_available())

    train_eval_loader = DataLoader(
        eval_train_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=eval_prefetch_factor if eval_num_workers > 0 else None,
    )
    val_eval_loader = DataLoader(
        eval_val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=eval_prefetch_factor if eval_num_workers > 0 else None,
    )
    evaluator = PretrainEvaluator(encoder=method.encoder, device=args.device)

    def _run_eval(epoch: int) -> None:
        lin = evaluator.linear_eval(train_loader=train_eval_loader, val_loader=val_eval_loader, num_classes=num_classes, cfg=eval_cfg)
        knn = evaluator.knn_eval(train_loader=train_eval_loader, val_loader=val_eval_loader, cfg=eval_cfg)
        clu = evaluator.clustering_eval(loader=val_eval_loader, num_classes=num_classes)
        print(f"epoch={epoch} eval:", {**lin, **knn, **clu})

    pre_pipeline = PretrainPipeline(method=method, pretrain_cfg=pre_cfg)
    print(f"[DEBUG] Method moved to device. Starting pretrain loop.", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU mem after model init: "
              f"allocated={torch.cuda.memory_allocated()/1024**2:.1f} MB  "
              f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB", flush=True)

    eval_interval = args.eval_interval
    last_eval_epoch = 0
    for epoch in range(1, pre_cfg.epochs + 1):
        m = pre_pipeline.train_one_epoch(pre_loader, epoch=epoch)
        print(f"epoch={epoch} pretrain loss={m['loss']:.4f}")
        if eval_interval > 0 and epoch % eval_interval == 0:
            _run_eval(epoch)
            last_eval_epoch = epoch

    # always evaluate at the final epoch if not already done
    if last_eval_epoch != pre_cfg.epochs:
        _run_eval(pre_cfg.epochs)


if __name__ == "__main__":
    main()
