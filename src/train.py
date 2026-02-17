from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

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
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ]
    )

    if defoca_cfg is None or not defoca_cfg.enabled:
        return base

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
    except ValueError:
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

    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    train_t, val_t = build_transforms(args.img_size)
    train_ds = try_build_split(dataset_name=args.dataset, root=args.root, split="train", transform=train_t, download=True)
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

    if args.task == "supervised":
        model = timm.create_model(args.arch, pretrained=False, num_classes=num_classes)
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

    encoder = TimmEncoder(args.arch, pretrained=False)
    if args.ssl_method == "simclr":
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
        ssl_ds = TwoCropDataset(base_ssl_ds, t1=t1, t2=t2)
        method = SimCLR(
            encoder=encoder,
            feature_dim=encoder.out_dim,
            proj_dim=args.ssl_proj_dim,
            hidden_dim=args.ssl_hidden_dim,
            temperature=args.ssl_temperature,
        )
    elif args.ssl_method == "vicreg":
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
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
        t1 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
        t2 = build_ssl_transforms(args.img_size, defoca_cfg=defoca_cfg, view_index=None)
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
        if defoca_cfg.enabled:
            defoca = DEFOCA(
                P=defoca_cfg.P,
                ratio=defoca_cfg.ratio,
                sigma=defoca_cfg.sigma,
                strategy=defoca_cfg.strategy,  # type: ignore[arg-type]
                V=defoca_cfg.V,
                max_attempts=defoca_cfg.max_attempts,
                ensure_unique=defoca_cfg.ensure_unique,
            )
            pick = DefocaPickView(defoca, view_index=None)

        tx = []
        for size, min_s, max_s in zip(args.swav_size_crops, args.swav_min_scale_crops, args.swav_max_scale_crops):
            t_list = [
                transforms.RandomResizedCrop(int(size), scale=(float(min_s), float(max_s))),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
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

    pre_loader = DataLoader(
        ssl_ds,
        batch_size=pre_cfg.batch_size,
        shuffle=True,
        num_workers=pre_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    pre_pipeline = PretrainPipeline(method=method, pretrain_cfg=pre_cfg)
    for epoch in range(1, pre_cfg.epochs + 1):
        m = pre_pipeline.train_one_epoch(pre_loader, epoch=epoch)
        print(f"epoch={epoch} pretrain loss={m['loss']:.4f}")

    # ============ evaluation ============
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

    train_eval_loader = DataLoader(
        eval_train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_eval_loader = DataLoader(
        eval_val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    evaluator = PretrainEvaluator(encoder=method.encoder, device=args.device)
    lin = evaluator.linear_eval(train_loader=train_eval_loader, val_loader=val_eval_loader, num_classes=num_classes, cfg=eval_cfg)
    knn = evaluator.knn_eval(train_loader=train_eval_loader, val_loader=val_eval_loader, cfg=eval_cfg)
    clu = evaluator.clustering_eval(loader=val_eval_loader, num_classes=num_classes)
    print("eval:", {**lin, **knn, **clu})


if __name__ == "__main__":
    main()
