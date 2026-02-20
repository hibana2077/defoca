from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

from src.ufgvc import UFGVCDataset
from src.snr_utils import (
    SNRConfig,
    SNRResult,
    aggregate_results,
    compute_saliency_absgrad,
    compute_snr_from_image,
    compute_snr_prime,
    saliency_to_patch_set,
    save_json,
)


def build_dataset(*, dataset: str, root: str, split: str, img_size: int) -> UFGVCDataset:
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return UFGVCDataset(dataset_name=dataset, root=root, split=split, transform=tfm, download=True)


def build_model(*, arch: str, pretrained: bool) -> torch.nn.Module:
    # num_classes left as default (usually 1000) for pretrained ImageNet.
    model = timm.create_model(arch, pretrained=pretrained)
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compute discriminative SNR (Eq.8) using saliency map")

    p.add_argument("--dataset", type=str, default="soybean")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    p.add_argument("--arch", type=str, default="resnet18")
    p.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet pretrained weights")

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--topk", type=float, default=0.25, help="Top-k ratio of patches as discriminative set S")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma for non-discriminative patches")

    p.add_argument("--energy", type=str, default="total", choices=["total", "highfreq"])
    p.add_argument("--cutoff-ratio", type=float, default=0.5, help="Only used when --energy highfreq")

    p.add_argument(
        "--saliency-target",
        type=str,
        default="predicted",
        choices=["predicted", "label", "index"],
        help="Which class logit to backprop for saliency",
    )
    p.add_argument("--target-index", type=int, default=0, help="Only used when --saliency-target index")

    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", type=str, default="")

    args = p.parse_args(argv)

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    cfg = SNRConfig(
        img_size=int(args.img_size),
        P=int(args.P),
        topk_ratio=float(args.topk),
        sigma=float(args.sigma),
        energy=args.energy,
        cutoff_ratio=float(args.cutoff_ratio),
        saliency_target=args.saliency_target,
        target_index=int(args.target_index),
    )

    ds = build_dataset(dataset=args.dataset, root=args.root, split=args.split, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(args.num_workers))

    model = build_model(arch=args.arch, pretrained=(not args.no_pretrained)).to(device)

    results: List[SNRResult] = []

    for i, batch in enumerate(loader):
        if i >= int(args.n_samples):
            break

        image, label = batch
        image = image[0]  # (C,H,W)
        label_i = int(label[0].item())

        # Saliency -> discriminative set S
        saliency_hw, target = compute_saliency_absgrad(
            model=model,
            image=image,
            label=label_i,
            cfg=cfg,
            device=device,
        )
        S = saliency_to_patch_set(saliency_hw=saliency_hw, P=cfg.P, topk_ratio=cfg.topk_ratio)

        # Baseline SNR
        base = compute_snr_from_image(image=image, S=S, cfg=cfg)
        # DEFOCA-style SNR' (blur non-discriminative patches)
        prime = compute_snr_prime(image=image, S=S, cfg=cfg)

        snr = float(base["snr"])
        snr_prime = float(prime["snr_prime"])
        results.append(
            SNRResult(
                snr=snr,
                snr_prime=snr_prime,
                delta=snr_prime - snr,
                ratio=(snr_prime / max(1e-12, snr)),
                target=int(target),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == int(args.n_samples):
            print(
                f"[{i+1}/{int(args.n_samples)}] snr={snr:.4f} snr'={snr_prime:.4f} Δ={snr_prime-snr:+.4f}",
                flush=True,
            )

    summary = aggregate_results(results)
    payload: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "root": args.root,
            "split": args.split,
            "arch": args.arch,
            "pretrained": (not args.no_pretrained),
            "device": str(device),
            "seed": int(args.seed),
        },
        "cfg": asdict(cfg),
        "summary": summary,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"snr_{args.dataset}_{args.split}_{stamp}.json"

    save_json(out_path, payload)

    print("\nSummary:")
    print(f"  n={summary['n']}")
    print(f"  snr mean±std = {summary['snr']['mean']:.4f} ± {summary['snr']['std']:.4f}")
    print(f"  snr' mean±std = {summary['snr_prime']['mean']:.4f} ± {summary['snr_prime']['std']:.4f}")
    print(f"  Δ mean±std = {summary['delta']['mean']:+.4f} ± {summary['delta']['std']:.4f}")
    if 'ttest_rel' in summary:
        print(f"  paired t-test p={summary['ttest_rel']['pvalue']:.3e}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
