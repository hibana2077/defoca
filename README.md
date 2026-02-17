# defoca

This repo contains a minimal classification pipeline for UFGVC datasets with **DEFOCA (Blur-to-Focus Attention)** as a *training-time* augmentation.

## Run

Train baseline (no DEFOCA):

`python -m src.train --dataset soybean --root ./data --epochs 10`

Train with DEFOCA (multi-view, train-only):

`python -m src.train --dataset soybean --root ./data --defoca --P 4 --ratio 0.25 --sigma 1.0 --strategy contiguous --V 8`

Notes:

- Global augmentations (resize/flip/jitter) are applied first, then DEFOCA patch blur, then normalization.
- Evaluation does **not** use DEFOCA.
