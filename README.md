# defoca

This repo contains a minimal classification pipeline for UFGVC datasets with **DEFOCA (Blur-to-Focus Attention)** as a *training-time* augmentation.

## Run

ViT (DeiT-S)

`python3 -m src.train --dataset cub_200_2011 --arch deit_small_patch16_224 --defoca --V 8 --P 4 --cea --cea-signal gradcam --cea-gate --cea-gate-target vit --cea-vit-block -1`

ResNet18

`python3 -m src.train --dataset cub_200_2011 --arch resnet18 --defoca --V 8 --P 4 --cea --cea-signal gradcam --cea-gate --cea-gate-target cnn --cea-cnn-stage layer3`

ResNet50

`python3 -m src.train --dataset cub_200_2011 --arch resnet50 --defoca --V 8 --P 4 --cea --cea-signal gradcam --cea-gate --cea-gate-target cnn --cea-cnn-stage layer3`

Notes:

- Global augmentations (resize/flip/jitter) are applied first, then DEFOCA patch blur, then normalization.
- Evaluation does **not** use DEFOCA.
