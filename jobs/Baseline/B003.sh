#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/defoca/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..

mkdir -p logs/Baseline

python3 -m src.train \
  --task pretrain \
  --ssl-method swav \
  --dataset soybean --root ./data \
  --arch resnet18 \
  --img-size 224 \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 4 \
  --lr 3e-4 --weight-decay 1e-4 \
  --ssl-hidden-dim 2048 \
  --swav-feat-dim 128 \
  --swav-n-prototypes 3000 \
  --swav-temperature 0.1 \
  --swav-epsilon 0.05 \
  --swav-sinkhorn-iters 3 \
  --swav-crops-for-assign 0 1 \
  --swav-nmb-crops 2 \
  --swav-size-crops 224 \
  --swav-min-scale-crops 0.14 \
  --swav-max-scale-crops 1.0 \
  --linear-epochs 20 --linear-lr 1e-2 --knn-k 20 --knn-t 0.1 \
  --seed 42 --device cuda \
  >> logs/Baseline/B003.log 2>&1