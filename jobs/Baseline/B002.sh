#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -J 0-4
#PBS -r y
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=00:55:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

set -euo pipefail

SCRIPT_DIR='/scratch/yp87/sl5952/defoca/jobs/Baseline'
EXP_FILE="$SCRIPT_DIR/experiments.txt"

# PBS array index (0-based). Fall back to 0 when running interactively.
IDX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
LINE_NO=$((IDX + 1))

if [[ ! -f "$EXP_FILE" ]]; then
  echo "ERROR: experiments file not found: $EXP_FILE" >&2
  exit 2
fi

# Pick the Nth non-empty, non-comment line.
LINE=$(awk -v n="$LINE_NO" 'NF && $1 !~ /^#/ {i++; if (i==n) {print; exit}}' "$EXP_FILE")
if [[ -z "${LINE:-}" ]]; then
  echo "ERROR: No experiment line $LINE_NO (IDX=$IDX) in $EXP_FILE" >&2
  exit 3
fi

SEED=$(echo "$LINE" | awk '{print $1}')

if [[ -z "${SEED:-}" ]]; then
  echo "ERROR: Bad experiment line $LINE_NO in $EXP_FILE (expected: <seed>): $LINE" >&2
  exit 4
fi

source /scratch/yp87/sl5952/defoca/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..

LOG_DIR="logs/Baseline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/B002_idx${IDX}_seed${SEED}.log"

{
  echo "===== DEFOCA job start ====="
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "IDX=$IDX LINE_NO=$LINE_NO"
  echo "EXPERIMENT=$LINE"
  echo "SEED=$SEED"
  echo "LOG_FILE=$LOG_FILE"
  echo "==========================="
} >> "$LOG_FILE"

python3 -u -m src.train \
  --task pretrain \
  --ssl-method simclr \
  --dataset cifar100 --root ./data \
  --arch resnet18 \
  --img-size 224 \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --lr 3e-4 --weight-decay 1e-4 \
  --ssl-proj-dim 128 --ssl-hidden-dim 2048 --ssl-temperature 0.2 \
  --linear-epochs 20 --linear-lr 1e-2 --knn-k 20 --knn-t 0.1 \
  --seed "$SEED" --device cuda \
  >> "$LOG_FILE" 2>&1