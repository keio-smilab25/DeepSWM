#!/bin/bash

# scripts/batch_pretrain_inference.sh
# Usage: ./scripts/batch_pretrain_inference.sh "2025-04-01 00:00" "2025-08-12 23:00"
# Extract SparseMAE features hourly between START and END (inclusive)

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 START_DATETIME END_DATETIME [--fold 1] [--cuda -1]"
  exit 1
fi

START="$1"; shift
END="$1"; shift
FOLD=1
CUDA=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold) FOLD="$2"; shift 2;;
    --cuda) CUDA="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if ! date -d "$START" >/dev/null 2>&1; then echo "Invalid START"; exit 1; fi
if ! date -d "$END" >/dev/null 2>&1; then echo "Invalid END"; exit 1; fi
if [[ $(date -d "$START" +%s) -gt $(date -d "$END" +%s) ]]; then echo "START > END"; exit 1; fi

current="$START"

while [[ $(date -d "$current" +%s) -le $(date -d "$END" +%s) ]]; do
  ts=$(date -u -d "$current" +"%Y%m%d_%H0000")
  echo "[pretrain-batch] $current (ts=$ts)"
  python ml/pretrain.py \
    --mode inference \
    --datetime "$ts" \
    --fold "$FOLD" \
    --data_root ml/datasets \
    --cuda_device "$CUDA" \
    --pretrain_checkpoint ml/checkpoints/pretrain/ours.pth || true
  current=$(date -u -d "$current +1 hour" +"%Y-%m-%d %H:%M")
  sleep 0.1
done

echo "Done batch pretrain inference range." 
