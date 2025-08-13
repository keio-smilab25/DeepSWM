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
  ts=$(date -d "$current" +"%Y%m%d_%H0000")
  echo "[pretrain-batch] $current (ts=$ts)"
  
  if python ml/pretrain.py \
    --mode inference \
    --datetime "$ts" \
    --fold "$FOLD" \
    --data_root ml/datasets \
    --cuda_device "$CUDA" \
    --pretrain_checkpoint ml/checkpoints/pretrain/SparseMAE.pth; then
    echo "✅ Successfully processed $current"
  else
    echo "⚠️  Failed to process $current (continuing...)"
  fi
  
  # Use Python to reliably increment the time by 1 hour
  current=$(python3 -c "
from datetime import datetime, timedelta
dt = datetime.strptime('$current', '%Y-%m-%d %H:%M')
dt += timedelta(hours=1)
print(dt.strftime('%Y-%m-%d %H:%M'))
")
  sleep 0.1
done

echo "✅ Batch pretrain inference completed successfully for range: $START to $END" 
