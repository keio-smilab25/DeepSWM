#!/bin/bash

# scripts/batch_main_inference.sh
# Usage: ./scripts/batch_main_inference.sh "2025-04-01 00:00" "2025-08-12 23:00"
# Run main model inference hourly between START and END (inclusive), writing pred_24.json cumulatively

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 START_DATETIME END_DATETIME [--fold 1] [--cuda -1] [--history 4] [--ckpt ml/checkpoints/main/DeepSWM.pth]"
  exit 1
fi

START="$1"; shift
END="$1"; shift
FOLD=1
CUDA=-1
HISTORY=4
CKPT="ml/checkpoints/main/DeepSWM.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold) FOLD="$2"; shift 2;;
    --cuda) CUDA="$2"; shift 2;;
    --history) HISTORY="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if ! date -d "$START" >/dev/null 2>&1; then echo "Invalid START"; exit 1; fi
if ! date -d "$END" >/dev/null 2>&1; then echo "Invalid END"; exit 1; fi
if [[ $(date -d "$START" +%s) -gt $(date -d "$END" +%s) ]]; then echo "START > END"; exit 1; fi

current="$START"

while [[ $(date -d "$current" +%s) -le $(date -d "$END" +%s) ]]; do
  ts_full=$(date -u -d "$current" +"%Y%m%d_%H0000")
  echo "[main-batch] $current (ts=$ts_full)"
  python ml/main.py \
    --params ml/params/main/params.yaml \
    --trial_name batch \
    --fold "$FOLD" \
    --data_root ml/datasets \
    --cuda_device "$CUDA" \
    --history "$HISTORY" \
    --mode inference \
    --resume_from_checkpoint "$CKPT" \
    --datetime "$ts_full" || true
  current=$(date -u -d "$current +1 hour" +"%Y-%m-%d %H:%M")
  sleep 0.1
done

echo "Done batch main inference range." 
