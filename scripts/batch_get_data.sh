#!/bin/bash

# scripts/batch_get_data.sh
# Usage: ./scripts/batch_get_data.sh "2025-04-01 00:00" "2025-08-12 23:00"
# Get (fetch) data hourly between START and END (inclusive) using data/get_data.py

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 START_DATETIME END_DATETIME"
  echo "Example: $0 \"2025-04-01 00:00\" \"2025-08-12 23:00\""
  exit 1
fi

START="$1"
END="$2"

# Parse to UTC epoch seconds
if ! START_TS=$(date -u -d "$START" +%s 2>/dev/null); then echo "Invalid START: $START"; exit 1; fi
if ! END_TS=$(date -u -d "$END" +%s 2>/dev/null); then echo "Invalid END: $END"; exit 1; fi
if [[ $START_TS -gt $END_TS ]]; then echo "START must be <= END"; exit 1; fi

# Ensure required Python deps for data/get_data.py are available
PYTHON_BIN=${PYTHON:-python}
if ! $PYTHON_BIN - <<'PY' 2>/dev/null
try:
    import astropy, requests, numpy, cv2, scipy, h5py
    from dateutil import tz  # python-dateutil
    import matplotlib.pyplot as plt  # noqa: F401
except Exception as e:
    raise SystemExit(1)
PY
then
  echo "Installing required Python packages for data fetching..." >&2
  $PYTHON_BIN -m pip install --quiet --upgrade pip
  $PYTHON_BIN -m pip install --quiet astropy requests numpy scipy h5py python-dateutil opencv-python-headless matplotlib
fi

# Iterate by hour using epoch arithmetic
TS=$START_TS
while [[ $TS -le $END_TS ]]; do
  human=$(date -u -d "@$TS" +"%Y-%m-%d %H:%M")
  mmddhh=$(date -u -d "@$TS" +"%m%d%H")
  echo "[get-data] $human (MMDDHH=$mmddhh)"
  $PYTHON_BIN data/get_data.py "$mmddhh" || true
  TS=$((TS + 3600))
  sleep 0.1
done

echo "Done batch get-data range." 
