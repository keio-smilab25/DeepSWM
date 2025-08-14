#!/bin/bash

# Inference execution script with error handling
# This script ensures workflow continues even if individual steps fail

set +e  # Don't stop script on errors

# Log file configuration
LOG_FILE="inference_log_$(date +%Y%m%d_%H%M).txt"
echo "Starting inference execution at $(date)" | tee "$LOG_FILE"

# Error counter
ERROR_COUNT=0

# Data fetching function
fetch_data() {
    local datetime=$1
    local step_name=$2
    
    echo "[$step_name] Fetching data for: $datetime" | tee -a "$LOG_FILE"
    python data/get_data.py "$datetime" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$step_name] Warning: Data fetch failed for $datetime" | tee -a "$LOG_FILE"
        ((ERROR_COUNT++))
        return 1
    else
        echo "[$step_name] Data fetch successful for $datetime" | tee -a "$LOG_FILE"
        return 0
    fi
}

# Feature extraction function
extract_features() {
    local datetime=$1
    
    echo "[FEATURES] Extracting features for: $datetime" | tee -a "$LOG_FILE"
    python ml/pretrain.py \
        --mode inference \
        --datetime "$datetime" \
        --fold 1 \
        --data_root ml/datasets \
        --cuda_device -1 \
        --pretrain_checkpoint ml/checkpoints/pretrain/SparseMAE.pth \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[FEATURES] Warning: Feature extraction failed" | tee -a "$LOG_FILE"
        ((ERROR_COUNT++))
        return 1
    else
        echo "[FEATURES] Feature extraction successful" | tee -a "$LOG_FILE"
        return 0
    fi
}

# Inference execution function
run_inference() {
    local datetime=$1
    
    echo "[INFERENCE] Running inference for: $datetime" | tee -a "$LOG_FILE"
    python ml/main.py \
        --params params/main/params.yaml \
        --fold 2 \
        --data_root ./ml/datasets \
        --cuda_device -1 \
        --history 4 \
        --trial_name 090 \
        --mode inference \
        --resume_from_checkpoint ./ml/checkpoints/main/DeepSWM.pth \
        --datetime "$datetime" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[INFERENCE] Warning: Inference failed" | tee -a "$LOG_FILE"
        ((ERROR_COUNT++))
        return 1
    else
        echo "[INFERENCE] Inference successful" | tee -a "$LOG_FILE"
        return 0
    fi
}

# Main execution
echo "=== Starting Inference Pipeline ===" | tee -a "$LOG_FILE"

# Calculate current and previous datetime
CURRENT_DATETIME=$(date -u +"%m%d%H")
PREV_DATETIME=$(date -u -d '1 hour ago' +"%m%d%H")
INFERENCE_DATETIME=$(date -u +"%Y%m%d_%H0000")

echo "Current datetime: $CURRENT_DATETIME" | tee -a "$LOG_FILE"
echo "Previous datetime: $PREV_DATETIME" | tee -a "$LOG_FILE"
echo "Inference datetime: $INFERENCE_DATETIME" | tee -a "$LOG_FILE"

# Fetch data
fetch_data "$CURRENT_DATETIME" "CURRENT"
fetch_data "$PREV_DATETIME" "PREVIOUS"

# Extract features
extract_features "$INFERENCE_DATETIME"

# Run inference
run_inference "$INFERENCE_DATETIME"

# Summary
echo "=== Execution Summary ===" | tee -a "$LOG_FILE"
echo "Total errors encountered: $ERROR_COUNT" | tee -a "$LOG_FILE"
echo "Execution completed at: $(date)" | tee -a "$LOG_FILE"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  Workflow completed with $ERROR_COUNT warnings" | tee -a "$LOG_FILE"
    echo "Check the log file: $LOG_FILE for details" | tee -a "$LOG_FILE"
else
    echo "✅ Workflow completed successfully with no errors" | tee -a "$LOG_FILE"
fi

# Move log file to logs directory if it exists
if [ -d "logs" ]; then
    mv "$LOG_FILE" "logs/"
    echo "Log moved to logs/$LOG_FILE"
fi

# Always exit successfully
exit 0 
