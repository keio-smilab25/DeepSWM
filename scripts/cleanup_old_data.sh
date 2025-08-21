#!/bin/bash
# Data cleanup script for DeepSWM project.
# Removes old data files while preserving the most recent 3 months (672 hours + buffer).

set -e  # Exit on any error

# Configuration
MONTHS_TO_KEEP=3
BUFFER_DAYS=7
YEARS_TO_KEEP_IMAGES=2
DATA_DIRS=("ml/datasets/all_data_hours" "ml/datasets/all_features")
IMAGE_DIR="data/images"

# Parse command line arguments
DRY_RUN=false
MODE="cleanup"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --dry-run           Show what would be deleted without actually deleting"
    echo "  --mode=MODE         Cleanup mode: 'cleanup' (default), 'oldest-hour', or 'images'"
    echo "  --months=N          Number of months to keep for datasets (default: 3)"
    echo "  --years=N           Number of years to keep for images (default: 2)"
    echo "  --help              Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --months=*)
            MONTHS_TO_KEEP="${1#*=}"
            shift
            ;;
        --years=*)
            YEARS_TO_KEEP_IMAGES="${1#*=}"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "cleanup" && "$MODE" != "oldest-hour" && "$MODE" != "images" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'cleanup', 'oldest-hour', or 'images'"
    exit 1
fi

# Function to get cutoff date (YYYYMMDD format)
get_cutoff_date() {
    local months=$1
    local buffer_days=$2
    
    # Calculate days to subtract (approximate: 30 days per month + buffer)
    local days_to_subtract=$((months * 30 + buffer_days))
    
    # Get cutoff date in YYYYMMDD format
    if command -v gdate >/dev/null 2>&1; then
        # macOS with GNU date
        gdate -d "${days_to_subtract} days ago" +%Y%m%d
    else
        # Linux date
        date -d "${days_to_subtract} days ago" +%Y%m%d
    fi
}

# Function to extract date from filename (YYYYMMDD_HHMMSS.h5 -> YYYYMMDD)
get_file_date() {
    local filename=$(basename "$1")
    echo "${filename:0:8}"  # Extract first 8 characters (YYYYMMDD)
}

# Function to cleanup old files in a directory
cleanup_directory() {
    local dir="$1"
    local cutoff_date="$2"
    local deleted_count=0
    local total_size=0
    
    echo
    echo "Processing directory: $dir"
    
    if [[ ! -d "$dir" ]]; then
        echo "Directory not found: $dir"
        return 0
    fi
    
    local total_files=$(find "$dir" -name "*.h5" | wc -l)
    echo "Total files found: $total_files"
    echo "Cutoff date: $cutoff_date"
    
    # Find and process old files
    while IFS= read -r -d '' file; do
        local file_date=$(get_file_date "$file")
        
        # Compare dates (string comparison works for YYYYMMDD format)
        if [[ "$file_date" < "$cutoff_date" ]]; then
            local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            local file_size_mb=$((file_size / 1024 / 1024))
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY RUN] Would delete: $file (${file_size_mb} MB)"
            else
                if rm "$file" 2>/dev/null; then
                    echo "Deleted: $file (${file_size_mb} MB)"
                else
                    echo "Error deleting: $file"
                    continue
                fi
            fi
            
            deleted_count=$((deleted_count + 1))
            total_size=$((total_size + file_size))
        fi
    done < <(find "$dir" -name "*.h5" -print0)
    
    echo "Files processed in $dir: $deleted_count"
    echo "$deleted_count $total_size"  # Return values
}

# Function to remove oldest hour of data
cleanup_oldest_hour() {
    local dir="$1"
    local deleted_count=0
    local total_size=0
    
    echo
    echo "Removing oldest hour data from: $dir"
    
    if [[ ! -d "$dir" ]]; then
        echo "Directory not found: $dir"
        return 0
    fi
    
    # Find the oldest file to determine the oldest hour
    local oldest_file=$(find "$dir" -name "*.h5" | sort | head -1)
    
    if [[ -z "$oldest_file" ]]; then
        echo "No files found in $dir"
        echo "0 0"
        return 0
    fi
    
    # Extract date and hour from oldest file (YYYYMMDD_HH)
    local filename=$(basename "$oldest_file")
    local oldest_hour="${filename:0:11}"  # YYYYMMDD_HH
    
    echo "Oldest hour: ${oldest_hour:0:8} ${oldest_hour:9:2}:00:00"
    
    # Find all files from that hour
    local hour_files=()
    while IFS= read -r -d '' file; do
        local file_prefix="${filename:0:11}"
        if [[ "$(basename "$file")" == "${oldest_hour}"* ]]; then
            hour_files+=("$file")
        fi
    done < <(find "$dir" -name "${oldest_hour}*.h5" -print0)
    
    echo "Files to delete: ${#hour_files[@]}"
    
    # Delete files from oldest hour
    for file in "${hour_files[@]}"; do
        local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        local file_size_mb=$((file_size / 1024 / 1024))
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] Would delete: $file (${file_size_mb} MB)"
        else
            if rm "$file" 2>/dev/null; then
                echo "Deleted: $file (${file_size_mb} MB)"
            else
                echo "Error deleting: $file"
                continue
            fi
        fi
        
        deleted_count=$((deleted_count + 1))
        total_size=$((total_size + file_size))
    done
    
    echo "$deleted_count $total_size"  # Return values
}

# Function to cleanup old image directories
cleanup_old_images() {
    local years_to_keep=$1
    local deleted_count=0
    local total_size=0
    
    echo
    echo "Cleaning up old image data from: $IMAGE_DIR"
    
    if [[ ! -d "$IMAGE_DIR" ]]; then
        echo "Directory not found: $IMAGE_DIR"
        echo "0 0"
        return 0
    fi
    
    # Calculate cutoff year and month (YYYYMM format)
    local current_year=$(date +%Y)
    local current_month=$(date +%m)
    local cutoff_year=$((current_year - years_to_keep))
    local cutoff_date="${cutoff_year}${current_month}"
    
    echo "Cutoff date: ${cutoff_year}-${current_month} (${years_to_keep} years ago)"
    
    # Find directories with MMDD format and check if they're old
    for dir_path in "$IMAGE_DIR"/*; do
        if [[ -d "$dir_path" ]]; then
            local dir_name=$(basename "$dir_path")
            
            # Check if directory name matches MMDD format (4 digits)
            if [[ "$dir_name" =~ ^[0-9]{4}$ ]]; then
                local dir_month="${dir_name:0:2}"
                local dir_day="${dir_name:2:2}"
                
                # Assume current year for comparison (you may want to adjust this logic)
                local dir_date="${current_year}${dir_month}"
                
                # If directory represents a date older than cutoff
                if [[ "$dir_date" < "$cutoff_date" ]]; then
                    # Calculate directory size
                    local dir_size=$(du -sb "$dir_path" 2>/dev/null | cut -f1 || echo "0")
                    local dir_size_mb=$((dir_size / 1024 / 1024))
                    
                    if [[ "$DRY_RUN" == "true" ]]; then
                        echo "[DRY RUN] Would delete directory: $dir_path (${dir_size_mb} MB)"
                    else
                        if rm -rf "$dir_path" 2>/dev/null; then
                            echo "Deleted directory: $dir_path (${dir_size_mb} MB)"
                        else
                            echo "Error deleting directory: $dir_path"
                            continue
                        fi
                    fi
                    
                    deleted_count=$((deleted_count + 1))
                    total_size=$((total_size + dir_size))
                fi
            fi
        fi
    done
    
    echo "Directories processed: $deleted_count"
    echo "$deleted_count $total_size"  # Return values
}

# Main execution
echo "Data cleanup script starting..."
echo "Mode: $MODE"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE - No files will be actually deleted"
fi

total_deleted=0
total_size=0

if [[ "$MODE" == "cleanup" ]]; then
    cutoff_date=$(get_cutoff_date "$MONTHS_TO_KEEP" "$BUFFER_DAYS")
    echo "Removing files older than $MONTHS_TO_KEEP months (cutoff: $cutoff_date)"
    
    for dir in "${DATA_DIRS[@]}"; do
        result=$(cleanup_directory "$dir" "$cutoff_date")
        deleted=$(echo "$result" | cut -d' ' -f1)
        size=$(echo "$result" | cut -d' ' -f2)
        total_deleted=$((total_deleted + deleted))
        total_size=$((total_size + size))
    done
    
elif [[ "$MODE" == "oldest-hour" ]]; then
    echo "Removing oldest 1 hour of data from each directory"
    
    for dir in "${DATA_DIRS[@]}"; do
        result=$(cleanup_oldest_hour "$dir")
        deleted=$(echo "$result" | cut -d' ' -f1)
        size=$(echo "$result" | cut -d' ' -f2)
        total_deleted=$((total_deleted + deleted))
        total_size=$((total_size + size))
    done

elif [[ "$MODE" == "images" ]]; then
    echo "Removing image data older than $YEARS_TO_KEEP_IMAGES years"
    
    result=$(cleanup_old_images "$YEARS_TO_KEEP_IMAGES")
    deleted=$(echo "$result" | cut -d' ' -f1)
    size=$(echo "$result" | cut -d' ' -f2)
    total_deleted=$((total_deleted + deleted))
    total_size=$((total_size + size))
fi

# Summary
echo
echo "=================================================="
echo "Cleanup Summary:"
echo "Mode: $MODE"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Total files would be deleted: $total_deleted"
    echo "Total size would be freed: $((total_size / 1024 / 1024)) MB"
    echo
    echo "This was a dry run. Remove --dry-run to actually delete files."
else
    echo "Total files deleted: $total_deleted"
    echo "Total size freed: $((total_size / 1024 / 1024)) MB"
fi

echo "Cleanup completed successfully!"
