import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import timedelta
import argparse


def create_history_features(input_dir, input_path, output_dir, history, step_hours):
    # Create output directory
    output_dir = os.path.join(
        output_dir, f"all_features_history_{history}_step_{step_hours}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Get input files
    if input_path:
        input_files = [input_path]
    else:
        input_files = sorted(
            glob(
                os.path.join(
                    input_dir,
                    "all_data_features_*.h5",
                )
            )
        )

    if len(input_files) == 0:
        raise ValueError(
            f"No input files found.\n"
            f"Specified path: {input_dir if not input_path else input_path}"
        )

    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")

    # Load features, labels, and timestamps
    features = []
    labels = []
    timestamps = []

    for file in input_files:
        try:
            with h5py.File(file, "r") as f:
                features.append(f["features"][:])
                labels.append(f["labels"][:])
                timestamps.extend([ts.decode("utf-8") for ts in f["timestamps"][:]])
        except Exception as e:
            print(f"Error occurred while loading file {file}: {e}")
            continue

    if not features:
        raise ValueError("No files with valid data found.")

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    timestamps = pd.to_datetime(timestamps)

    print(
        f"Total features: {len(features)}, labels: {len(labels)}, timestamps: {len(timestamps)}"
    )

    # Process each timestamp
    successful_count = 0
    error_count = 0

    for current_timestamp in tqdm(timestamps, desc="Processing timestamps"):
        try:
            current_timestamp_str = current_timestamp.strftime("%Y%m%d_%H%M%S")

            # Generate history timestamps
            history_timestamps = [
                current_timestamp - timedelta(hours=j * step_hours)
                for j in range(history)
            ]

            # Extract features and labels
            history_features = []
            for ts in history_timestamps:
                idx = np.where(timestamps == ts)[0]
                if len(idx) > 0:
                    history_features.append(features[idx[0]])
                else:
                    history_features.append(np.zeros_like(features[0]))

            history_features = np.array(history_features)
            history_label = labels[np.where(timestamps == current_timestamp)[0][0]]

            # Create filename and save
            filename = f"{current_timestamp_str}.h5"
            output_file = os.path.join(output_dir, filename)

            with h5py.File(output_file, "w") as f:
                f.create_dataset("features", data=history_features)
                f.create_dataset("label", data=history_label)
                f.create_dataset(
                    "timestamps",
                    data=[
                        ts.strftime("%Y-%m-%d %H:%M:%S").encode("utf-8")
                        for ts in history_timestamps
                    ],
                )
                mask = np.array(
                    [1 if ts in timestamps else 0 for ts in history_timestamps]
                )
                f.create_dataset("mask", data=mask)

            successful_count += 1
        except Exception as e:
            print(f"Error processing timestamp {current_timestamp_str}: {e}")
            error_count += 1
            continue

    print(f"Processing completed:")
    print(f"  - Successfully processed files: {successful_count}")
    print(f"  - Files with errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description="Create history features")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/all_features",
        help="Input directory path",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Single input file path (takes precedence over input_dir if specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/all_features/completed",
        help="Output directory path",
    )
    parser.add_argument(
        "--history", type=int, default=672, help="Number of history points"
    )
    parser.add_argument(
        "--step_hours", type=int, default=1, help="Step time (in hours)"
    )

    args = parser.parse_args()

    create_history_features(
        args.input_dir, args.input_path, args.output_dir, args.history, args.step_hours
    )


if __name__ == "__main__":
    main()
