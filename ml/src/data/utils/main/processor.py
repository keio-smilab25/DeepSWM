import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import List, Tuple


class DataProcessor:
    """Handles data processing operations"""
    
    @staticmethod
    def _is_in_periods(date, periods):
        """Determine if the specified date is within the periods"""
        return any(start <= date <= end for start, end in periods)
    
    @staticmethod
    def process_files_with_stats(data_dir: str, periods: List, split: str) -> Tuple[List, np.ndarray, np.ndarray, List, List]:
        """Process data files and calculate statistics"""
        data_indices = []
        y_data = []
        count = np.zeros(10)
        mean = np.zeros(10)
        M2 = np.zeros(10)
        timestamps = []
        
        # Convert periods to datetime
        periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods]
        
        filtered_files = [
            f
            for f in sorted(os.listdir(data_dir))
            if f.endswith(".h5")
            and DataProcessor._is_in_periods(
                pd.to_datetime(f.split(".")[0], format="%Y%m%d_%H%M%S"),
                periods
            )
        ]
        
        print(
            f"Found {len(filtered_files)} files for period {split} ({periods[0][0]} to {periods[-1][1]})"
        )
        
        for file in tqdm(filtered_files, desc=f"Processing {split} data"):
            file_path = os.path.join(data_dir, file)
            try:
                with h5py.File(file_path, "r") as f:
                    X = f["X"][:]
                    y = f["y"][()]
                    ts = f["timestamp"][()]
                    
                    # Process label
                    if isinstance(y, (bytes, np.bytes_)):
                        y_str = y.decode("utf-8")
                        label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                        y = label_map.get(y_str, 0)
                    elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                        label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                        y = label_map.get(y[0], 0)
                    
                    if y == 0:
                        continue
                    
                    X = np.nan_to_num(X, 0)
                    
                    # Calculate statistics
                    valid_img = False
                    for j in range(10):
                        channel_data = X[j]
                        if np.any(channel_data):
                            valid_img = True
                            new_count = count[j] + 1
                            new_mean = (
                                mean[j] + (channel_data.mean() - mean[j]) / new_count
                            )
                            new_M2 = M2[j] + (channel_data.mean() - mean[j]) * (
                                channel_data.mean() - new_mean
                            )
                            
                            if not np.isnan(new_mean) and not np.isnan(new_M2):
                                count[j] = new_count
                                mean[j] = new_mean
                                M2[j] = new_M2
                    
                    if valid_img:
                        data_indices.append(file_path)
                        ts_str = ts.decode("utf-8")
                        try:
                            timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                            timestamps.append(timestamp)
                            y_data.append(y)
                        except ValueError as e:
                            print(f"Error parsing timestamp {ts_str}: {str(e)}")
                            continue
                            
            except (OSError, KeyError) as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        
        stds = np.where(count > 0, np.sqrt(M2 / count), 0)
        stds[stds == 0] = 1
        
        print(f"Processed {len(data_indices)} valid files for {split} period")
        return data_indices, mean, stds, y_data, timestamps
    
    @staticmethod
    def process_files_without_stats(data_dir: str, periods: List, split: str) -> Tuple[List, List, List]:
        """Process data files without calculating statistics"""
        data_indices = []
        y_data = []
        timestamps = []
        
        # Convert periods to datetime
        periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods]
        
        filtered_files = [
            f
            for f in sorted(os.listdir(data_dir))
            if f.endswith(".h5")
            and DataProcessor._is_in_periods(
                pd.to_datetime(f.split(".")[0], format="%Y%m%d_%H%M%S"),
                periods
            )
        ]
        
        print(
            f"Found {len(filtered_files)} files for period {split} ({periods[0][0]} to {periods[-1][1]})"
        )
        
        for file in tqdm(filtered_files, desc=f"Processing {split} data"):
            file_path = os.path.join(data_dir, file)
            try:
                with h5py.File(file_path, "r") as f:
                    X = f["X"][:]
                    y = f["y"][()]
                    ts = f["timestamp"][()]
                    
                    # Process label
                    if isinstance(y, (bytes, np.bytes_)):
                        y_str = y.decode("utf-8")
                        label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                        y = label_map.get(y_str, 0)
                    elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                        label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                        y = label_map.get(y[0], 0)
                    
                    if y == 0:
                        continue
                    
                    X = np.nan_to_num(X, 0)
                    
                    if np.any(X):
                        data_indices.append(file_path)
                        ts_str = ts.decode("utf-8")
                        try:
                            timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                            timestamps.append(timestamp)
                            y_data.append(y)
                        except ValueError as e:
                            print(f"Error parsing timestamp {ts_str}: {str(e)}")
                            continue
                            
            except (OSError, KeyError) as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        
        print(f"Processed {len(data_indices)} valid files for {split} period")
        return data_indices, y_data, timestamps 
