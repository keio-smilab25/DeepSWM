import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import pickle

from src.data.utils.main.processor import DataProcessor
from src.data.utils.main.loader import DataLoader


class SolarFlareDataset(Dataset):
    def __init__(
        self, data_dir, periods, split="train", cache_dirs=None, force_recalc=False
    ):
        self.data_dir = data_dir
        self.periods = [
            (pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods
        ]
        self.split = split
        
        # For pretrain, we use a simplified cache approach since CacheManager expects different parameters
        # Create our own simple cache directory management
        self.cache_dir = cache_dirs[split]
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if split == "train":
            self.p1, self.p99 = self._compute_and_save_stats(
                data_dir, self.cache_dir, force_recalc=force_recalc
            )
        else:
            self.p1, self.p99 = self._load_stats(cache_dirs["train"])

        self.data_indices, self.labels, self.timestamps = self._process_data()
        self.valid_indices = self._get_valid_indices()
        print(f"{split.capitalize()} dataset: {len(self.valid_indices)} samples")

    def _compute_and_save_stats(self, data_dir, cache_dir, force_recalc=False):
        """Compute and save statistics (replacing StatsHandler.compute_and_save_stats)"""
        stats_file = os.path.join(cache_dir, "stats.npz")
        
        if not force_recalc and os.path.exists(stats_file):
            print(f"Loading cached stats from {stats_file}")
            data = np.load(stats_file)
            return data['p1'], data['p99']
        
        print("Computing statistics...")
        all_values = []
        
        for file_name in tqdm(os.listdir(data_dir)):
            if file_name.endswith('.h5'):
                file_path = os.path.join(data_dir, file_name)
                try:
                    with h5py.File(file_path, 'r') as f:
                        for key in f.keys():
                            if 'image' in key.lower():
                                data = f[key][:]
                                all_values.append(data.flatten())
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    continue
        
        if all_values:
            all_values = np.concatenate(all_values)
            p1, p99 = np.percentile(all_values, [1, 99])
            
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(stats_file, p1=p1, p99=p99)
            print(f"Saved stats: p1={p1:.4f}, p99={p99:.4f}")
            return p1, p99
        else:
            print("Warning: No data found for statistics computation")
            return 0.0, 1.0

    def _load_stats(self, cache_dir):
        """Load statistics (replacing StatsHandler.load_stats)"""
        stats_file = os.path.join(cache_dir, "stats.npz")
        if os.path.exists(stats_file):
            data = np.load(stats_file)
            return data['p1'], data['p99']
        else:
            print(f"Warning: Stats file not found at {stats_file}")
            return 0.0, 1.0

    def _normalize_solar_data(self, X, p1, p99):
        """Normalize solar data (replacing StatsHandler.normalize_solar_data)"""
        # Clip and normalize to [0, 1] range
        X = np.clip(X, p1, p99)
        X = (X - p1) / (p99 - p1)
        return X

    def _process_data(self):
        print(f"Processing data files for {self.split} period...")
        data_indices = []
        y_data = []
        timestamps = []

        # Filter files based on periods (copied from DataProcessor logic)
        filtered_files = [
            f
            for f in sorted(os.listdir(self.data_dir))
            if f.endswith(".h5")
            and self._is_in_periods(
                pd.to_datetime(f.split(".")[0], format="%Y%m%d_%H%M%S"),
                self.periods
            )
        ]
        
        print(f"Found {len(filtered_files)} files for {self.split} period")

        for file in tqdm(filtered_files, desc="Processing files"):
            file_path = os.path.join(self.data_dir, file)
            
            try:
                with h5py.File(file_path, "r") as f:
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

                    data_indices.append(file_path)
                    timestamps.append(ts)
                    y_data.append(y)
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue

        return data_indices, y_data, timestamps

    def _is_in_periods(self, date, periods):
        """Determine if the specified date is within the periods"""
        return any(start <= date <= end for start, end in periods)

    def _get_valid_indices(self):
        """Get valid indices for the dataset"""
        valid_indices_file = os.path.join(self.cache_dir, "valid_indices.pkl")
        
        # Try to load cached valid indices
        if os.path.exists(valid_indices_file):
            try:
                with open(valid_indices_file, 'rb') as f:
                    valid_indices = pickle.load(f)
                valid_indices = [idx for idx in valid_indices if idx < len(self.data_indices)]
                if len(valid_indices) > 0:
                    return valid_indices
                print("Warning: No valid indices found in cache, recalculating...")
            except Exception as e:
                print(f"Error loading cached indices: {e}")

        print("Finding valid indices...")
        valid_indices = []
        for i in range(len(self.data_indices)):
            try:
                file_path = self.data_indices[i]
                with h5py.File(file_path, "r") as f:
                    y = f["y"][()]
                    
                    # Decode label directly
                    if isinstance(y, (bytes, np.bytes_)):
                        y_str = y.decode("utf-8")
                        label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                        decoded_y = label_map.get(y_str, 0)
                    elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                        label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                        decoded_y = label_map.get(y[0], 0)
                    else:
                        decoded_y = y
                    
                    if decoded_y != 0 and i < len(self.labels):
                        valid_indices.append(i)
            except Exception as e:
                print(f"Error processing index {i}: {str(e)}")
                continue

        if len(valid_indices) == 0:
            raise RuntimeError("No valid indices found in the dataset")

        # Save valid indices to cache
        try:
            with open(valid_indices_file, 'wb') as f:
                pickle.dump(valid_indices, f)
        except Exception as e:
            print(f"Warning: Could not save valid indices to cache: {e}")
            
        return valid_indices

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        file_path = self.data_indices[valid_idx]

        try:
            with h5py.File(file_path, "r") as f:
                X = f["X"][:]
                y = f["y"][()]
                timestamp = f["timestamp"][()]
                
                # Process label
                if isinstance(y, (bytes, np.bytes_)):
                    y_str = y.decode("utf-8")
                    label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                    y = label_map.get(y_str, 0)
                elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                    label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                    y = label_map.get(y[0], 0)
                
                if X is None:
                    raise ValueError("Could not process H5 file")

                X = np.nan_to_num(X, 0)
                X = self._normalize_solar_data(X, self.p1, self.p99)
                X = torch.from_numpy(X).float()
                y_onehot = DataLoader.create_onehot_label(y)
                return X, y_onehot, timestamp

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.valid_indices)


class AllDataDataset(Dataset):
    def __init__(self, data_dir, cache_dir):
        self.data_dir = data_dir
        self.p1, self.p99 = self._load_stats(cache_dir)
        self.file_paths = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])

    def _load_stats(self, cache_dir):
        """Load statistics (replacing StatsHandler.load_stats)"""
        stats_file = os.path.join(cache_dir, "stats.npz")
        if os.path.exists(stats_file):
            data = np.load(stats_file)
            return data['p1'], data['p99']
        else:
            print(f"Warning: Stats file not found at {stats_file}")
            return 0.0, 1.0

    def _normalize_solar_data(self, X, p1, p99):
        """Normalize solar data (replacing StatsHandler.normalize_solar_data)"""
        # Clip and normalize to [0, 1] range
        X = np.clip(X, p1, p99)
        X = (X - p1) / (p99 - p1)
        return X

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_paths[idx])

        try:
            with h5py.File(file_path, "r") as f:
                X = f["X"][:]
                y = f["y"][()]
                timestamp = f["timestamp"][()]
                
                # Process label
                if isinstance(y, (bytes, np.bytes_)):
                    y_str = y.decode("utf-8")
                    label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                    y = label_map.get(y_str, 0)
                elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                    label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                    y = label_map.get(y[0], 0)
                
                if X is None:
                    raise ValueError("Could not process H5 file")

                X = np.nan_to_num(X, 0)
                X = self._normalize_solar_data(X, self.p1, self.p99)
                X = torch.from_numpy(X).float()
                y_onehot = DataLoader.create_onehot_label(y)
                return X, y_onehot, timestamp

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            X = torch.zeros((10, 256, 256), dtype=torch.float32)
            y_onehot = torch.zeros(4, dtype=torch.float32)
            timestamp = pd.to_datetime(
                self.file_paths[idx].split(".")[0], format="%Y%m%d_%H%M%S"
            )
            return X, y_onehot, timestamp
