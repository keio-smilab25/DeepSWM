import os
import pickle
import numpy as np
from typing import Any, Tuple, List, Optional


class CacheManager:
    """Manages caching for dataset operations"""
    
    def __init__(self, cache_root: str, fold_num: int, split: str, history: int):
        self.cache_root = cache_root
        self.fold_num = fold_num
        self.split = split
        self.history = history
        
        # Setup cache directories
        fold_dir = os.path.join(cache_root, f"fold{fold_num}")
        self.cache_dir = os.path.join(fold_dir, split)
        self.train_cache_dir = os.path.join(fold_dir, "train")
        
        # Create directories
        for d in [self.cache_dir, self.train_cache_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Cache file paths
        self.indices_file = os.path.join(self.cache_dir, "data_indices.pkl")
        self.means_file = os.path.join(self.train_cache_dir, "means.npy")
        self.stds_file = os.path.join(self.train_cache_dir, "stds.npy")
        self.labels_file = os.path.join(self.cache_dir, "labels.npy")
        self.timestamps_file = os.path.join(self.cache_dir, "timestamps.pkl")
        self.valid_indices_file = os.path.join(
            self.cache_dir, f"{history}_valid_indices.pkl"
        )
    
    def cache_exists(self, cache_type: str) -> bool:
        """Check if cache exists for given type"""
        if cache_type == "all":
            files = [self.indices_file, self.means_file, self.stds_file, 
                    self.labels_file, self.timestamps_file]
        elif cache_type == "data":
            files = [self.indices_file, self.labels_file, self.timestamps_file]
        elif cache_type == "stats":
            files = [self.means_file, self.stds_file]
        elif cache_type == "valid_indices":
            files = [self.valid_indices_file]
        else:
            return False
        
        return all(os.path.exists(f) for f in files)
    
    def save_data_cache(self, data_indices: List, labels: List, timestamps: List):
        """Save data cache"""
        os.makedirs(os.path.dirname(self.indices_file), exist_ok=True)
        with open(self.indices_file, "wb") as f:
            pickle.dump(data_indices, f)
        np.save(self.labels_file, labels)
        with open(self.timestamps_file, "wb") as f:
            pickle.dump(timestamps, f)
    
    def save_stats_cache(self, means: np.ndarray, stds: np.ndarray):
        """Save statistics cache"""
        np.save(self.means_file, means)
        np.save(self.stds_file, stds)
    
    def save_valid_indices(self, valid_indices: List):
        """Save valid indices cache"""
        with open(self.valid_indices_file, "wb") as f:
            pickle.dump(valid_indices, f)
    
    def load_cached_data(self) -> Tuple[List, List, List]:
        """Load cached data without stats"""
        with open(self.indices_file, "rb") as f:
            data_indices = pickle.load(f)
        labels = np.load(self.labels_file).tolist()
        with open(self.timestamps_file, "rb") as f:
            timestamps = pickle.load(f)
        return data_indices, labels, timestamps
    
    def load_cached_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load cached statistics"""
        means = np.load(self.means_file)
        stds = np.load(self.stds_file)
        return means, stds
    
    def load_cached_full_data(self) -> Tuple[List, np.ndarray, np.ndarray, List, List]:
        """Load all cached data including stats"""
        with open(self.indices_file, "rb") as f:
            data_indices = pickle.load(f)
        means = np.load(self.means_file)
        stds = np.load(self.stds_file)
        labels = np.load(self.labels_file).tolist()
        with open(self.timestamps_file, "rb") as f:
            timestamps = pickle.load(f)
        return data_indices, means, stds, labels, timestamps
    
    def load_valid_indices(self) -> List:
        """Load cached valid indices"""
        with open(self.valid_indices_file, "rb") as f:
            return pickle.load(f) 
