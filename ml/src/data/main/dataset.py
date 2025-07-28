"""Dataset for Flare Transformer"""

import torch
from torch.utils.data import Dataset
import pandas as pd

# Import utility modules
from src.data.utils.main.cache import CacheManager
from src.data.utils.main.processor import DataProcessor
from src.data.utils.main.validator import DataValidator
from src.data.utils.main.loader import DataLoader


class SolarFlareDatasetWithFeatures(Dataset):
    def __init__(
        self, data_dir, periods, history=4, split="train", args=None, transform=None
    ):
        """
        Args:
            data_dir (str): Path to data directory
            periods (list): List of periods in the format [(start_date, end_date), ...]
            history (int): Length of history
            split (str): Dataset type ("train", "valid", "test")
            args: Additional arguments
            transform: Transform for data augmentation
        """
        self.data_dir = data_dir
        self.periods = [
            (pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods
        ]
        self.history = history
        self.split = split
        self.args = args
        self.transform = transform
        self.return_timestamp = False
        self.features_dir = args.features_path

        # Initialize cache manager
        fold_num = getattr(args, "fold", 3)
        self.cache_manager = CacheManager(
            args.cache_root, fold_num, split, history
        )

        # Process data based on split type
        if split == "train":
            self._process_train_data()
        else:
            self._process_validation_data()

        # Get valid indices
        self._process_valid_indices()

        print(f"Fold {fold_num} - {self.split}:")
        print("total samples: ", len(self.data_indices))
        print("valid samples: ", len(self.valid_indices))

    def _process_train_data(self):
        """Process training data with statistics calculation"""
        if (not self.args.dataset["force_recalc_stats"] and 
            self.cache_manager.cache_exists("all")):
            print(f"Loading cached data for {self.split} period...")
            (self.data_indices, self.means, self.stds, 
             self.labels, self.timestamps) = self.cache_manager.load_cached_full_data()
            return

        print(f"Processing data files for {self.split} period...")
        (self.data_indices, self.means, self.stds, 
         self.labels, self.timestamps) = DataProcessor.process_files_with_stats(
            self.data_dir, self.periods, self.split
        )

        # Save to cache
        self.cache_manager.save_data_cache(
            self.data_indices, self.labels, self.timestamps
        )
        self.cache_manager.save_stats_cache(self.means, self.stds)

    def _process_validation_data(self):
        """Process validation/test data without statistics calculation"""
        # Load training statistics
        if not self.cache_manager.cache_exists("stats"):
            raise FileNotFoundError(
                f"Training statistics not found. Please process training data first."
            )
        
        self.means, self.stds = self.cache_manager.load_cached_stats()

        # Load or process validation data
        if (not self.args.dataset["force_recalc_stats"] and 
            self.cache_manager.cache_exists("data")):
            print(f"Loading cached data for {self.split} period...")
            (self.data_indices, self.labels, 
             self.timestamps) = self.cache_manager.load_cached_data()
            return

        print(f"Processing data files for {self.split} period...")
        (self.data_indices, self.labels, 
         self.timestamps) = DataProcessor.process_files_without_stats(
            self.data_dir, self.periods, self.split
        )

        # Save to cache
        self.cache_manager.save_data_cache(
            self.data_indices, self.labels, self.timestamps
        )

    def _process_valid_indices(self):
        """Process and cache valid indices"""
        if (not self.args.dataset["force_recalc_indices"] and 
            self.cache_manager.cache_exists("valid_indices")):
            print(f"Loading cached valid indices for {self.split}...")
            self.valid_indices = self.cache_manager.load_valid_indices()
            return

        # Calculate valid indices
        self.valid_indices = DataValidator.get_valid_indices(
            self.data_indices, self.timestamps, self.labels, 
            self.history, self.split
        )

        # Save cache
        self.cache_manager.save_valid_indices(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        latest_idx = self.valid_indices[idx]
        history_indices = list(range(latest_idx - self.history + 1, latest_idx + 1))

        # Load image sequence
        X = DataLoader.load_image_sequence(
            self.data_indices, history_indices, self.means, self.stds
        )

        assert (
            len(X) == self.history
        ), f"Expected history length {self.history}, got {len(X)}"

        # Stack list to convert to Tensor
        X = torch.stack(X, dim=0)  # [history, channels, height, width]

        # Apply data augmentation (only for train)
        if self.transform is not None and self.split == "train":
            X = self.transform(X.to(self.args.device)).cpu()

        # Load feature data
        latest_timestamp = self.timestamps[latest_idx]
        h = DataLoader.load_features(self.features_dir, latest_timestamp)

        # Create one-hot label
        y = self.labels[latest_idx]
        y_onehot = DataLoader.create_onehot_label(y)

        # Clean tensors
        X = DataLoader.clean_tensor(X)

        if self.return_timestamp:
            return X, h, y_onehot, latest_timestamp
        else:
            return X, h, y_onehot
