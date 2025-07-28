import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import List


class DataValidator:
    """Handles data validation operations"""
    
    @staticmethod
    def _check_valid_index(i: int, data_indices: List, timestamps: List, labels: List, history: int):
        """Internal method to check if an index is valid"""
        current_time = timestamps[i]
        total_missing_images = 0
        total_images = history * 10
        
        for h in range(history):
            if i - h < 0:
                return None
            
            history_time = timestamps[i - h]
            expected_time = current_time - pd.Timedelta(hours=h)
            
            if (
                history_time.hour != expected_time.hour
                or history_time.date() != expected_time.date()
            ):
                return None
            
            try:
                with h5py.File(data_indices[i - h], "r") as f:
                    x = f["X"][:]
                    
                    for c in range(10):
                        channel_data = x[c]
                        if (
                            np.std(channel_data) < 1e-6
                            or np.all(channel_data == 0)
                            or np.all(np.isnan(channel_data))
                        ):
                            total_missing_images += 1
                            
            except Exception as e:
                total_missing_images += 10
            
            if total_missing_images >= 10:
                return None
        
        return i if labels[i] > 0 else None
    
    @staticmethod
    def get_valid_indices(data_indices: List, timestamps: List, labels: List, history: int, split: str) -> List:
        """Get valid indices (serial processing version)"""
        print(f"Calculating valid indices for {split}...")
        valid_indices = []
        
        for i in tqdm(range(len(data_indices)), desc="Checking valid indices"):
            result = DataValidator._check_valid_index(i, data_indices, timestamps, labels, history)
            if result is not None:
                valid_indices.append(result)
        
        print(f"Found {len(valid_indices)} valid indices for {split}")
        return valid_indices 
