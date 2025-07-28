import os
import numpy as np
import torch
import h5py
from typing import List


class DataLoader:
    """Handles data loading operations"""
    
    @staticmethod
    def load_image_sequence(data_indices: List, history_indices: List, means: np.ndarray, stds: np.ndarray) -> List[torch.Tensor]:
        """Load sequence of images with normalization"""
        X = []
        
        for i in history_indices:
            try:
                with h5py.File(data_indices[i], "r") as f:
                    x = f["X"][:]
                    x = np.nan_to_num(x, 0)
                    x_normalized = (x - means[:, None, None]) / (
                        stds[:, None, None] + 1e-8
                    )
                    X.append(torch.from_numpy(x_normalized).float())
            except Exception as e:
                x = np.zeros((10, 256, 256), dtype=np.float32)
                X.append(torch.from_numpy(x).float())
        
        return X
    
    @staticmethod
    def load_features(features_dir: str, timestamp) -> torch.Tensor:
        """Load feature data"""
        feature_file = os.path.join(
            features_dir, f"{timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
        )
        
        try:
            with h5py.File(feature_file, "r") as f:
                h = f["features"][:]
                h = torch.from_numpy(h.astype(np.float32))
        except:
            h = torch.zeros((672, 128), dtype=torch.float32)
        
        # Check and process NaN and Inf
        if torch.isnan(h).any() or torch.isinf(h).any():
            h = torch.zeros_like(h)
        
        return h
    
    @staticmethod
    def create_onehot_label(y: int) -> torch.Tensor:
        """Create one-hot encoded label"""
        y_onehot = torch.zeros(4, dtype=torch.float32)
        y_onehot[y - 1] = 1.0
        return y_onehot
    
    @staticmethod
    def clean_tensor(X: torch.Tensor) -> torch.Tensor:
        """Clean tensor of NaN and Inf values"""
        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)
        return X 
