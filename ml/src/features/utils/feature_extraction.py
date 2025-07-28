import os
import numpy as np
import torch
from tqdm import tqdm
import gc
from typing import Dict, List
from torch.utils.data.dataloader import DataLoader
from src.experiments.utils.pretrain.io import save_features


def extract_mae_features(model, dataloader, mask_ratio, device):
    """
    Extract features using the model

    Parameters:
        model: Model used for feature extraction
        dataloader: Data loader
        mask_ratio: Mask ratio
        device: Computation device

    Returns:
        features: Extracted features
        labels: Labels
        timestamps: Timestamps
    """
    model.eval()
    batch_size = dataloader.batch_size
    total_samples = len(dataloader.dataset)

    # Pre-allocate memory
    features = np.zeros((total_samples, model.embed_dim))
    labels = np.zeros((total_samples, 4))  # 4 dimensions for one-hot encoding
    timestamps = []

    with torch.no_grad():
        start_idx = 0
        for batch_idx, (images, label, timestamp) in enumerate(tqdm(dataloader)):
            try:
                current_batch_size = images.size(0)
                images = images.to(device)
                batch_features, _, _ = model.forward_encoder_pyramid(images, mask_ratio)
                mean_features = batch_features[:, 1:, :].mean(dim=1)

                # Store directly in NumPy array on CPU
                end_idx = start_idx + current_batch_size
                features[start_idx:end_idx] = mean_features.cpu().numpy()
                labels[start_idx:end_idx] = label.numpy()
                timestamps.extend(timestamp)

                # Update index
                start_idx = end_idx

                # Free memory
                del images
                del batch_features
                del mean_features
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error in batch processing: {str(e)}")
                continue

    return features, labels, np.array(timestamps)


def process_dataset_features(
    model, loader, mask_ratio, device, output_dir, dataset_type, trial_name
):
    """
    Extract and save features for a dataset

    Parameters:
        model: Model used for feature extraction
        loader: Data loader
        mask_ratio: Mask ratio
        device: Computation device
        output_dir: Output directory
        dataset_type: Dataset type (train/val/test)
        trial_name: Trial name
    """
    try:
        print(f"Extracting features for {dataset_type} dataset...")
        features, labels, timestamps = extract_mae_features(
            model, loader, mask_ratio, device
        )

        print(f"Saving features for {dataset_type} dataset...")
        save_features(
            output_dir, features, labels, timestamps, dataset_type, trial_name
        )

        print(f"Completed processing {dataset_type} dataset")
        return True
    except Exception as e:
        print(f"Error processing {dataset_type} dataset: {str(e)}")
        return False


def process_features(
    model: torch.nn.Module,
    data_loader: DataLoader,
    mask_ratio: float,
    device: torch.device,
    output_dir: str,
    dataset_type: str,
    trial_name: str,
) -> bool:
    """
    Extract and save features

    Parameters:
        model: Model to extract features
        data_loader: Data loader
        mask_ratio: Mask ratio
        device: Device
        output_dir: Output directory
        dataset_type: Dataset type
        trial_name: Trial name

    Returns:
        success: Whether the processing was successful
    """
    try:
        process_dataset_features(
            model,
            data_loader,
            mask_ratio,
            device,
            output_dir,
            dataset_type,
            trial_name,
        )
        return True
    except Exception as e:
        print(f"Error processing features for {dataset_type}: {str(e)}")
        return False


def process_all_features(
    model: torch.nn.Module,
    loaders: List[DataLoader],
    dataset_types: List[str],
    mask_ratio: float,
    device: torch.device,
    output_dir: str,
    trial_name: str,
) -> Dict[str, bool]:
    """
    Extract features for multiple datasets

    Parameters:
        model: Model to extract features
        loaders: List of data loaders
        dataset_types: List of dataset types
        mask_ratio: Mask ratio
        device: Device
        output_dir: Output directory
        trial_name: Trial name

    Returns:
        results: Processing results for each dataset
    """
    results = {}

    for loader, dataset_type in zip(loaders, dataset_types):
        print(f"Processing features for {dataset_type} dataset...")
        success = process_features(
            model, loader, mask_ratio, device, output_dir, dataset_type, trial_name
        )
        results[dataset_type] = success

        # Memory release
        gc.collect()
        torch.cuda.empty_cache()

    return results
