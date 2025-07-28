from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from src.data.pretrain.dataset import SolarFlareDataset, AllDataDataset


def collate_fn(batch):
    """Function for processing batch data"""
    images, labels, timestamps = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels, list(timestamps)


def create_data_loader(dataset, batch_size, num_workers, shuffle=True, is_train=True):
    """Function to create a data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def setup_datasets(
    args, cache_dirs, train_periods=None, val_periods=None, test_periods=None
):
    """
    Function to set up datasets

    Parameters:
        args: Command line arguments
        cache_dirs: Cache directories
        train_periods: List of training periods [(start_date, end_date), ...]
        val_periods: List of validation periods [(start_date, end_date), ...]
        test_periods: List of test periods [(start_date, end_date), ...]

    Returns:
        train_dataset, val_dataset, test_dataset: Each dataset
    """
    print("Setting up datasets...")

    # Default period settings
    if train_periods is None:
        train_periods = [("2010-05-01", "2019-11-30")]

    if val_periods is None:
        val_periods = [("2019-12-01", "2020-12-31")]

    if test_periods is None:
        test_periods = [("2021-01-01", "2022-11-30")]

    # Training dataset
    train_dataset = SolarFlareDataset(
        data_dir=args.input_dir,
        periods=train_periods,
        split="train",
        cache_dirs=cache_dirs,
        force_recalc=args.force_recalc,
    )

    # Validation dataset
    val_dataset = SolarFlareDataset(
        data_dir=args.input_dir,
        periods=val_periods,
        split="val",
        cache_dirs=cache_dirs,
        force_recalc=args.force_recalc,
    )

    # Test dataset
    test_dataset = SolarFlareDataset(
        data_dir=args.input_dir,
        periods=test_periods,
        split="test",
        cache_dirs=cache_dirs,
        force_recalc=args.force_recalc,
    )

    return train_dataset, val_dataset, test_dataset


def setup_dataloaders(args, train_dataset, val_dataset, test_dataset):
    """Function to set up data loaders"""
    print("Setting up dataloaders...")

    train_loader = create_data_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, is_train=True
    )

    val_loader = create_data_loader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False, is_train=False
    )

    test_loader = create_data_loader(
        test_dataset, args.batch_size, args.num_workers, shuffle=False, is_train=False
    )

    return train_loader, val_loader, test_loader


def setup_all_data_loader(args, cache_dirs):
    """Function to set up data loader for all data"""
    print("Setting up all data loader...")

    all_dataset = AllDataDataset(
        data_dir=args.input_dir,
        cache_dir=cache_dirs["train"],
    )

    all_loader = create_data_loader(
        all_dataset, args.batch_size, args.num_workers, shuffle=False, is_train=False
    )

    return all_loader


def setup_visualization_loader(args, cache_dirs, visualization_periods=None):
    """
    Function to set up data loader for visualization

    Parameters:
        args: Command line arguments
        cache_dirs: Cache directories
        visualization_periods: List of visualization periods [(start_date, end_date), ...]

    Returns:
        test_loader: Data loader for visualization
    """
    print("Setting up visualization loader...")

    # Default period settings
    if visualization_periods is None:
        visualization_periods = [("2019-12-01", "2022-11-30")]

    test_dataset = SolarFlareDataset(
        data_dir=args.input_dir,
        periods=visualization_periods,
        split="test",
        cache_dirs=cache_dirs,
        force_recalc=True,  # Clear cache to reflect period changes
    )

    # Set batch size to 1 for timestamp processing in visualization
    test_loader = create_data_loader(test_dataset, 1, 4, shuffle=False, is_train=True)

    return test_loader


def parse_time_range(timestamp_str, hours_before=12, hours_after=12):
    """
    Function to parse time range from timestamp string

    Parameters:
        timestamp_str: Timestamp string (YYYYMMDD_HHMMSS format)
        hours_before: How many hours before the specified time
        hours_after: How many hours after the specified time

    Returns:
        (start_time, end_time): Time range tuple, or None
    """
    if timestamp_str is None:
        return None

    selected_time = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
    start_time = selected_time - pd.Timedelta(hours=hours_before)
    end_time = selected_time + pd.Timedelta(hours=hours_after)

    return (start_time, end_time)
