import os
import pandas as pd
from argparse import Namespace
from torch.utils.data import DataLoader
from src.data.main.dataset import SolarFlareDatasetWithFeatures
from src.data.main.sampler import TrainBalancedBatchSampler
from torch.utils.data._utils.collate import default_collate
from src.data.main.transforms import SolarTransforms
import torch


def load_datasets(args: Namespace, debug: bool):
    data_dir = args.data_path

    solar_transforms = SolarTransforms()

    train_periods = args.train_periods
    val_periods = args.val_periods
    test_periods = args.test_periods

    train_dataset = SolarFlareDatasetWithFeatures(
        data_dir=data_dir,
        periods=train_periods,
        history=args.history,
        split="train",
        args=args,
        transform=solar_transforms,
    )

    val_dataset = SolarFlareDatasetWithFeatures(
        data_dir=data_dir,
        periods=val_periods,
        history=args.history,
        split="valid",
        args=args,
    )

    test_dataset = SolarFlareDatasetWithFeatures(
        data_dir=data_dir,
        periods=test_periods,
        history=args.history,
        split="test",
        args=args,
    )

    return train_dataset, val_dataset, test_dataset


def custom_collate_fn(batch):
    X, h, y = zip(*batch)
    X = torch.stack([item.to(memory_format=torch.contiguous_format) for item in X])
    h = torch.stack(h)
    y = torch.stack(y)
    return X, h, y


def prepare_dataloaders(args: Namespace, debug: bool, imbalance: bool):
    print(f"Prepare Dataloaders for fold {args.fold}")

    if not imbalance:
        args.stage2 = True
    else:
        args.stage2 = False

    train_dataset, val_dataset, test_dataset = load_datasets(args, debug)

    print(f"CPU count: {os.cpu_count()}")
    base_kwargs = {
        "num_workers": min(24, os.cpu_count()),
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    train_kwargs = {
        **base_kwargs,
        "collate_fn": custom_collate_fn,
    }

    if imbalance:
        train_dl = DataLoader(
            train_dataset, batch_size=args.bs, shuffle=True, **train_kwargs
        )
    else:
        batch_sampler = TrainBalancedBatchSampler(
            train_dataset, n_classes=4, n_samples=args.bs // 4, fold_index=args.fold
        )
        train_dl = DataLoader(
            train_dataset, batch_sampler=batch_sampler, **train_kwargs
        )

    val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, **base_kwargs)

    test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, **base_kwargs)

    print(f"\nDataset sizes for fold {args.fold}:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Valid: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    print(f"\nPeriods for fold {args.fold}:")
    print("Train periods:")
    for start, end in args.train_periods:
        print(f"  {start} to {end}")
    print("Validation periods:")
    for start, end in args.val_periods:
        print(f"  {start} to {end}")
    print("Test periods:")
    for start, end in args.test_periods:
        print(f"  {start} to {end}")

    sample = None
    return (train_dl, val_dl, test_dl), sample
