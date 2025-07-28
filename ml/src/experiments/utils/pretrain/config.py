"""
Configuration utilities for pretraining
"""

import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Parameters:
        config_path: Path to the YAML configuration file

    Returns:
        config: Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_args_from_config(args: Any, config: Dict[str, Any]) -> Any:
    """
    Update arguments with values from config

    Parameters:
        args: Argument namespace
        config: Configuration dictionary

    Returns:
        args: Updated argument namespace
    """
    # Set default values for critical parameters
    default_values = {
        "trial_name": "mae_default",
        "mode": "train",
        "fold": 1,
        "data_root": "data",
        "input_dir": "data/solar_flare",
        "output_dir": "results/features",
        "batch_size": 32,
        "num_workers": 4,
        "epochs": 20,
        "mask_ratio": 0.75,
        "cuda_device": 0,
    }

    # First apply default values if not set
    for key, value in default_values.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Then update with config values
    for key, value in config.items():
        if not isinstance(value, dict):
            # Only update if the command line argument wasn't explicitly provided
            if not hasattr(args, key) or getattr(args, key) == default_values.get(
                key, None
            ):
                setattr(args, key, value)

    # Handle nested dictionaries
    if "model" in config:
        args.model_config = config["model"]

    if "visualize" in config:
        args.visualize_config = config["visualize"]

        # Set visualize_timestamp if it exists in config and not explicitly set in args
        if (
            not hasattr(args, "visualize_timestamp") or args.visualize_timestamp is None
        ) and "timestamp" in config["visualize"]:
            args.visualize_timestamp = config["visualize"]["timestamp"]

    if "feature_extraction" in config:
        args.feature_extraction_config = config["feature_extraction"]

    return args


def get_periods_from_config(config: Dict[str, Any], fold: int) -> Dict[str, list]:
    """
    Get periods for the specified fold from config

    Parameters:
        config: Configuration dictionary
        fold: Fold number

    Returns:
        periods: Dictionary containing periods for train, val, and test
    """
    if "fold_periods" not in config or fold not in config["fold_periods"]:
        # Default periods if not specified in config
        FOLD_PERIODS = {
            1: {
                "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2019-05-31")],
                "val": [("2011-06-01", "2011-11-30"), ("2019-06-01", "2019-11-30")],
                "test": [("2019-12-01", "2022-11-30")],
            },
            2: {
                "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2019-11-30")],
                "val": [("2011-06-01", "2011-11-30"), ("2019-12-01", "2020-05-31")],
                "test": [("2020-06-01", "2022-05-31")],
            },
            3: {
                "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2020-05-31")],
                "val": [("2011-06-01", "2011-11-30"), ("2020-06-01", "2020-11-30")],
                "test": [("2020-12-01", "2022-11-30")],
            },
            4: {
                "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2020-11-30")],
                "val": [("2011-06-01", "2011-11-30"), ("2020-12-01", "2021-05-31")],
                "test": [("2021-06-01", "2023-05-31")],
            },
            5: {
                "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2021-05-31")],
                "val": [("2011-06-01", "2011-11-30"), ("2021-06-01", "2021-11-30")],
                "test": [("2021-12-01", "2023-11-30")],
            },
        }
        return FOLD_PERIODS.get(fold, FOLD_PERIODS[1])

    # Convert list of lists to list of tuples for periods
    periods = {}
    for split, period_list in config["fold_periods"][fold].items():
        periods[split] = [tuple(period) for period in period_list]

    return periods
