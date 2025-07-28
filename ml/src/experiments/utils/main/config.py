"""
Configuration utilities for main experiments
"""

import yaml
import os
import torch
from argparse import Namespace
from typing import Dict, Any, Tuple


def parse_params(args: Namespace, dump: bool = False) -> Tuple[Namespace, Dict[str, Any]]:
    """Parse YAML parameters and merge with command line arguments"""
    
    # Load YAML configuration
    with open(args.params, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Parse device
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )
    args.device = device

    # Default dataset settings
    dataset_config = {
        "force_preprocess": True,
        "force_recalc_indices": True,
        "force_recalc_stats": True,
    }

    # Merge dataset settings from YAML
    if "dataset" in yaml_config:
        dataset_config.update(yaml_config["dataset"])

    # Process model settings
    model_config = yaml_config.get("model", {})
    model_selected = model_config.get("selected")
    model_models = model_config.get("models", {})

    # Recursively convert model settings to Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    # Convert model configurations to Namespace objects
    model_models = {
        name: dict_to_namespace(config) for name, config in model_models.items()
    }

    model_namespace = Namespace(selected=model_selected, models=model_models)

    # Merge command line arguments and YAML settings
    args_dict = vars(args)

    # Add top-level settings (weight_decay, lr, epochs, bs, etc.)
    top_level_params = {
        k: v for k, v in yaml_config.items() if k not in ["dataset", "model"]
    }
    args_dict.update(top_level_params)

    # Add dataset and model settings
    args_dict["dataset"] = dataset_config
    args_dict["model"] = model_namespace

    args = Namespace(**args_dict)

    # Set stage2 parameters from YAML
    stage2_config = yaml_config.get("stage2", {})
    args.lr_for_2stage = stage2_config.get("lr")
    args.epoch_for_2stage = stage2_config.get("epochs")

    # Set periods based on fold number
    FOLD_PERIODS = {
        1: {
            "train": [("2011-12-01", "2012-05-31"), ("2012-06-01", "2019-05-31")],
            "val": [("2011-06-01", "2011-11-30"), ("2019-06-01", "2019-11-30")],
            "test": [("2019-12-01", "2021-11-30")],
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

    # Select periods for the specified fold
    selected_periods = FOLD_PERIODS[args.fold]
    args.train_periods = selected_periods["train"]
    args.val_periods = selected_periods["val"]
    args.test_periods = selected_periods["test"]

    # Build various paths
    args.data_path = os.path.join(args.data_root, "all_data_hours")
    args.features_path = os.path.join(
        args.data_root, "all_features/completed_old/all_features_history_672_step_1"
    )
    args.cache_root = os.path.join(args.data_root, "main")

    # Set imbalance based on stage
    if args.mode == "train" and args.resume_from_checkpoint:
        args.imbalance = True if args.stage == 1 else False

    if dump:
        print("Configuration:")
        print(yaml.dump(yaml_config, default_flow_style=False))

    return args, yaml_config


def namespace_to_dict(ns):
    """Convert Namespace to dict recursively."""
    from argparse import Namespace
    if isinstance(ns, Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, (list, tuple)):
        return [namespace_to_dict(x) for x in ns]
    else:
        return ns
