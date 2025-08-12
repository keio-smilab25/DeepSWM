"""Pre-training script for Solar Flare Prediction Model"""


import argparse
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
import h5py

from src.experiments.pretrain.experiment import PretrainExperimentManager
from src.experiments.utils.pretrain.config import (
    load_config,
    update_args_from_config,
)
from src.experiments.utils.pretrain.inference import run_pretrain_inference


def setup_pretrain_args():
    """Setup command line arguments for pretraining and inference"""
    parser = argparse.ArgumentParser(description="Pre-train Solar Flare Prediction Model")
    
    # Essential arguments
    parser.add_argument("--input_dir", type=str, help="Input data directory (training)")
    parser.add_argument("--output_dir", type=str, help="Output directory for features (training)")
    parser.add_argument("--trial_name", type=str, help="Trial name")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Data root directory (used by inference mode as ml/datasets)")
    parser.add_argument("--fold", type=int, default=1, choices=[1,2,3,4,5], help="Fold number")
    
    # Training parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "visualize", "inference"], help="Mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio")
    
    # Hardware settings
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    # Optional settings
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--visualize_timestamp", type=str, help="Timestamp for visualization")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation of cached data")

    # Inference (feature extraction) options
    parser.add_argument("--datetime", help="YYYYMMDD_HHMMSS: process a single hour (inference mode)")
    parser.add_argument("--date", help="YYYYMMDD: process all hours for the date (inference mode)")
    parser.add_argument("--pretrain_checkpoint", default="checkpoints/pretrain/SparseMAE.pth", help="SparseMAE checkpoint path (relative to ml/)")

    return parser.parse_args()



def main():
    """Main function for pre-training and inference"""
    # Parse arguments locally
    args = setup_pretrain_args()
    
    if args.mode == "inference":
        # Run feature extraction using SparseMAE
        run_pretrain_inference(args)
        return

    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        args = update_args_from_config(args, config)
    
    # Create experiment
    experiment = PretrainExperimentManager(args)
    
    if args.mode == "train":
        # Run pre-training
        experiment.train()
        
    elif args.mode == "test":
        # Test the pre-trained model
        experiment.test()
        
    elif args.mode == "visualize":
        # Visualize reconstruction results
        experiment.visualize()
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
