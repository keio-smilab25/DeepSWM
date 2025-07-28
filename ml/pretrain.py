"""Pre-training script for Solar Flare Prediction Model"""


import argparse

from src.experiments.pretrain.experiment import PretrainExperimentManager
from src.experiments.utils.pretrain.config import (
    load_config,
    update_args_from_config,
)


def setup_pretrain_args():
    """Setup command line arguments for pretraining"""
    parser = argparse.ArgumentParser(description="Pre-train Solar Flare Prediction Model")
    
    # Essential arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Input data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--trial_name", type=str, required=True, help="Trial name")
    parser.add_argument("--data_root", type=str, default="/data2/01flare24", help="Data root directory")
    parser.add_argument("--fold", type=int, default=1, choices=[1,2,3,4,5], help="Fold number")
    
    # Training parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "visualize"], help="Mode")
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
    
    return parser.parse_args()


def main():
    """Main function for pre-training"""
    # Parse arguments locally
    args = setup_pretrain_args()
    
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
