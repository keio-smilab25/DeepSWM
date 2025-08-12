"""Train Deep Space Weather Model"""

import os
import warnings
import argparse
import torch.multiprocessing as mp

# Suppress torchvision beta transforms warnings
os.environ['TORCHVISION_DISABLE_BETA_TRANSFORMS_WARNING'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

from src.experiments.main.experiment import ExperimentManager
from src.experiments.utils.main.config import parse_params
from src.experiments.utils.main.inference import run_main_inference


def setup_main_args():
    """Setup command line arguments for main experiment"""
    parser = argparse.ArgumentParser(description="Train Solar Flare Prediction Model")
    
    # Essential arguments
    parser.add_argument("--params", type=str, required=True, help="Path to params.yaml")
    parser.add_argument("--trial_name", type=str, required=True, help="Trial name")
    parser.add_argument("--fold", type=int, required=True, choices=[1,2,3,4,5], help="Fold number")
    parser.add_argument("--data_root", type=str, default="/data2/01flare24", help="Data root directory")
    
    # Mode and stage
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "inference"], help="Mode")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Training stage")
    parser.add_argument("--imbalance", action="store_true", help="Use imbalanced dataset")
    
    # Dataset parameters
    parser.add_argument("--history", type=int, default=4, help="History length for sequences")
    
    # Training parameters
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adamw", "radam_free", "adam"], help="Optimizer to use")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--cosine_epochs", type=int, default=20, help="Number of epochs for cosine decay")
    parser.add_argument("--detail_summary", action="store_true", help="Show detailed model summary")
    parser.add_argument("--calculate_valid_indices", action="store_true", help="Calculate valid indices for each run")
    
    # Hardware settings
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device")
    
    # Optional settings
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Inference-specific
    parser.add_argument("--datetime", type=str, help="YYYYMMDD_HHMMSS: run inference for a single timestamp")
    
    return parser.parse_args()


def main():
    # Parse arguments locally
    args = setup_main_args()

    # Inference-only shortcut (no YAML parse/ExperimentManager needed)
    if args.mode == "inference":
        run_main_inference(args)
        return
    
    # Parse YAML parameters and merge with args
    args, _ = parse_params(args, dump=True)
    experiment = ExperimentManager(args)

    if args.mode == "train":
        if args.resume_from_checkpoint:
            experiment.logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            experiment.train()
        else:
            # Stage 1: Training with base model
            experiment.train()

            # Stage 2: Training with reduced learning rate (optional)
            if hasattr(args, 'lr_for_2stage') and args.lr_for_2stage:
                experiment.prepare_stage2_training(lr=args.lr_for_2stage)
                experiment.train(epochs=args.epoch_for_2stage)
        
        # Save predictions after training completion
        experiment.save_predictions()
    
    elif args.mode == "test":
        # Test the model using the best available checkpoint
        experiment.test_from_checkpoint()
        
        experiment.logger.info("Test completed")
    
    else:
        experiment.logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
