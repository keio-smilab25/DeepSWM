"""Pretraining experiment management utilities"""

import os
import torch
from torchinfo import summary
from typing import Dict, Any, List, Tuple, Optional

from src.models.pretrain.sparsemae import (
    mae_vit_base_patch16_dec512d8b,
    vit_for_FT64d4b,
    vit_for_FT32d4b,
    vit_for_FT128db,
)
from src.data.pretrain.dataloader import (
    setup_datasets,
    setup_dataloaders,
    setup_all_data_loader,
    setup_visualization_loader,
    parse_time_range,
)
from src.experiments.utils.pretrain.losses import Losser
from src.experiments.utils.pretrain.logs import PretrainLogger
from src.experiments.utils.pretrain.io import setup_checkpoint_dir, load_model
from src.experiments.utils.pretrain.config import get_periods_from_config
from src.experiments.pretrain.engine import (
    train_mae,
    eval_epoch,
    process_features,
    visualize_model_outputs,
    process_all_features,
    run_pretrain_workflow,
)


class PretrainExperimentManager:
    """Manager class for Deep Space Weather Model pretraining experiments"""

    def __init__(self, args):
        self.args = args
        
        # Set up logger
        self.logger = PretrainLogger(args.trial_name, args.fold, args.use_wandb)
        
        # Set up cache directories
        self.cache_dirs = self._setup_cache_dirs()
        
        # Set up checkpoint directory
        self.checkpoint_dir = setup_checkpoint_dir(args.trial_name)
        
        # Log configuration
        self.logger.log_config(args)
        
        # Set up device
        self.device = self._setup_device()
        
        # Get periods from config
        self.periods = self._get_periods()
        
        # Log system information
        self._log_system_info()
        
        # Set up model
        self.model = self._setup_model()
        
        # Log model summary
        self._log_model_summary()

    def _setup_cache_dirs(self) -> Dict[str, str]:
        """Set up cache directory paths"""
        cache_base = os.path.join(self.args.data_root, f"pretrain/cache/fold{self.args.fold}")
        cache_dirs = {
            "train": os.path.join(cache_base, "train"),
            "val": os.path.join(cache_base, "val"),
            "test": os.path.join(cache_base, "test"),
        }
        for dir_path in cache_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return cache_dirs

    def _setup_device(self) -> torch.device:
        """Set up computation device"""
        # Ensure cuda_device is set
        if not hasattr(self.args, "cuda_device") or self.args.cuda_device is None:
            self.args.cuda_device = 0

        device = torch.device(
            f"cuda:{self.args.cuda_device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger.log_info(f"Using device: {device}")
        return device

    def _get_periods(self) -> Dict[str, List]:
        """Get periods from config"""
        config = getattr(self.args, 'config_dict', {})
        periods = get_periods_from_config(config, self.args.fold)
        return {
            "train": periods.get("train", []),
            "val": periods.get("val", []),
            "test": periods.get("test", []),
        }

    def _log_system_info(self):
        """Log PyTorch and CUDA system information"""
        self.logger.log_info(f"PyTorch Version: {torch.__version__}")
        self.logger.log_info(f"PyTorch CUDA Version: {torch.version.cuda}")
        self.logger.log_info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.log_info(f"CUDA Device Count: {torch.cuda.device_count()}")
            device_idx_to_check = getattr(self.args, 'cuda_device', 0)
            
            if device_idx_to_check < torch.cuda.device_count():
                self.logger.log_info(f"Checking properties for CUDA Device: {device_idx_to_check}")
                self.logger.log_info(f"Device Name: {torch.cuda.get_device_name(device_idx_to_check)}")
                self.logger.log_info(f"Compute Capability: {torch.cuda.get_device_capability(device_idx_to_check)}")
            else:
                self.logger.log_info(f"Configured CUDA device index {device_idx_to_check} is out of range. Available devices: {torch.cuda.device_count()}.")
        else:
            self.logger.log_info("CUDA is not available to PyTorch.")

    def _get_model_from_config(self) -> torch.nn.Module:
        """Create model based on configuration"""
        model_config = getattr(self.args, "model_config", {})
        model_type = model_config.get("type", "vit_for_FT128db")

        # Common parameters
        common_params = {
            "in_chans": model_config.get("in_chans", 10),
            "mask_ratio": model_config.get("mask_ratio", getattr(self.args, "mask_ratio", 0.75)),
            "stdwise": model_config.get("stdwise", False),
            "pyramid": model_config.get("pyramid", True),
            "sunspot": model_config.get("sunspot", True),
            "base_mask_ratio": model_config.get("base_mask_ratio", 0.5),
            "sunspot_spatial_ratio": model_config.get("sunspot_spatial_ratio", 0.35),
            "feature_mask_ratio": model_config.get("feature_mask_ratio", 0.75),
        }

        # Create model based on type
        if model_type == "vit_for_FT128db":
            model = vit_for_FT128db(**common_params)
        elif model_type == "vit_for_FT64d4b":
            model = vit_for_FT64d4b(**common_params)
        elif model_type == "vit_for_FT32d4b":
            model = vit_for_FT32d4b(**common_params)
        elif model_type == "mae_vit_base_patch16_dec512d8b":
            model = mae_vit_base_patch16_dec512d8b(**common_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def _setup_model(self) -> torch.nn.Module:
        """Set up model from config or use default"""
        if hasattr(self.args, "model_config"):
            return self._get_model_from_config()
        else:
            # Fallback to default model
            return vit_for_FT128db(
                in_chans=10,
                mask_ratio=getattr(self.args, "mask_ratio", 0.75),
                stdwise=False,
                pyramid=True,
                sunspot=True,
                base_mask_ratio=0.5,
                sunspot_spatial_ratio=0.3,
                feature_mask_ratio=0.5,
            ).to(self.device)

    def _log_model_summary(self):
        """Log model summary"""
        batch_size = getattr(self.args, "batch_size", 32)
        model_summary = summary(
            self.model, input_size=(batch_size, 10, 256, 256), verbose=0
        )
        self.logger.log_model_summary(model_summary)

    def train(self):
        """Run training workflow"""
        # Set up datasets and dataloaders
        train_dataset, val_dataset, test_dataset = setup_datasets(
            self.args, self.cache_dirs, 
            self.periods["train"], self.periods["val"], self.periods["test"]
        )
        train_loader, val_loader, test_loader = setup_dataloaders(
            self.args, train_dataset, val_dataset, test_dataset
        )

        # Run pretraining workflow
        self.model = run_pretrain_workflow(
            self.args, self.model, train_loader, val_loader, test_loader, 
            self.checkpoint_dir, self.logger
        )

    def test(self):
        """Run test evaluation"""
        # Set up datasets and dataloaders
        _, _, test_dataset = setup_datasets(
            self.args, self.cache_dirs, 
            self.periods["train"], self.periods["val"], self.periods["test"]
        )
        _, _, test_loader = setup_dataloaders(self.args, None, None, test_dataset)

        # Load model
        self.model = load_model(self.model, self.checkpoint_dir, self.args.trial_name)
        self.model.eval()

        # Test evaluation
        losser = Losser(self.model, device=self.device)
        test_metrics = eval_epoch(self.model, test_loader, losser)
        self.logger.log_final_metrics(test_metrics)

    def visualize(self):
        """Run visualization"""
        vis_periods = self.periods["test"]  # Visualization periods are the same as test periods
        
        # Set up visualization dataloader
        test_loader = setup_visualization_loader(self.args, self.cache_dirs, vis_periods)

        # Filter by timestamp if specified
        time_range = None
        if hasattr(self.args, 'visualize_timestamp') and self.args.visualize_timestamp:
            visualize_config = getattr(self.args, 'visualize_config', {})
            time_range = parse_time_range(
                self.args.visualize_timestamp,
                hours_before=visualize_config.get("hours_before", 12),
                hours_after=visualize_config.get("hours_after", 12),
            )

            if time_range:
                start_time, end_time = time_range
                self.logger.log_info(
                    f"Using time range around {self.args.visualize_timestamp}: {start_time} to {end_time}."
                )

        # Load model
        self.model = load_model(self.model, self.checkpoint_dir, self.args.trial_name).to(self.device)
        self.model.eval()

        # Run visualization
        visualize_config = getattr(self.args, 'visualize_config', {})
        visualize_model_outputs(
            self.model,
            test_loader,
            self.device,
            os.path.join("results", "reconstruct_images", self.args.trial_name),
            self.args.trial_name,
            num_images=visualize_config.get("num_images", 30),
            use_sunspot_masking=visualize_config.get("use_sunspot_masking", True),
            time_range=time_range,
        )

    def embed(self):
        """Extract features for train/val/test sets"""
        # Set up datasets and dataloaders
        train_dataset, val_dataset, test_dataset = setup_datasets(
            self.args, self.cache_dirs, 
            self.periods["train"], self.periods["val"], self.periods["test"]
        )
        train_loader, val_loader, test_loader = setup_dataloaders(
            self.args, train_dataset, val_dataset, test_dataset
        )

        # Load model
        self.model = load_model(self.model, self.checkpoint_dir, self.args.trial_name)
        self.model.eval()

        # Extract features
        process_all_features(
            self.model,
            [train_loader, val_loader, test_loader],
            ["train", "val", "test"],
            getattr(self.args, "mask_ratio", 0.75),
            self.device,
            self.args.output_dir,
            self.args.trial_name,
        )

    def inference_all(self):
        """Extract features for all data"""
        # Set up all data loader
        all_loader = setup_all_data_loader(self.args, self.cache_dirs)

        # Load model
        self.model = load_model(self.model, self.checkpoint_dir, self.args.trial_name)
        self.model.eval()

        # Extract features for all data
        process_features(
            self.model,
            all_loader,
            getattr(self.args, "mask_ratio", 0.75),
            self.device,
            self.args.output_dir,
            "all_data",
            self.args.trial_name,
        ) 
