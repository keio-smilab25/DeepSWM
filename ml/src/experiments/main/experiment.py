"""Experiment management utilities"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from argparse import Namespace
from typing import Optional, Tuple, Any

from src.experiments.utils.main.statistics import Stat, compute_statistics
from src.experiments.utils.utils import fix_seed, analyze_model_complexity, get_model_summary, is_better_score
from src.experiments.utils.main.utils import process_predictions_and_observations, get_model_class
from src.experiments.utils.main.scheduler import create_optimizer
from src.experiments.utils.main.losses import LossConfig, Losser
from src.experiments.utils.main.config import namespace_to_dict
from src.experiments.main.engine import train_epoch, eval_epoch
from src.experiments.utils.main.logs import Log, Logger, setup_logging_main
from src.data.utils.main.io import save_checkpoint, load_checkpoint, save_test_results, save_predictions
from src.data.main.dataloader import prepare_dataloaders


class ExperimentManager:
    """Manager class for Deep Space Weather Model"""

    def __init__(self, args: Namespace):
        # Setup logging
        self.logger = setup_logging_main(args.trial_name)
        self.logger.info(f"Using fold {args.fold}")
        self.logger.info("Dataset configuration:")
        self.logger.info(f"force_preprocess: {args.dataset.get('force_preprocess')}")
        self.logger.info(f"force_recalc_indices: {args.dataset.get('force_recalc_indices')}")
        self.logger.info(f"force_recalc_stats: {args.dataset.get('force_recalc_stats')}")

        self.log_writer = Logger(args, wandb=args.wandb, logger=self.logger)
        fix_seed(seed=42)

        if args.wandb:
            wandb.init(
                project="SolarFlarePrediction2",
                name=f"{args.trial_name}_fold{args.fold}",
            )

        self.current_stage = args.stage
        self.args = args

        # Calculate statistics
        stats_dir = os.path.join(args.cache_root, "statistics", f"fold{args.fold}")
        full_climatology, gmgs_score_matrix, stat = compute_statistics(
            data_dir=args.data_path,
            stats_dir=stats_dir,
            train_periods=args.train_periods,
            force_recalc=args.dataset["force_recalc_stats"],
            logger=self.logger,
        )
        self.stat = stat

        # Initialize dataloaders only if no checkpoint to resume from
        if args.mode == "train" and not args.resume_from_checkpoint:
            sample = self.load_dataloaders(args, args.imbalance)
        else:
            args.detail_summary = False
            sample = None

        # Prepare model and optimizer
        model, losser, optimizer, scheduler, stat = self._build(args, sample)

        self.model = model
        self.losser = losser
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stat = stat
        self.best_valid_gmgs = float("-inf")
        self.best_train_loss = None
        self.best_valid_loss = None
        self.best_valid_score = None
        self.train_score = None
        self.valid_score = None
        self.test_score = None
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

        # Add attributes for confusion matrix
        self.best_valid_predictions = None
        self.best_valid_observations = None
        self.test_predictions = None
        self.test_observations = None

        # Add variables for early stopping
        self.es_metric = args.early_stopping["metric"]
        self.patience = args.early_stopping["patience"]
        self.patience_counter = 0
        self.should_stop = False

        # Initialize best score based on metric
        self.best_metric_value = (
            float("-inf") if "GMGS" in self.es_metric else float("inf")
        )

        # Add variable for stage management
        self.current_stage = 1

        # Log configuration
        self.logger.info("\n=== Configuration ===")
        self.logger.info(yaml.dump(vars(args), default_flow_style=False))
        self.logger.info("==================\n")

    def _build(self, args: Namespace, sample: Any) -> Tuple[nn.Module, Losser, Any, Any, Stat]:
        """Build model, losser, optimizer, scheduler, stat"""
        print("Prepare model and optimizer", end="")
        
        loss_config = LossConfig(
            lambda_bss=args.factor["BS"],
            lambda_gmgs=args.factor["GMGS"],
            lambda_ce=args.factor["CE"],
            score_mtx=self.stat.gmgs_score_matrix,
            fold=args.fold,
            class_weights=args.class_weights,
            model_name=args.model.selected,
            stage=self.current_stage,
        )

        # Model
        Model = get_model_class(args.model.selected)
        architecture_params = namespace_to_dict(
            args.model.models[args.model.selected].architecture_params
        )
        model = Model(**architecture_params).to(args.device)

        # Analyze model complexity
        analyze_model_complexity(model, args.device, self.logger)
        get_model_summary(model, self.logger)

        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer(
            model, 
            args.optimizer, 
            args.lr, 
            args.weight_decay,
            getattr(args, 'warmup_epochs', 10),
            getattr(args, 'cosine_epochs', 100)
        )

        losser = Losser(loss_config, device=args.device)
        stat = self.stat

        print(" ... ok")
        return model, losser, optimizer, scheduler, stat

    def train_epoch(self, epoch):
        (train_dl, val_dl, _) = self.dataloaders

        # Set to training mode
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

        # train
        train_score, train_loss = train_epoch(
            self.model,
            self.optimizer,
            train_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
        )

        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        # validation
        valid_score, valid_loss = eval_epoch(
            self.model,
            val_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
            mode="valid",
            optimizer=self.optimizer, 
        )

        if self.es_metric == "valid_GMGS":
            current_metric_value = valid_score["valid_GMGS"]
        else:  # valid_loss
            current_metric_value = np.mean(valid_loss)

        if is_better_score(current_metric_value, self.best_metric_value, self.es_metric):
            self.best_metric_value = current_metric_value
            self.patience_counter = 0 

            best_checkpoint_path = save_checkpoint(
                self.model, self.optimizer, self.best_valid_gmgs, self.current_stage, self.args
            )
            self.logger.info(
                f"New best model (stage {self.current_stage}) saved with "
                f"{self.es_metric}: {current_metric_value:.4f}"
            )

            # Process when the best score is updated
            self.best_train_loss = np.mean(train_loss)
            self.best_valid_loss = np.mean(valid_loss)
            self.best_valid_score = valid_score
            
            # Get predictions and observations from stat
            valid_predictions = np.array(self.stat.predictions["valid"])
            valid_observations = np.array(self.stat.observations["valid"])
            
            # Process predictions and observations
            processed_pred, processed_obs = process_predictions_and_observations(
                valid_predictions, valid_observations, self.stat
            )
            self.best_valid_predictions = processed_pred
            self.best_valid_observations = processed_obs
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs without "
                    f"improvement in {self.es_metric}"
                )

        self.log_writer.write(
            epoch,
            [
                Log("train", np.mean(train_loss), train_score),
                Log("valid", np.mean(valid_loss), valid_score),
            ],
        )

        self.logger.info(
            f"Epoch {epoch}: Train loss:{np.mean(train_loss):.4f}  Valid loss:{np.mean(valid_loss):.4f}"
        )
        self.logger.info(
            f"Epoch {epoch}: Train score:{train_score}  Valid score:{valid_score}"
        )

        if self.args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": np.mean(train_loss),
                    **train_score,
                    "valid_loss": np.mean(valid_loss),
                    **valid_score,
                }
            )

        self.train_score = train_score
        self.valid_score = valid_score
        self.train_loss = np.mean(train_loss)
        self.valid_loss = np.mean(valid_loss)

    def train(self, lr: Optional[float] = None, epochs: Optional[int] = None):
        """Train the model for the specified number of epochs"""
        if self.current_stage == 1:
            lr = lr or self.args.lr
            epochs = epochs or self.args.epochs
        else:
            lr = lr or self.args.lr_for_2stage
            epochs = epochs or self.args.epoch_for_2stage

        # Load checkpoint if resuming
        start_epoch = 0
        if self.args.resume_from_checkpoint and os.path.exists(self.args.resume_from_checkpoint):
            self.logger.info(f"Loading checkpoint: {self.args.resume_from_checkpoint}")
            config, self.best_valid_gmgs, start_epoch = load_checkpoint(
                self.model,
                self.optimizer,
                self.args.resume_from_checkpoint,
                self.args.device,
                scheduler=self.scheduler,
            )
            self.logger.info(f"Resumed from epoch {start_epoch}")

        self.logger.info(f"Starting stage {self.current_stage} training")
        self.logger.info(f"Learning rate: {lr}, Epochs: {epochs}")

        for epoch in range(start_epoch, epochs):
            if self.should_stop:
                self.logger.info("Early stopping triggered")
                break

            self.logger.info(f"====== Epoch {epoch} ======")
            self.train_epoch(epoch)

            # Execute scheduler step for AdamW
            if self.scheduler and self.args.optimizer == "adamw":
                self.scheduler.step()

        # Evaluate best model
        self.logger.info(f"\n=== Evaluating stage {self.current_stage} ===")
        best_checkpoint_path = os.path.join(
            "checkpoints",
            "main",
            f"{self.args.trial_name}_stage{self.current_stage}_best.pth",
        )
        config, self.best_valid_gmgs, _ = load_checkpoint(
            self.model,
            self.optimizer,
            best_checkpoint_path,
            self.args.device,
            scheduler=self.scheduler,
        )
        self.test(save_qualitative=False)
        self.print_best_metrics(stage=f"{self.current_stage}st")
        
        # Save predictions after training completion
        self.save_predictions(stage=self.current_stage)

    def load_dataloaders(self, args: Namespace, imbalance: bool):
        dataloaders, sample = prepare_dataloaders(args, args.debug, imbalance)
        self.train_dl, self.valid_dl, self.test_dl = dataloaders 
        self.dataloaders = dataloaders
        return sample

    def test(self, save_qualitative: bool = False):
        """Test model and log results to wandb"""
        self.test_score, test_loss = eval_epoch(
            self.model,
            self.test_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
            mode="test",
            save_qualitative=save_qualitative,
            trial_name=self.args.trial_name,
            optimizer=self.optimizer,
        )
        self.test_loss = np.mean(test_loss)

        # Get test predictions and observations from stat
        test_predictions = np.array(self.stat.predictions["test"])
        test_observations = np.array(self.stat.observations["test"])
        
        # Process predictions and observations
        self.test_predictions, self.test_observations = process_predictions_and_observations(
            test_predictions, test_observations, self.stat
        )

        if self.args.wandb:
            wandb.log(
                {
                    "test_loss": self.test_loss,
                    **self.test_score,
                }
            )

    def test_from_checkpoint(self, checkpoint_path: str = None, save_qualitative: bool = True):
        """Load best checkpoint and test the model"""
        # Determine checkpoint path
        if checkpoint_path is None:
            # Try to find the best checkpoint (prefer stage2 over stage1)
            stage2_path = f"checkpoints/main/{self.args.trial_name}_stage2_best.pth"
            stage1_path = f"checkpoints/main/{self.args.trial_name}_stage1_best.pth"
            
            if os.path.exists(stage2_path):
                checkpoint_path = stage2_path
                self.current_stage = 2
            elif os.path.exists(stage1_path):
                checkpoint_path = stage1_path
                self.current_stage = 1
            else:
                self.logger.error("No checkpoint found for testing")
                return
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint for testing: {checkpoint_path}")
            config, self.best_valid_gmgs, _ = load_checkpoint(
                self.model,
                self.optimizer,
                checkpoint_path,
                self.args.device,
                scheduler=self.scheduler,
            )
            self.logger.info(f"Loaded checkpoint successfully")
        else:
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Run test
        self.test(save_qualitative=save_qualitative)
        
        # Save predictions
        self.save_predictions()
        
        self.logger.info("Test completed successfully")

    def save_predictions(self, stage: int = None):
        """Save predictions and observations for analysis"""
        stage = stage or self.current_stage
        
        # Call the save_predictions function from io.py
        results_dir = save_predictions(
            best_valid_predictions=getattr(self, 'best_valid_predictions', None),
            best_valid_observations=getattr(self, 'best_valid_observations', None),
            best_valid_score=getattr(self, 'best_valid_score', None),
            best_valid_loss=getattr(self, 'best_valid_loss', None),
            test_predictions=getattr(self, 'test_predictions', None),
            test_observations=getattr(self, 'test_observations', None),
            test_score=getattr(self, 'test_score', None),
            test_loss=getattr(self, 'test_loss', None),
            trial_name=self.args.trial_name,
            fold=self.args.fold,
            stage=stage
        )
        
        return results_dir

    def freeze_feature_extractor(self):
        """Freeze feature extractor"""
        self.model.freeze_feature_extractor()

    def reset_optimizer(self, lr: Optional[float] = None):
        """Reset optimizer with new lr"""
        self.optimizer, self.scheduler = create_optimizer(
            self.model, 
            self.args.optimizer, 
            lr or self.args.lr, 
            self.args.weight_decay,
            getattr(self.args, 'warmup_epochs', 10),
            getattr(self.args, 'cosine_epochs', 100)
        )

        loss_config = LossConfig(
            lambda_bss=self.args.factor["BS"],
            lambda_gmgs=self.args.factor["GMGS"],
            lambda_ce=self.args.factor["CE"],
            score_mtx=self.stat.gmgs_score_matrix,
            fold=self.args.fold,
            class_weights=self.args.class_weights,
            model_name=self.args.model.selected,
            stage=2,
        )

        self.losser = Losser(loss_config, self.args.device)

    def print_best_metrics(self, stage: str = "1st"):
        """Output evaluation metrics for the best model"""
        self.log_writer.print_best_metrics(self, stage) 
