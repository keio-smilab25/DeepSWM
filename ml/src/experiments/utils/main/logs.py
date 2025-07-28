import wandb as wandb_runner
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import logging
import numpy as np
from torchinfo import summary
import csv

from src.experiments.utils.logs import setup_logging as common_setup_logging, setup_tensorboard, setup_wandb, close_wandb


@dataclass
class Log:
    stage: str
    loss: float
    score: Any


def setup_logging_main(trial_name: str, fold: int = 0) -> logging.Logger:
    """Setup logging for main experiment"""
    return common_setup_logging(trial_name, fold, "main")


class Logger:
    def __init__(self, args: Namespace, wandb: bool, logger: logging.Logger) -> None:
        if args.wandb:
            wandb_runner.init(project="SolarFlarePrediction2", name=args.trial_name)
        self.wandb_enabled = wandb
        self.logger = logger

    def write(self, epoch: int, logs: List[Log]):
        l: Dict[str, Any] = {"epoch": epoch}
        for lg in logs:
            l[f"{lg.stage}_loss"] = lg.loss
            l.update(lg.score)

        if self.wandb_enabled:
            wandb_runner.log(l)

    def print_model_summary(self, model, args=None, mock_sample=None):
        """Print model summary"""
        if args and args.detail_summary and mock_sample:
            summary(model, [(args.bs, *feature.shape) for feature in mock_sample[0]])
        else:
            summary(model)

    def print_best_metrics(self, experiment, stage: str = "1st"):
        """Output evaluation metrics for the best model"""
        self.logger.info(f"\n========== Best Model Metrics ({stage} stage) ==========")
        self.logger.info(f"Best Valid GMGS: {experiment.best_valid_gmgs:.4f}")
        self.logger.info(f"Train Loss: {experiment.best_train_loss:.4f}")
        self.logger.info(f"Valid Loss: {experiment.best_valid_loss:.4f}")

        # Output scores in specified order
        metrics = ["GMGS", "BSS", "TSS", "ACC"]
        
        # Check if scores are available
        if (experiment.train_score is not None and 
            experiment.valid_score is not None and 
            experiment.test_score is not None):
            
            self.logger.info(f"\n--- Metrics ---")
            for metric in metrics:
                if (metric in experiment.train_score and 
                    metric in experiment.valid_score and 
                    metric in experiment.test_score):
                    self.logger.info(
                        f"{metric} - Train: {experiment.train_score[metric]:.4f}, "
                        f"Valid: {experiment.valid_score[metric]:.4f}, "
                        f"Test: {experiment.test_score[metric]:.4f}"
                    )

        self.logger.info("=" * 50)

    def save_final_metrics(self, experiment, output_dir: str, trial_name: str):
        """Save final metrics to CSV file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare metrics data
        metrics_data = []
        metrics = ["GMGS", "BSS", "TSS", "ACC"]
        
        for i, (train_score, valid_score, test_score) in enumerate(
            zip(experiment.train_score, experiment.valid_score, experiment.test_score)
        ):
            for metric in metrics:
                if metric in train_score:
                    metrics_data.append({
                        'trial_name': trial_name,
                        'fold': i,
                        'metric': metric,
                        'train': train_score[metric],
                        'valid': valid_score[metric],
                        'test': test_score[metric],
                        'best_valid_gmgs': experiment.best_valid_gmgs,
                        'train_loss': experiment.best_train_loss,
                        'valid_loss': experiment.best_valid_loss
                    })
        
        # Save to CSV
        metrics_file = os.path.join(output_dir, f"{trial_name}_final_metrics.csv")
        with open(metrics_file, 'w', newline='') as csvfile:
            fieldnames = ['trial_name', 'fold', 'metric', 'train', 'valid', 'test', 
                         'best_valid_gmgs', 'train_loss', 'valid_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in metrics_data:
                writer.writerow(row)
        
        self.logger.info(f"Final metrics saved to: {metrics_file}")

    def finish(self):
        """Finish logging"""
        if self.wandb_enabled:
            close_wandb()
