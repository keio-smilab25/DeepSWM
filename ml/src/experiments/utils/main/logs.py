import wandb as wandb_runner
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import logging
from torchinfo import summary
import csv
from sklearn.metrics import confusion_matrix

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
        try:
            if args and args.detail_summary and mock_sample:
                model_summary = summary(model, [(args.bs, *feature.shape) for feature in mock_sample[0]], verbose=0)
                self.logger.info(f"Detailed Model Summary:\n{model_summary}")
            else:
                self.logger.info("Generating basic model summary...")
                model_summary = summary(model, verbose=0)
                self.logger.info(f"Basic Model Summary:\n{model_summary}")
        except Exception as e:
            self.logger.warning(f"Could not generate model summary via Logger: {e}")
            self.logger.info(f"Model structure:\n{model}")

    def print_best_metrics(self, experiment, stage: str = "1st"):
        """Output evaluation metrics for the best model"""
        self.logger.info(f"========== Best Model Metrics ({stage} stage) ==========")
        self.logger.info(f"Best Valid GMGS: {experiment.best_valid_gmgs:.4f}")
        self.logger.info(f"Train Loss: {experiment.best_train_loss:.4f}")
        self.logger.info(f"Valid Loss: {experiment.best_valid_loss:.4f}")

        # Output scores in specified order (like the old implementation)
        metrics = ["GMGS", "BSS", "TSS", "ACC"]
        
        # Valid scores - use the correct format from statistics module
        if hasattr(experiment, 'best_valid_score') and experiment.best_valid_score:
            valid_scores = []
            for metric in metrics:
                # Use the "valid_METRIC" format as returned by stat.aggregate()
                if f"valid_{metric}" in experiment.best_valid_score:
                    valid_scores.append(f"{metric}: {experiment.best_valid_score[f'valid_{metric}']:.4f}")
            if valid_scores:
                self.logger.info(f"Valid Scores: {' '.join(valid_scores)}")
            else:
                self.logger.info("Valid Scores: ")
        else:
            self.logger.info("Valid Scores: ")
        
        # Test scores - use the correct format from statistics module
        if hasattr(experiment, 'test_score') and experiment.test_score:
            test_scores = []
            for metric in metrics:
                # Use the "test_METRIC" format as returned by stat.aggregate()
                if f"test_{metric}" in experiment.test_score:
                    test_scores.append(f"{metric}: {experiment.test_score[f'test_{metric}']:.4f}")
            if test_scores:
                self.logger.info(f"Test Scores: {' '.join(test_scores)}")
            else:
                self.logger.info("Test Scores: ")
        
        # Output confusion matrices
        self.logger.info("")
        self.logger.info("Confusion Matrices:")
        
        # Valid confusion matrix using stat.confusion_matrix method
        if hasattr(experiment, 'best_valid_observations') and hasattr(experiment, 'best_valid_predictions'):
            self.logger.info("Valid Confusion Matrix:")
            try:
                # Try using experiment.stat.confusion_matrix method first
                if hasattr(experiment, 'stat') and hasattr(experiment.stat, 'confusion_matrix'):
                    valid_cm = experiment.stat.confusion_matrix(
                        experiment.best_valid_predictions,
                        experiment.best_valid_observations,
                    )
                    for row in valid_cm:
                        self.logger.info(f"    {row}")
                else:
                    # Fallback to sklearn
                    self._log_confusion_matrix(experiment.best_valid_observations, experiment.best_valid_predictions)
            except Exception as e:
                self.logger.warning(f"Could not generate valid confusion matrix: {e}")
        
        # Test confusion matrix using stat.confusion_matrix method
        if hasattr(experiment, 'test_observations') and hasattr(experiment, 'test_predictions'):
            self.logger.info("")
            self.logger.info("Test Confusion Matrix:")
            try:
                # Try using experiment.stat.confusion_matrix method first
                if hasattr(experiment, 'stat') and hasattr(experiment.stat, 'confusion_matrix'):
                    test_cm = experiment.stat.confusion_matrix(
                        experiment.test_predictions,
                        experiment.test_observations,
                    )
                    for row in test_cm:
                        self.logger.info(f"    {row}")
                else:
                    # Fallback to sklearn
                    self._log_confusion_matrix(experiment.test_observations, experiment.test_predictions)
            except Exception as e:
                self.logger.warning(f"Could not generate test confusion matrix: {e}")
        
        self.logger.info("=" * 50)
    
    def _log_confusion_matrix(self, observations, predictions):
        """Log confusion matrix in the desired format"""
        try:
            
            # Get unique classes
            classes = sorted(set(observations) | set(predictions))
            cm = confusion_matrix(observations, predictions, labels=classes)
            
            # Log each row of the confusion matrix
            for row in cm:
                row_str = "    [" + " ".join(f"{val:4d}" for val in row) + "]"
                self.logger.info(row_str)
                
        except Exception as e:
            self.logger.warning(f"Could not generate confusion matrix: {e}")

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
