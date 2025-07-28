import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb
from typing import Optional


def setup_logging(trial_name: str, fold: int = 0, experiment_type: str = "main") -> logging.Logger:
    """Setup logging for experiments
    
    Args:
        trial_name: Trial name
        fold: Fold number
        experiment_type: Type of experiment ("main" or "pretrain")
        
    Returns:
        logging.Logger: Configured logger
    """
    log_dir = os.path.join("logs", experiment_type, trial_name, f"fold{fold}")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
    log_dir, f"{trial_name}_fold{fold}_{timestamp}.log"
    )

    logger = logging.getLogger(f"{experiment_type}_experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate messages

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_tensorboard(trial_name: str, fold: int = 0, experiment_type: str = "main") -> SummaryWriter:
    """Setup TensorBoard writer
    
    Args:
        trial_name: Trial name
        fold: Fold number
        experiment_type: Type of experiment ("main" or "pretrain")
        
    Returns:
        SummaryWriter: TensorBoard writer
    """
    log_dir = os.path.join("logs", experiment_type, trial_name, f"fold{fold}")
    return SummaryWriter(log_dir)


def setup_wandb(project_name: str, trial_name: str, fold: int = 0) -> None:
    """Setup Weights & Biases
    
    Args:
        project_name: W&B project name
        trial_name: Trial name
        fold: Fold number
    """
    run_name = f"{trial_name}_fold{fold}"
    wandb.init(project=project_name, name=run_name)


def close_wandb() -> None:
    """Close Weights & Biases run"""
    wandb.finish()
