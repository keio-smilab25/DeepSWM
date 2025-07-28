"""
Training and evaluation functions for pre-training
"""

import torch
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
import math

from src.experiments.utils.pretrain.losses import Losser
from src.experiments.utils.pretrain.io import save_model, load_model, apply_model_state
from src.features.utils.feature_extraction import process_features, process_all_features
from src.experiments.utils.pretrain.visualize import visualize_model_outputs


def train_epoch(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_dl: DataLoader,
    losser: Losser,
) -> Dict[str, float]:
    """
    Perform training for one epoch

    Parameters:
        model: Model to train
        optimizer: Optimizer
        train_dl: Training data loader
        losser: Loss function

    Returns:
        metrics: Training metrics
    """
    model.train()
    losser.clear()
    total_loss = 0.0
    num_batches = 0
    count = 0

    for _, (x, y, _) in enumerate(tqdm(train_dl, desc="Training")):
        # if count > 10:
        #     break
        # count += 1
        
        optimizer.zero_grad()
        imgs = x.to(losser.device).float()
        pred, mask = model(imgs)
        loss = losser(imgs, pred, mask)

        if loss is not None:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    metrics = {
        "mse": avg_loss,
        "mae": losser.get_metrics()["mae"],
        "solar_mse": losser.get_metrics()["solar_mse"],
        "solar_mae": losser.get_metrics()["solar_mae"],
    }

    return metrics


def eval_epoch(
    model: torch.nn.Module, val_dl: DataLoader, losser: Losser
) -> Dict[str, float]:
    """
    Evaluate the model

    Parameters:
        model: Model to evaluate
        val_dl: Evaluation data loader
        losser: Loss function

    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    losser.clear()
    count = 0

    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl, desc="Evaluating")):
            # if count > 10:
            #     break
            # count += 1
            
            imgs = x.to(losser.device).float()
            pred, mask = model(imgs)
            losser.evaluate(imgs, pred, mask)

    return losser.get_metrics()


def train_mae(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    losser: Losser,
    trial_name: str,
    checkpoint_dir: str,
    num_epochs: int = 10,
    logger: Optional[Any] = None,
) -> torch.nn.Module:
    """
    Train the MAE model

    Parameters:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        losser: Loss function
        trial_name: Trial name
        checkpoint_dir: Checkpoint directory
        num_epochs: Number of epochs
        logger: Logger

    Returns:
        model: Trained model
    """
    best_val_loss = float("inf")
    best_model = None

    for e in range(num_epochs):
        # Training
        train_metrics = train_epoch(model, optimizer, train_loader, losser)
        lr_scheduler.step()

        # Validation
        val_metrics = eval_epoch(model, val_loader, losser)

        # Log using logger
        is_best = val_metrics["solar_mse"] < best_val_loss
        if logger:
            logger.log_train_step(
                e,
                train_metrics["mse"],
                val_metrics,
                optimizer.param_groups[0]["lr"],
                is_best=is_best,
            )

        # Save the best model
        if is_best:
            best_val_loss = val_metrics["solar_mse"]
            best_model = model.state_dict().copy()
            save_model(model, checkpoint_dir, trial_name, is_best=True)

            if logger:
                logger.log_info(
                    f"Epoch {e}: New best model with val_solar_mse = {best_val_loss:.6f}"
                )

    # Final test evaluation
    if best_model:
        model = apply_model_state(model, best_model)
        test_metrics = eval_epoch(model, val_loader, losser)

        if logger:
            logger.log_final_metrics(test_metrics)

    return model


def run_pretrain_workflow(
    args,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_dir: str,
    logger: Optional[Any] = None,
) -> torch.nn.Module:
    """
    Run the pre-training workflow

    Parameters:
        args: Command line arguments
        model: Model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        checkpoint_dir: Checkpoint directory
        logger: Logger

    Returns:
        model: Trained model
    """
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    # Optimizer and scheduler settings
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-3, betas=(0.9, 0.95), weight_decay=0.05
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / 20 * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_func, verbose=True
    )

    # Loss function settings
    losser = Losser(model, device=device)

    # Model training
    model = train_mae(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        losser,
        args.trial_name,
        checkpoint_dir,
        num_epochs=args.epochs,
        logger=logger,
    )

    # Feature extraction
    process_all_features(
        model,
        [train_loader, val_loader, test_loader],
        ["train", "val", "test"],
        args.mask_ratio,
        device,
        args.output_dir,
        args.trial_name,
    )

    return model 
