import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def save_predictions(
    best_valid_predictions=None,
    best_valid_observations=None,
    best_valid_score=None,
    best_valid_loss=None,
    test_predictions=None,
    test_observations=None,
    test_score=None,
    test_loss=None,
    trial_name=None,
    fold=None,
    stage=None
):
    """
    Save predictions and observations for analysis
    
    Args:
        best_valid_predictions: Best validation predictions
        best_valid_observations: Best validation observations  
        best_valid_score: Best validation score
        best_valid_loss: Best validation loss
        test_predictions: Test predictions
        test_observations: Test observations
        test_score: Test score
        test_loss: Test loss
        trial_name: Trial name
        fold: Fold number
        stage: Training stage
        
    Returns:
        results_dir: Directory where results are saved
    """
    # Create results directory
    results_dir = os.path.join("results", "main", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best validation predictions
    if best_valid_predictions is not None:
        valid_results = {
            'predictions': best_valid_predictions,
            'observations': best_valid_observations,
            'score': best_valid_score,
            'loss': best_valid_loss,
            'fold': fold,
            'stage': stage,
            'trial_name': trial_name
        }
        
        valid_path = os.path.join(results_dir, f"valid_predictions_stage{stage}_fold{fold}.npz")
        np.savez(valid_path, **valid_results)
        logger.info(f"Saved validation predictions to {valid_path}")
    
    # Save test predictions
    if test_predictions is not None:
        test_results = {
            'predictions': test_predictions,
            'observations': test_observations,
            'score': test_score,
            'loss': test_loss,
            'fold': fold,
            'stage': stage,
            'trial_name': trial_name
        }
        
        test_path = os.path.join(results_dir, f"test_predictions_stage{stage}_fold{fold}.npz")
        np.savez(test_path, **test_results)
        logger.info(f"Saved test predictions to {test_path}")
        
    return results_dir


def save_checkpoint(model, optimizer, best_valid_gmgs, stage, args):
    checkpoint_dir = os.path.join("checkpoints", "main")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"{args.trial_name}_stage{stage}_best.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_gmgs": best_valid_gmgs,
            "stage": stage,
            "config": vars(args),
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device, scheduler=None):
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except KeyError as e:
        logger.warning(f"Failed to load optimizer state: {e}")
        logger.info("Initializing optimizer with default state")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    if k == "step":
                        state[k] = torch.tensor(0, device=device)

    # Restore scheduler state (if exists)
    if scheduler and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except:
            logger.warning("Failed to load scheduler state")

    best_valid_gmgs = checkpoint.get("best_valid_gmgs", float("-inf"))
    start_epoch = checkpoint.get("epoch", 0) + 1 if "epoch" in checkpoint else 0

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return checkpoint.get("config", {}), best_valid_gmgs, start_epoch


def save_test_results(predictions, observations, test_indices, trial_name, stage):
    """
    Save test results to file
    
    Parameters:
        predictions: Model predictions
        observations: True observations
        test_indices: Test data indices
        trial_name: Trial name
        stage: Training stage
        
    Returns:
        results_path: Path to saved results file
    """
    results_dir = os.path.join("results", "main")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir, f"{trial_name}_stage{stage}_test_results_{timestamp}.csv"
    )
    
    # Convert to numpy arrays if necessary
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    if hasattr(observations, 'numpy'):
        observations = observations.numpy()
    if hasattr(test_indices, 'numpy'):
        test_indices = test_indices.numpy()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'index': test_indices,
        'true_label': observations,
        'predicted_label': np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions,
    })
    
    # Add prediction probabilities if available
    if predictions.ndim > 1:
        for i in range(predictions.shape[1]):
            results_df[f'prob_class_{i}'] = predictions[:, i]
    
    # Save to CSV
    results_df.to_csv(results_path, index=False)
    logger.info(f"Test results saved to {results_path}")
    
    return results_path


def save_samples(samples, category, trial_name):
    """Function to save samples for qualitative evaluation

    Args:
        samples (list): List of samples to save [(X, true_class, pred_class, file_path), ...]
        category (str): Sample category ("TP" or "TN")
        trial_name (str): Trial name
    """
    base_dir = f"results/qualitative_results/{trial_name}/{category}"
    os.makedirs(base_dir, exist_ok=True)

    for idx, (x, true_class, pred_class, file_path) in enumerate(samples):
        sample_dir = os.path.join(
            base_dir, f"true_{true_class}_pred_{pred_class}_sample{idx}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        # Save images of 12 channels at the last time step
        x_last = x[-1].numpy()  # (12, 256, 256)
        for ch in range(12):
            plt.figure(figsize=(8, 8))
            plt.imshow(x_last[ch], cmap="viridis")
            plt.colorbar()
            plt.title(f"Channel {ch}")
            plt.savefig(os.path.join(sample_dir, f"channel_{ch}.png"))
            plt.close()

        # Save file path information
        with open(os.path.join(sample_dir, "info.txt"), "w") as f:
            f.write(f"File: {file_path}\n")
            f.write(f"True class: {true_class}\n")
            f.write(f"Predicted class: {pred_class}\n")
