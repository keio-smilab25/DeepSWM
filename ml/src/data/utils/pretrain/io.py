import os
import numpy as np
import h5py
import torch
import gc


def save_features(output_dir, features, labels, timestamps, dataset_type, trial_name):
    """
    Save extracted features in H5 format

    Parameters:
        output_dir: Output directory
        features: Features
        labels: Labels
        timestamps: Timestamps
        dataset_type: Dataset type (train/val/test)
        trial_name: Trial name
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_file = os.path.join(output_dir, f"{dataset_type}_features_{trial_name}.h5")

    with h5py.File(feature_file, "w") as f:
        # Create datasets in chunks
        f.create_dataset("features", data=features, chunks=True)
        f.create_dataset("labels", data=labels, chunks=True)
        f.create_dataset("timestamps", data=timestamps.astype("S"), chunks=True)

    print(f"Saved {dataset_type} features to {feature_file}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Timestamps shape: {timestamps.shape}")

    gc.collect()


def save_model(model, checkpoint_dir, trial_name, is_best=False):
    """
    Save model state

    Parameters:
        model: Model to save
        checkpoint_dir: Checkpoint directory
        trial_name: Trial name
        is_best: Whether to save as the best model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save regular checkpoint
    model_path = os.path.join(checkpoint_dir, f"mae_model_{trial_name}.pth")
    torch.save(model.state_dict(), model_path)

    # Save as best model
    if is_best:
        best_model_path = os.path.join(
            checkpoint_dir, f"best_mae_model_{trial_name}.pth"
        )
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        print(f"Model saved to {model_path}")


def load_model(model, checkpoint_dir, trial_name, use_best=True):
    """
    Load saved model state

    Parameters:
        model: Target model
        checkpoint_dir: Checkpoint directory
        trial_name: Trial name
        use_best: Whether to load the best model

    Returns:
        model: Loaded model
    """
    if use_best:
        model_path = os.path.join(checkpoint_dir, f"best_mae_model_{trial_name}.pth")
    else:
        model_path = os.path.join(checkpoint_dir, f"mae_model_{trial_name}.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found")

    return model


def setup_checkpoint_dir(trial_name):
    """
    Set up checkpoint directory

    Parameters:
        trial_name: Trial name

    Returns:
        checkpoint_dir: Path to checkpoint directory
    """
    checkpoint_dir = os.path.join("checkpoints", "pretrain", trial_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def apply_model_state(model, state_dict):
    """
    Apply state dictionary to model

    Parameters:
        model: Target model
        state_dict: State dictionary to apply

    Returns:
        model: Model with applied state
    """
    model.load_state_dict(state_dict)
    return model
