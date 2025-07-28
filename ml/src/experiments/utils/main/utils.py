import numpy as np

from typing import Any, Tuple


def process_predictions_and_observations(predictions: np.ndarray, observations: np.ndarray, stat: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Process predictions and observations for analysis
    
    Args:
        predictions: Model predictions
        observations: True observations
        stat: Statistics object
        
    Returns:
        Tuple of processed predictions and observations
    """
    # Convert predictions to class labels if they are probabilities
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions.flatten()
    
    # Ensure observations are 1D
    if observations.ndim > 1:
        obs_labels = np.argmax(observations, axis=1) if observations.shape[1] > 1 else observations.flatten()
    else:
        obs_labels = observations
    
    return pred_labels, obs_labels


def get_model_class(name: str):
    """Get model class by name."""
    import src.models.main.model
    return src.models.main.model.__dict__[name]
