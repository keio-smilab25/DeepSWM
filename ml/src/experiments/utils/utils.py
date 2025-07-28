import numpy as np
import torch
import random
import logging
from torchinfo import summary
from typing import Tuple, Any


def fix_seed(seed: int) -> None:
    """Fix random seed for reproducibility
    
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_device: int = 0) -> torch.device:
    """Get computation device
    
    Args:
        cuda_device (int): CUDA device ID
        
    Returns:
        torch.device: Device for computation
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_complexity(model: torch.nn.Module, device: torch.device, logger: logging.Logger) -> None:
    """Analyze model complexity and log information
    
    Args:
        model: PyTorch model
        device: Device where model is located
        logger: Logger instance
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Parameters:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    logger.info(f"  Model size: {size_all_mb:.3f} MB")


def get_model_summary(model: torch.nn.Module, logger: logging.Logger, input_size: Tuple = (1, 90, 4, 512, 512)) -> None:
    """Get and log model summary
    
    Args:
        model: PyTorch model
        logger: Logger instance
        input_size: Input tensor size for summary
    """
    try:
        model_summary = summary(model, input_size=input_size, verbose=0)
        logger.info(f"Model Summary:\n{model_summary}")
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")


def save_args_to_file(args, filepath: str):
    """Save arguments to a file
    
    Args:
        args: Arguments object
        filepath (str): Path to save file
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert args to dict, handling Namespace objects
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    # Convert non-serializable objects to strings
    serializable_dict = {}
    for key, value in args_dict.items():
        if isinstance(value, (str, int, float, bool, list)):
            serializable_dict[key] = value
        else:
            serializable_dict[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def is_better_score(current_value: float, best_value: float, metric_name: str) -> bool:
    """
    Determine if current value is better than best score
    GMGS: higher is better, loss: lower is better
    """
    if "GMGS" in metric_name:
        return current_value > best_value
    else:  # loss
        return current_value < best_value 
