import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Optional
import json

from .config import RANDOM_SEED, LOGS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Get the appropriate device for training.
    
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def save_config(config: dict, path: Path):
    """
    Save configuration dictionary to a JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {path}")

def load_config(path: Path) -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {path}")
    return config

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """
    Create a directory for experiment results.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
    
    Returns:
        Path to the experiment directory
    """
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir

def main():
    """Test utility functions."""
    # Test seed setting
    set_seed(42)
    logger.info("Random seed set to 42")
    
    # Test device selection
    device = get_device()
    logger.info(f"Selected device: {device}")
    
    # Test configuration handling
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    }
    config_path = LOGS_DIR / 'test_config.json'
    save_config(config, config_path)
    loaded_config = load_config(config_path)
    logger.info(f"Loaded config: {loaded_config}")
    
    # Test time formatting
    time_str = format_time(3661)
    logger.info(f"Formatted time: {time_str}")
    
    # Test experiment directory creation
    exp_dir = create_experiment_dir(LOGS_DIR, 'test_experiment')
    logger.info(f"Created experiment directory: {exp_dir}")

if __name__ == "__main__":
    main()
