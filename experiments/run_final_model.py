import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import logging
from typing import Dict, Any

from src.model import create_model
from src.dataset import load_data, create_data_loaders
from src.train import train_model, save_model
from src.evaluate import evaluate_and_save
from src.utils import set_seed, get_device, create_experiment_dir
from src.config import LOGS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'final_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_best_params(optuna_dir: Path) -> Dict[str, Any]:
    """
    Load best parameters from Optuna results.
    
    Args:
        optuna_dir: Directory containing Optuna results
    
    Returns:
        Dictionary of best parameters
    """
    results_path = optuna_dir / 'optuna_results.json'
    if not results_path.exists():
        raise ValueError("No Optuna results found")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_parameters']
    logger.info("Best parameters from Optuna:")
    logger.info(json.dumps(best_params, indent=2))
    
    return best_params

def main():
    """Train and evaluate the final model with best parameters."""
    # Set random seed
    set_seed()
    
    # Get device
    device = get_device()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(LOGS_DIR, 'final_model')
    
    # Load best parameters from Optuna
    optuna_dir = LOGS_DIR / 'optuna_search'
    best_params = load_best_params(optuna_dir)
    
    # Load data
    X, y = load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    
    # Create model with best parameters
    hidden_dims = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    model = create_model(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=best_params['dropout_rate']
    )
    
    # Train model
    logger.info("Training final model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=best_params['learning_rate'],
        device=device
    )
    
    # Save model
    save_model(model, experiment_dir / 'final_model.pth')
    
    # Save training history
    with open(experiment_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Evaluate model
    logger.info("Evaluating final model...")
    evaluate_and_save(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name='final_model'
    )
    
    logger.info("Final model training and evaluation completed")

if __name__ == "__main__":
    main()
