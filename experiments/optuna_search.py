import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import json
import logging
from typing import Dict, List, Tuple, Any
import time

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
        logging.FileHandler(LOGS_DIR / 'optuna_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: torch.device
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input dimension
        device: Device to train on
    
    Returns:
        Validation loss
    """
    # Define hyperparameter search space
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f'n_units_l{i}', 32, 256))
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    
    # Create model
    model = create_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device
    )
    
    # Return best validation loss
    return min(history['val_loss'])

def run_optuna_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: torch.device,
    experiment_dir: Path,
    n_trials: int = 10
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        input_dim: Input dimension
        device: Device to train on
        experiment_dir: Directory to save results
        n_trials: Number of optimization trials
    
    Returns:
        Dictionary of best parameters and their performance
    """
    # Create study
    study = optuna.create_study(direction='minimize')
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, input_dim, device),
        n_trials=n_trials
    )
    
    # Get best parameters
    best_params = study.best_params
    best_val_loss = study.best_value
    
    # Train best model
    n_layers = best_params['n_layers']
    hidden_dims = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
    
    best_model = create_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=best_params['dropout_rate']
    )
    
    best_model, _ = train_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=best_params['learning_rate'],
        device=device
    )
    
    # Save best model
    save_model(best_model, experiment_dir / 'best_model.pth')
    
    # Evaluate best model
    evaluate_and_save(
        model=best_model,
        test_loader=test_loader,
        device=device,
        model_name='best_model'
    )
    
    # Save study results
    with open(experiment_dir / 'optuna_results.json', 'w') as f:
        json.dump({
            'best_parameters': best_params,
            'best_val_loss': float(best_val_loss),
            'n_trials': n_trials,
            'study_history': [
                {
                    'number': t.number,
                    'value': float(t.value),
                    'params': t.params
                }
                for t in study.trials
            ]
        }, f, indent=4)
    
    return {
        'best_parameters': best_params,
        'best_val_loss': best_val_loss
    }

def main():
    """Run Optuna hyperparameter optimization."""
    # Set random seed
    set_seed()
    
    # Get device
    device = get_device()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(LOGS_DIR, 'optuna_search')
    
    # Load data
    X, y = load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    
    # Run optimization
    best_results = run_optuna_search(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=X.shape[1],
        device=device,
        experiment_dir=experiment_dir
    )
    
    logger.info("Best parameters found:")
    logger.info(json.dumps(best_results, indent=2))

if __name__ == "__main__":
    main() 