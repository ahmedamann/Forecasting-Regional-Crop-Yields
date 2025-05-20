import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from sklearn.metrics import mean_squared_error, r2_score
import json
import pickle

from .config import FIGURES_DIR, PREDICTIONS_DIR, LOGS_DIR, PROCESSED_DATA_DIR
from .model import create_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_scalers() -> Dict:
    """
    Load saved scalers.
    
    Returns:
        Dictionary of scalers
    """
    scalers_path = PROCESSED_DATA_DIR / "scalers.pkl"
    with open(scalers_path, 'rb') as f:
        return pickle.load(f)

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on test data.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (MSE, R2 score, true values, predictions)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Load scalers
    scalers = load_scalers()
    target_scaler = scalers['target_scaler']
    
    # Inverse transform
    try:
        all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).ravel()
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).ravel()
    except Exception as e:
        logger.error("Inverse transform failed.")
        raise e
    
    # 🔒 Filter NaNs AFTER inverse_transform
    mask = ~np.isnan(all_preds) & ~np.isnan(all_targets)
    if not np.any(mask):
        logger.error("All predictions or targets became NaN after inverse transform.")
        return float("nan"), float("nan"), np.array([]), np.array([])

    all_preds = all_preds[mask]
    all_targets = all_targets[mask]

    logger.info(f"Evaluating on {len(all_preds)} samples (after removing NaNs)")
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return mse, r2, all_targets, all_preds

def plot_predictions(
    true_values: np.ndarray,
    predictions: np.ndarray,
    save_path: Path
):
    """
    Create and save scatter plot of true values vs predictions.
    
    Args:
        true_values: Array of true values
        predictions: Array of predictions
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()],
             [true_values.min(), true_values.max()],
             'r--', lw=2)
    plt.xlabel('True Yield (kg/ha)')
    plt.ylabel('Predicted Yield (kg/ha)')
    plt.title('True Values vs Predictions')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(
    true_values: np.ndarray,
    predictions: np.ndarray,
    save_path: Path
):
    """
    Create and save histogram of prediction errors.
    
    Args:
        true_values: Array of true values
        predictions: Array of predictions
        save_path: Path to save the plot
    """
    errors = predictions - true_values
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error (kg/ha)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics: Dict[str, float], save_path: Path):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save the metrics
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_predictions(
    true_values: np.ndarray,
    predictions: np.ndarray,
    save_path: Path
):
    """
    Save true values and predictions to a CSV file.
    
    Args:
        true_values: Array of true values
        predictions: Array of predictions
        save_path: Path to save the predictions
    """
    df = pd.DataFrame({
        'true_value': true_values,
        'prediction': predictions,
        'error': predictions - true_values
    })
    df.to_csv(save_path, index=False)

def evaluate_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str = "model"
):
    """
    Evaluate model and save all results.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        model_name: Name of the model for saving files
    """
    # Evaluate model
    mse, r2, true_values, predictions = evaluate_model(model, test_loader, device)

    if len(true_values) == 0 or len(predictions) == 0:
        logger.error("No valid predictions available to plot or save. Skipping.")
        return
    
    # Create metrics dictionary
    metrics = {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2)
    }
    
    # Save metrics
    metrics_path = LOGS_DIR / f"{model_name}_metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Create and save plots
    plot_predictions(
        true_values,
        predictions,
        FIGURES_DIR / f"{model_name}_predictions.png"
    )
    plot_error_distribution(
        true_values,
        predictions,
        FIGURES_DIR / f"{model_name}_errors.png"
    )
    logger.info(f"Plots saved to {FIGURES_DIR}")
    
    # Save predictions
    save_predictions(
        true_values,
        predictions,
        PREDICTIONS_DIR / f"{model_name}_predictions.csv"
    )
    logger.info(f"Predictions saved to {PREDICTIONS_DIR}")

from .model import create_model

def main():
    """Evaluate the trained model on real test data."""
    from .dataset import load_data, create_data_loaders
    from .config import LOGS_DIR

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and create data loaders
    X, y = load_data()
    _, _, test_loader = create_data_loaders(X, y)

    # Recreate model architecture
    input_dim = X.shape[1]
    model = create_model(input_dim).to(device)

    # Load model weights
    model_path = LOGS_DIR / 'model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    # Evaluate and save results
    evaluate_and_save(model, test_loader, device)

if __name__ == "__main__":
    main()
