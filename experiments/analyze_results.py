import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import mean_squared_error, r2_score

from src.config import LOGS_DIR, FIGURES_DIR
from src.model import create_model
from src.dataset import load_data, create_data_loaders
from src.utils import get_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """
    Load model predictions from CSV files.
    
    Args:
        predictions_dir: Directory containing prediction files
    
    Returns:
        DataFrame with predictions
    """
    predictions = []
    for pred_file in predictions_dir.glob('*_predictions.csv'):
        df = pd.read_csv(pred_file)
        df['model'] = pred_file.stem.replace('_predictions', '')
        predictions.append(df)
    
    return pd.concat(predictions, ignore_index=True)

def plot_prediction_comparison(predictions_df: pd.DataFrame, save_path: Path):
    """
    Create scatter plot comparing predictions from different models.
    
    Args:
        predictions_df: DataFrame with predictions
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for model in predictions_df['model'].unique():
        model_data = predictions_df[predictions_df['model'] == model]
        plt.scatter(
            model_data['true_value'],
            model_data['prediction'],
            alpha=0.5,
            label=model
        )
    
    plt.plot(
        [predictions_df['true_value'].min(), predictions_df['true_value'].max()],
        [predictions_df['true_value'].min(), predictions_df['true_value'].max()],
        'r--',
        lw=2,
        label='Perfect Prediction'
    )
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Model Predictions Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_comparison(predictions_df: pd.DataFrame, save_path: Path):
    """
    Create box plot comparing prediction errors from different models.
    
    Args:
        predictions_df: DataFrame with predictions
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='error', data=predictions_df)
    plt.xlabel('Model')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error Distribution by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_feature_importance(
    model: torch.nn.Module,
    feature_names: List[str],
    device: torch.device,
    save_path: Path
):
    """
    Analyze feature importance using model weights.
    
    Args:
        model: Trained PyTorch model
        feature_names: List of feature names
        device: Device to use
        save_path: Path to save the plot
    """
    # Get weights from first layer
    weights = model.model[0].weight.data.abs().mean(dim=0).cpu().numpy()
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': weights
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(predictions_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate evaluation metrics for each model.
    
    Args:
        predictions_df: DataFrame with predictions
    
    Returns:
        Dictionary of metrics for each model
    """
    metrics = {}
    
    for model in predictions_df['model'].unique():
        model_data = predictions_df[predictions_df['model'] == model]
        
        mse = mean_squared_error(
            model_data['true_value'],
            model_data['prediction']
        )
        r2 = r2_score(
            model_data['true_value'],
            model_data['prediction']
        )
        
        metrics[model] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    
    return metrics

def main():
    """Run analysis of model results."""
    # Load predictions
    predictions_df = load_predictions(LOGS_DIR / 'predictions')
    
    # Create comparison plots
    plot_prediction_comparison(
        predictions_df,
        FIGURES_DIR / 'model_comparison.png'
    )
    plot_error_comparison(
        predictions_df,
        FIGURES_DIR / 'error_comparison.png'
    )
    
    # Calculate and save metrics
    metrics = calculate_metrics(predictions_df)
    with open(LOGS_DIR / 'model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Load best model for feature importance analysis
    device = get_device()
    model_path = LOGS_DIR / 'final_model' / 'final_model.pth'
    if model_path.exists():
        # Load data to get feature names
        X, y = load_data()
        feature_names = X.columns.tolist()
        
        # Create and load model
        model = create_model(input_dim=len(feature_names))
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        # Analyze feature importance
        analyze_feature_importance(
            model,
            feature_names,
            device,
            FIGURES_DIR / 'feature_importance.png'
        )
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()
