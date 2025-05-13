import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from .config import PROCESSED_DATA_DIR, BATCH_SIZE

class YieldDataset(Dataset):
    """
    PyTorch Dataset for crop yield prediction.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed features and targets.
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    data_path = PROCESSED_DATA_DIR / "features_processed.csv"
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('target_yield', axis=1)
    y = df['target_yield']
    
    return X, y

def create_data_loaders(
    X: pd.DataFrame,
    y: pd.Series,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        batch_size: Batch size for data loaders
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate split indices
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = YieldDataset(
        X.iloc[train_indices].values,
        y.iloc[train_indices].values
    )
    val_dataset = YieldDataset(
        X.iloc[val_indices].values,
        y.iloc[val_indices].values
    )
    test_dataset = YieldDataset(
        X.iloc[test_indices].values,
        y.iloc[test_indices].values
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def main():
    """Test data loading and dataset creation."""
    # Load data
    X, y = load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    
    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test batch loading
    for features, targets in train_loader:
        print(f"Batch shape: {features.shape}")
        print(f"Target shape: {targets.shape}")
        break

if __name__ == "__main__":
    main()
