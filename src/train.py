import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
from tqdm import tqdm

from .model import create_model
from .config import (
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    LOGS_DIR, RANDOM_SEED
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for features, targets in tqdm(train_loader, desc="Training"):
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), targets)

        if torch.isnan(outputs).any():
            logger.warning("NaNs detected in model outputs during training.")
            logger.info(f"Sample outputs: {outputs[:5]}")
        
        if torch.isinf(outputs).any():
            logger.warning("Infs detected in model outputs during training.")
            logger.info(f"Sample outputs: {outputs[:5]}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc="Validation"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
    
    Returns:
        Tuple of (best model, training history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def save_model(model: nn.Module, path: Path):
    """
    Save the model state.
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def main():
    """Train model using real data."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    from .dataset import load_data, create_data_loaders
    
    # Load data
    X, y = load_data()
    input_dim = X.shape[1]

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"X NaNs: {X.isna().sum().sum()}, y NaNs: {y.isna().sum()}")

    assert not X.isna().any().any(), "X contains NaNs"
    assert not y.isna().any(), "y contains NaNs"

    # Create model
    model = create_model(input_dim)

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(X, y)

    # Train model
    model, history = train_model(model, train_loader, val_loader)

    # Save model
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, LOGS_DIR / 'model.pth')

if __name__ == "__main__":
    main()
