import torch
import torch.nn as nn
from typing import List, Optional

class YieldMLP(nn.Module):
    """
    Multi-Layer Perceptron for crop yield prediction.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.model(x)

def create_model(
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.1
) -> YieldMLP:
    """
    Create a YieldMLP model with default or specified architecture.
    
    Args:
        input_dim: Number of input features
        hidden_dims: Optional list of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Initialized YieldMLP model
    """
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    
    return YieldMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
