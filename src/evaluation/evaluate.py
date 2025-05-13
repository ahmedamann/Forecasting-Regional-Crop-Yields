import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from src.data.data_loader import load_and_preprocess_data
from src.models.model import CropYieldModel

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.savefig('prediction_plot.png')
    plt.close()
    
    # Plot error distribution
    errors = all_predictions - all_targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.savefig('error_distribution.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = 'data/processed/crop_data.csv'  # Update this path to your data file
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(data_path)
    
    # Load model
    input_dim = next(iter(train_loader))[0].shape[1]
    model = CropYieldModel(input_dim).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('models/trained_model.pth'))
    
    # Evaluate model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main() 