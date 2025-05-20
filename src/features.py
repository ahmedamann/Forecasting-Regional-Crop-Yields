import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import logging
from pathlib import Path
from typing import Tuple, List, Dict

from src.config import PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data() -> pd.DataFrame:
    """
    Load all processed data files and merge them into a single DataFrame using a two-step process:
    1. First merge all environmental data together
    2. Then merge with yield data
    
    Returns:
        Merged DataFrame with all features
    """
    # Load processed yield data
    yield_df = pd.read_csv(PROCESSED_DATA_DIR / "yield_processed.csv")
    logger.info(f"Loaded yield data with shape: {yield_df.shape}")
    
    # Get all environmental data files (exclude yield_processed.csv and features_processed.csv)
    env_files = [
        f for f in PROCESSED_DATA_DIR.glob("*_processed.csv")
        if f.name not in ["yield_processed.csv", "features_processed.csv"]
    ]
    
    # Step 1: Merge all environmental data
    env_dfs = []
    for file in env_files:
        try:
            df = pd.read_csv(file)
            logger.info(f"Loading {file.name} with shape: {df.shape}")
            
            # Verify required columns exist
            if 'country' not in df.columns or 'year' not in df.columns:
                logger.error(f"Error: {file.name} is missing required columns")
                raise ValueError(f"{file.name} is missing required columns")
            
            env_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    # Merge all environmental data
    merged_env = env_dfs[0]
    for df in env_dfs[1:]:
        merged_env = pd.merge(merged_env, df, on=['country', 'year'], how='outer')
    
    # Drop rows with NaN values in environmental data
    merged_env = merged_env.dropna()
    logger.info(f"Merged environmental data shape: {merged_env.shape}")
    
    # Step 2: Merge with yield data
    merged_data = pd.merge(yield_df, merged_env, on=['country', 'year'], how='right')
    
    # Drop rows where yield value is absent
    merged_data = merged_data.dropna(subset=['target_yield', 'yield']).reset_index(drop=True)
    logger.info(f"Final merged data shape: {merged_data.shape}")
    
    return merged_data

def select_features_by_mi(X: pd.DataFrame, y: pd.Series, n_features: int = 35) -> List[str]:
    """
    Select features using mutual information regression.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_features: Number of features to select
    
    Returns:
        List of selected feature names
    """
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    selected_features = mi_scores.nlargest(n_features).index.tolist()
    return selected_features

def apply_pca(X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA to reduce dimensionality while preserving specified variance.
    
    Args:
        X: Feature DataFrame
        n_components: Variance ratio to preserve
    
    Returns:
        Tuple of (transformed DataFrame, fitted PCA object)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    return X_pca, pca

def prepare_features(df: pd.DataFrame, target_col: str = 'target_yield') -> Tuple[pd.DataFrame, pd.Series, List[str], Dict]:
    """
    Prepare features for model training.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
    
    Returns:
        Tuple of (feature DataFrame, target Series, selected feature names, scalers dictionary)
    """
    # Separate features and target
    X = df.drop(['country', 'year', target_col], axis=1)
    y = df[target_col]
    
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Scale features
    X_scaled = pd.DataFrame(
        feature_scaler.fit_transform(X),
        columns=X.columns
    )
    
    # Scale target
    y_scaled = pd.Series(
        target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel(),
        index=y.index
    )
    
    # Select features using MI
    selected_features = select_features_by_mi(X_scaled, y_scaled)
    X_selected = X_scaled[selected_features]
    
    # Apply PCA
    X_pca, pca = apply_pca(X_selected)
    
    # Store scalers for later use
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'pca': pca
    }
    
    return X_pca, y_scaled, selected_features, scalers

def main():
    """Main feature engineering pipeline."""
    # Load processed data
    df = load_processed_data()
    
    # Prepare features
    X, y, selected_features, scalers = prepare_features(df, target_col='target_yield')
    
    # Save processed features with correct target column name
    features_df = pd.concat([X, y.rename('target_yield')], axis=1)
    output_path = PROCESSED_DATA_DIR / "features_processed.csv"
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed features to {output_path}")
    
    # Save selected feature names
    features_path = PROCESSED_DATA_DIR / "selected_features.txt"
    with open(features_path, 'w') as f:
        f.write('\n'.join(selected_features))
    logger.info(f"Saved selected feature names to {features_path}")
    
    # Save scalers
    scalers_path = PROCESSED_DATA_DIR / "scalers.pkl"
    import pickle
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Saved scalers to {scalers_path}")

if __name__ == "__main__":
    main()
