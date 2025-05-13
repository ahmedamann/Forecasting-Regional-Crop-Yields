import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import logging
from typing import Dict, Tuple, List

from .config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, COUNTRY_LOOKUP_FILE,
    YIELD_FILE, LAND_COVER_FILE, ENV_FILES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_monthly_into_year_columns(var_name: str, df: pd.DataFrame, agg_functions: List[str]) -> pd.DataFrame:
    """
    Convert monthly columns into yearly summaries using specified aggregation functions.
    
    Args:
        var_name: Name of the variable being processed
        df: DataFrame with monthly columns
        agg_functions: List of aggregation functions to apply ('mean', 'max', 'min', 'std')
    
    Returns:
        DataFrame with yearly aggregated columns
    """
    df = df.copy()
    month_cols = [col for col in df.columns if 'month' in col]

    funcs = {
        'mean': lambda x: x.mean(axis=1),
        'max': lambda x: x.max(axis=1),
        'min': lambda x: x.min(axis=1),
        'std': lambda x: x.std(axis=1),
    }

    if month_cols:
        for fun in agg_functions:
            df[f'{var_name}_{fun}'] = funcs[fun](df[month_cols])
        df = df.drop(month_cols, axis=1)
    else:
        logger.warning(f"No Month Columns in {var_name}")
    return df

def handle_missing_values_by_interpolation(file_path: Path) -> Tuple[str, pd.DataFrame]:
    """
    Handle missing values in the data using linear interpolation.
    
    Args:
        file_path: Path to the data file
    
    Returns:
        Tuple of (variable name, processed DataFrame)
    """
    var_name = "_".join(file_path.stem.split('_')[:2])
    df = pd.read_csv(file_path)
    
    if df.isnull().sum().values.any():
        logger.info(f"Missing Values Exist for {var_name}. Handling it using Linear Interpolation...")
        month_cols = [col for col in df.columns if 'month' in col]
        df[month_cols] = df[month_cols].interpolate(method='linear', axis=1)
    else:
        logger.info(f"No Missing Values for {var_name}")
    
    return var_name, df

def map_to_country(df: pd.DataFrame, country_data: pd.DataFrame) -> pd.DataFrame:
    """
    Map latitude and longitude coordinates to the nearest country centroid.
    
    Args:
        df: DataFrame with latitude and longitude columns
        country_data: DataFrame with country centroid information
    
    Returns:
        DataFrame with country mapping and coordinates dropped
    """
    valid_country_data = country_data.dropna(subset=['centroid latitude', 'centroid longitude']).copy()
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
        valid_country_data[['centroid latitude', 'centroid longitude']].values
    )
    distances, indices = nbrs.kneighbors(df[['latitude', 'longitude']].values)
    df['country'] = valid_country_data.iloc[indices.flatten()]['country'].values

    df = df.drop(['longitude', 'latitude'], axis=1)
    return df

def process_environmental_data(file_path: Path, country_data: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Process environmental data files with interpolation, aggregation, and country mapping.
    
    Args:
        file_path: Path to the environmental data file
        country_data: DataFrame with country information
    
    Returns:
        Tuple of (variable name, processed DataFrame)
    """
    logger.info(f'Processing {file_path}...')
    var_name, df = handle_missing_values_by_interpolation(file_path)
    
    # Monthly to year summaries
    df = aggregate_monthly_into_year_columns(var_name, df, ['mean', 'max', 'min', 'std'])

    # Map to nearest country centroid
    df_stats = map_to_country(df, country_data)
    
    # Group by country and year
    country_agg = df_stats.groupby(['country', 'year']).mean().reset_index()

    return var_name, country_agg

def handle_land_cover_data(file_path: Path, country_data: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Process land cover data with country mapping and column renaming.
    
    Args:
        file_path: Path to the land cover data file
        country_data: DataFrame with country information
    
    Returns:
        Tuple of (variable name, processed DataFrame)
    """
    var_name = "_".join(file_path.stem.split('_')[:2])
    df = pd.read_csv(file_path)
    
    # Map to country
    df = map_to_country(df, country_data)
    
    # Group by country and year
    df = df.groupby(['country', 'year']).mean().reset_index()

    # Fix column names
    df.columns = ['country', 'year'] + [f'Land_cover_percent_class_{i}_mean' for i in range(1, 18)]

    return var_name, df

def process_yield_data():
    """
    Process the yield data to match the notebook logic and save to processed directory.
    """
    yield_data = pd.read_csv(YIELD_FILE)
    # Filter for yield only values
    valid_yield_data = yield_data[yield_data['Element'] == 'Yield']
    # Drop unnecessary columns
    valid_yield_data = valid_yield_data.drop(['Domain', 'Item Code (CPC)', 'Flag', 'Element'], axis=1)
    # Rename columns
    df = valid_yield_data.rename(columns={"Value": "yield", "Country": "country", "Year": "year"})
    # Sum up all the yield for all items
    df = df.groupby(['country', 'year'])['yield'].sum().reset_index()
    # Get target yield column (next year)
    df["target_yield"] = df.groupby(["country"])["yield"].shift(-1)
    # Drop all the missing rows in target_yield column
    df = df.dropna(subset=["target_yield"])
    # Keep values as float
    df['target_yield'] = df['target_yield'].astype(float)
    # Save processed yield data
    output_path = PROCESSED_DATA_DIR / "yield_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed yield data to {output_path}")
    return df

def main():
    """Main preprocessing pipeline."""
    # Process yield data first
    process_yield_data()
    # Read country lookup data
    country_data = pd.read_csv(COUNTRY_LOOKUP_FILE)
    
    # Process environmental data
    processed_env_data = {}
    for file in ENV_FILES[:-1]:  # Exclude land cover file
        file_path = RAW_DATA_DIR / file
        var_name, data = process_environmental_data(file_path, country_data)
        processed_env_data[var_name] = data
    
    # Process land cover data
    lc_var_name, lc_data = handle_land_cover_data(LAND_COVER_FILE, country_data)
    processed_env_data[lc_var_name] = lc_data
    
    # Save processed data
    for var_name, data in processed_env_data.items():
        output_path = PROCESSED_DATA_DIR / f"{var_name}_processed.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
