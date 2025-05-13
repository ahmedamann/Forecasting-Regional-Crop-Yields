import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Output directories
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                FIGURES_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
COUNTRY_LOOKUP_FILE = RAW_DATA_DIR / "country_latitude_longitude_area_lookup.csv"
YIELD_FILE = RAW_DATA_DIR / "Yield_and_Production_data.csv"
LAND_COVER_FILE = RAW_DATA_DIR / "Land_cover_percent_data.csv"

# Environmental data files
ENV_FILES = [
    'SoilMoi0_10cm_inst_data.csv',
    'SoilMoi10_40cm_inst_data.csv',
    'SoilMoi40_100cm_inst_data.csv',
    'SoilMoi100_200cm_inst_data.csv',
    'SoilTMP0_10cm_inst_data.csv',
    'SoilTMP10_40cm_inst_data.csv',
    'SoilTMP40_100cm_inst_data.csv',
    'SoilTMP100_200cm_inst_data.csv',
    'Rainf_tavg_data.csv',
    'Snowf_tavg_data.csv',
    'TVeg_tavg_data.csv',
    'TWS_inst_data.csv',
    'CanopInt_inst_data.csv',
    'ESoil_tavg_data.csv',
    'Land_cover_percent_data.csv',
]

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 200
