# Regional Crop Yield Forecasting

This project implements a machine learning system for forecasting regional crop yields using environmental and agricultural data.

## Project Structure

```
.
│
├── 📁 data/
│   ├── raw/                # Original data files
│   └── external/           # Final cleaned files
│
├── 📁 src/
│   ├── config.py           # Configurations (paths, constants, etc.)
│   ├── preprocess.py       # Data loading and cleaning
│   ├── features.py         # Feature engineering
│   ├── dataset.py          # Dataset creation and PyTorch loaders
│   ├── model.py            # MLP model architecture
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation metrics
│   └── utils.py            # Common utilities
│
├── 📁 experiments/
│   ├── optuna_search.py    # Hyperparameter tuning
│   ├── run_final_model.py  # Final model training
│   └── analyze_results.py  # Results analysis
│
├── 📁 notebooks/
│   └── Code.ipynb          # Original Notebook
│
└── 📁 outputs/
    ├── figures/            # Saved plots
    ├── predictions/        # Model predictions
    └── logs/               # Training logs
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your raw data files in the `data/raw/` directory
2. Run preprocessing:
```bash
python src/preprocess.py
```
3. Train the model:
```bash
python experiments/run_final_model.py
```


## Model

The project implements a Multi-Layer Perceptron (MLP) for crop yield prediction, using environmental and agricultural features. 