# ML_project_01_house-prices-predictor
Predict California housing prices using Linear Regression and Random Forest models with scikit-learn.
# House Price Prediction

Predict California house prices using Linear Regression and Random Forest.

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 0.533 | 0.746 |
| Random Forest | **0.326** | **0.503** |

**Random Forest is 39% better than Linear Regression**

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Train
python -m src.train --model_type linear --out_dir models
python -m src.train --model_type rf --out_dir models

# Evaluate
python -m src.evaluate --model_path models/linear_pipeline.joblib
python -m src.evaluate --model_path models/rf_pipeline.joblib

# Predict
python -m src.predict --model_path models/rf_pipeline.joblib --input_json "{\"MedInc\":8.3,\"HouseAge\":41,\"AveRooms\":6.98,\"AveBedrms\":1.02,\"Population\":322,\"AveOccup\":2.56,\"Latitude\":37.88,\"Longitude\":-122.23}"

Dataset
California housing from sklearn.datasets.fetch_california_housing

Project Structure
├── src/           # Training, evaluation, prediction scripts
├── models/        # Saved models (.joblib)
├── reports/       # Metrics and plots
└── requirements.txt

Dependencies
numpy, pandas, scikit-learn, matplotlib, joblib


