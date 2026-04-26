# Godavari Streamflow Prediction Project

Production-style end-to-end machine learning pipeline for time-series streamflow forecasting in the Godavari basin.

## Project Structure

- `data_loader.py` - schema validation, robust dataset loading
- `preprocessing.py` - cleaning, feature engineering, split logic, encoding/scaling
- `models.py` - Random Forest, XGBoost, and LSTM model training
- `evaluate.py` - metrics, EDA plots, prediction plots, feature importance
- `train.py` - orchestration script for full workflow and artifact generation
- `requirements.txt` - Python dependencies

## Expected Input

CSV with:
- `Date`
- `Rainfall_mm`
- `Temperature_C`
- `Streamflow_m3s` (target)
- `Rainfall_t-1`, `Rainfall_t-2`, `Rainfall_t-3`
- `Streamflow_t-1`, `Streamflow_t-2`, `Streamflow_t-3`
- `Data_Type` (`Real` / `Synthetic`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python train.py --data-path "../godavari_streamflow_dataset.csv" --output-dir "./artifacts"
```

## Outputs

Generated in `artifacts/`:
- `predictions.csv` (test-period predictions for all models)
- `model_comparison.csv` (MAE, RMSE, R2 comparison)
- `next_7_days_forecast.csv` (bonus forecast)
- `run_metadata.json`
- `models/best_model.pkl`
- `models/best_lstm.keras` (if LSTM wins)
- EDA + model plots in `plots/`

## Notes

- Time-aware split:
  - Train: `2018-2022`
  - Validation: `2023`
  - Test: `2024-2026`
- LSTM uses sliding windows and dropout regularization.
- Best model is chosen by minimum test RMSE.
