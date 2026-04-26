from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from data_loader import DatasetConfig, load_dataset
from evaluate import (
    build_comparison_table,
    ensure_dir,
    plot_actual_vs_predicted,
    plot_eda,
    plot_feature_importance,
    regression_metrics,
)
from models import train_random_forest, train_xgboost
from preprocessing import (
    clean_and_engineer_features,
    encode_categoricals,
    time_based_split,
)


@dataclass
class Artifacts:
    output_dir: Path
    model_dir: Path
    plot_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Godavari Streamflow Prediction Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="godavari_streamflow_dataset.csv",
        help="Path to input CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save plots, models and outputs",
    )
    return parser.parse_args()


def initialize_artifacts(output_dir: Path) -> Artifacts:
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(plot_dir)
    return Artifacts(output_dir=output_dir, model_dir=model_dir, plot_dir=plot_dir)


def build_prediction_dataframe(
    test_dates: pd.Series,
    y_test,
    rf_preds,
    xgb_preds,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual_Streamflow": y_test,
            "RF_Predicted": rf_preds,
            "XGB_Predicted": xgb_preds,
        }
    )
    return pred_df


def iterative_7day_forecast(best_model_info: Dict, prepared_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate next 7 days autoregressive forecast using latest available feature state.
    Assumes rainfall and temperature persistence from the most recent observation.
    """
    model_type = best_model_info["model_type"]
    model = best_model_info["model"]
    encoded_columns = best_model_info["encoded_columns"]

    recent = prepared_df.sort_values("Date").iloc[-1:].copy()
    latest_date = prepared_df["Date"].max()

    streamflow_hist = prepared_df.sort_values("Date")["Streamflow_m3s"].tail(3).tolist()
    rainfall_hist = prepared_df.sort_values("Date")["Rainfall_mm"].tail(3).tolist()

    forecasts = []
    current_state = recent.iloc[0].to_dict()

    for step in range(1, 8):
        future_date = latest_date + pd.Timedelta(days=step)
        month = future_date.month
        day = future_date.day
        if month in (6, 7, 8, 9):
            season = "Monsoon"
        elif month in (10, 11, 12, 1, 2):
            season = "Winter"
        else:
            season = "Summer"

        row = {
            "Rainfall_mm": float(current_state["Rainfall_mm"]),
            "Temperature_C": float(current_state["Temperature_C"]),
            "Rainfall_t-1": rainfall_hist[-1],
            "Rainfall_t-2": rainfall_hist[-2],
            "Rainfall_t-3": rainfall_hist[-3],
            "Streamflow_t-1": streamflow_hist[-1],
            "Streamflow_t-2": streamflow_hist[-2],
            "Streamflow_t-3": streamflow_hist[-3],
            "Month": month,
            "Day": day,
            "Data_Type": "Real",
            "Season": season,
        }
        future_df = pd.DataFrame([row])
        encoded_future = pd.get_dummies(future_df, columns=["Data_Type", "Season"], drop_first=False)
        encoded_future = encoded_future.reindex(columns=encoded_columns, fill_value=0)

        pred = float(model.predict(encoded_future)[0])

        forecasts.append({"Date": future_date.date().isoformat(), "Forecast_Streamflow": pred})

        streamflow_hist.append(pred)
        streamflow_hist = streamflow_hist[-3:]
        rainfall_hist.append(float(current_state["Rainfall_mm"]))
        rainfall_hist = rainfall_hist[-3:]

    return pd.DataFrame(forecasts)


def save_best_model(best_model_info: Dict, model_dir: Path) -> None:
    """Persist best model payload as best_model.pkl."""
    best_model_path = model_dir / "best_model.pkl"
    payload = {
        "model_type": best_model_info["model_type"],
        "encoded_columns": best_model_info["encoded_columns"],
        "feature_columns": best_model_info["feature_columns"],
        "model": best_model_info["model"],
    }
    joblib.dump(payload, best_model_path)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts = initialize_artifacts(output_dir)

    config = DatasetConfig(data_path=data_path)
    raw_df = load_dataset(config)
    df = clean_and_engineer_features(raw_df)

    plot_eda(df, artifacts.plot_dir)

    split = time_based_split(df, target_col="Streamflow_m3s")
    X_train_e, X_val_e, X_test_e = encode_categoricals(split.X_train, split.X_val, split.X_test)

    rf_model = train_random_forest(X_train_e, split.y_train)
    xgb_model = train_xgboost(X_train_e, split.y_train, X_val_e, split.y_val)

    rf_preds = rf_model.predict(X_test_e)
    xgb_preds = xgb_model.predict(X_test_e)

    rf_metrics = regression_metrics(split.y_test.to_numpy(), rf_preds)
    xgb_metrics = regression_metrics(split.y_test.to_numpy(), xgb_preds)

    comparison = build_comparison_table(
        {
            "RandomForest": rf_metrics,
            "XGBoost": xgb_metrics,
        }
    )
    comparison.to_csv(artifacts.output_dir / "model_comparison.csv", index=False)

    pred_df = build_prediction_dataframe(
        split.test_df["Date"].dt.date.astype(str),
        split.y_test.to_numpy(),
        rf_preds,
        xgb_preds,
    )
    pred_df.to_csv(artifacts.output_dir / "predictions.csv", index=False)

    plot_actual_vs_predicted(
        split.test_df["Date"],
        split.y_test.to_numpy(),
        {
            "RandomForest": rf_preds,
            "XGBoost": xgb_preds,
        },
        artifacts.plot_dir / "actual_vs_predicted.png",
    )

    plot_feature_importance(
        rf_model,
        X_train_e.columns,
        artifacts.plot_dir / "rf_feature_importance.png",
        "Random Forest Feature Importance",
    )
    plot_feature_importance(
        xgb_model,
        X_train_e.columns,
        artifacts.plot_dir / "xgb_feature_importance.png",
        "XGBoost Feature Importance",
    )

    best_model_name = comparison.iloc[0]["Model"]
    if best_model_name == "RandomForest":
        best_model_info = {
            "model_type": "random_forest",
            "model": rf_model,
            "encoded_columns": list(X_train_e.columns),
            "feature_columns": list(split.X_train.columns),
        }
    else:
        best_model_info = {
            "model_type": "xgboost",
            "model": xgb_model,
            "encoded_columns": list(X_train_e.columns),
            "feature_columns": list(split.X_train.columns),
        }

    save_best_model(best_model_info, artifacts.model_dir)

    forecast_df = iterative_7day_forecast(best_model_info, df)
    forecast_df.to_csv(artifacts.output_dir / "next_7_days_forecast.csv", index=False)

    with open(artifacts.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(data_path),
                "output_dir": str(output_dir),
                "best_model": best_model_name,
                "comparison_metrics": comparison.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Artifacts saved under: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
