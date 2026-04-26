from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_eda(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Streamflow_m3s"], color="tab:blue")
    plt.title("Streamflow Over Time")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (m3/s)")
    plt.tight_layout()
    plt.savefig(output_dir / "streamflow_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Rainfall_mm", y="Streamflow_m3s", alpha=0.5)
    plt.title("Rainfall vs Streamflow")
    plt.tight_layout()
    plt.savefig(output_dir / "rainfall_vs_streamflow.png", dpi=150)
    plt.close()

    monthly = (
        df.assign(Month=df["Date"].dt.month)
        .groupby("Month", as_index=False)["Streamflow_m3s"]
        .mean()
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly, x="Month", y="Streamflow_m3s", marker="o")
    plt.title("Monthly Seasonal Pattern (Mean Streamflow)")
    plt.xticks(np.arange(1, 13))
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_seasonality.png", dpi=150)
    plt.close()

    corr_cols = [c for c in df.columns if c not in {"Date", "Season", "Data_Type"}]
    corr = df[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred_map: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label="Actual", linewidth=2, color="black")
    for model_name, preds in y_pred_map.items():
        plt.plot(dates, preds, label=model_name, alpha=0.85)
    plt.title("Actual vs Predicted Streamflow")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (m3/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(
    model,
    feature_names: Iterable[str],
    output_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    importances = pd.Series(model.feature_importances_, index=list(feature_names)).sort_values(
        ascending=False
    )
    top = importances.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, orient="h")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    return table.sort_values("RMSE", ascending=True).reset_index(drop=True)
