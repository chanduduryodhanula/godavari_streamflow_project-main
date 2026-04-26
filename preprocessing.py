from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


NUMERIC_COLUMNS: List[str] = [
    "Rainfall_mm",
    "Temperature_C",
    "Streamflow_m3s",
    "Rainfall_t-1",
    "Rainfall_t-2",
    "Rainfall_t-3",
    "Streamflow_t-1",
    "Streamflow_t-2",
    "Streamflow_t-3",
]


def month_to_season(month: int) -> str:
    """Map month to basin-relevant meteorological season."""
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (10, 11, 12, 1, 2):
        return "Winter"
    return "Summer"


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Handle inconsistencies and add engineered time features."""
    data = df.copy()
    data["Data_Type"] = data["Data_Type"].astype(str).str.strip().str.title()
    data["Data_Type"] = data["Data_Type"].where(
        data["Data_Type"].isin(["Real", "Synthetic"]), "Unknown"
    )

    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].interpolate(limit_direction="both")
    data = data.dropna(subset=["Streamflow_m3s"]).reset_index(drop=True)

    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Season"] = data["Month"].apply(month_to_season)
    return data


def get_model_feature_columns() -> List[str]:
    """Feature set consumed by ML models (excluding target/date)."""
    return [
        "Rainfall_mm",
        "Temperature_C",
        "Rainfall_t-1",
        "Rainfall_t-2",
        "Rainfall_t-3",
        "Streamflow_t-1",
        "Streamflow_t-2",
        "Streamflow_t-3",
        "Month",
        "Day",
        "Data_Type",
        "Season",
    ]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def time_based_split(df: pd.DataFrame, target_col: str = "Streamflow_m3s") -> SplitData:
    """Split data by year ranges without leakage."""
    train_mask = (df["Date"] >= "2018-01-01") & (df["Date"] <= "2022-12-31")
    val_mask = (df["Date"] >= "2023-01-01") & (df["Date"] <= "2023-12-31")
    test_mask = (df["Date"] >= "2024-01-01") & (df["Date"] <= "2026-12-31")

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One or more time splits are empty. Confirm dataset covers 2018-2026."
        )

    features = get_model_feature_columns()
    X_train, y_train = train_df[features], train_df[target_col]
    X_val, y_val = val_df[features], val_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def encode_categoricals(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical fields with aligned columns."""
    X_train_e = pd.get_dummies(X_train, columns=["Data_Type", "Season"], drop_first=False)
    X_val_e = pd.get_dummies(X_val, columns=["Data_Type", "Season"], drop_first=False)
    X_test_e = pd.get_dummies(X_test, columns=["Data_Type", "Season"], drop_first=False)

    X_val_e = X_val_e.reindex(columns=X_train_e.columns, fill_value=0)
    X_test_e = X_test_e.reindex(columns=X_train_e.columns, fill_value=0)
    return X_train_e, X_val_e, X_test_e
