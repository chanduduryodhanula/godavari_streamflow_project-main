from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    data_path: Path
    date_column: str = "Date"
    target_column: str = "Streamflow_m3s"


REQUIRED_COLUMNS: List[str] = [
    "Date",
    "Rainfall_mm",
    "Temperature_C",
    "Streamflow_m3s",
    "Rainfall_t-1",
    "Rainfall_t-2",
    "Rainfall_t-3",
    "Streamflow_t-1",
    "Streamflow_t-2",
    "Streamflow_t-3",
    "Data_Type",
]


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load dataset, enforce schema, and return date-sorted DataFrame."""
    if not config.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {config.data_path}")

    df = pd.read_csv(config.data_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column], errors="coerce")
    df = df.dropna(subset=[config.date_column]).sort_values(config.date_column)
    df = df.reset_index(drop=True)
    return df
