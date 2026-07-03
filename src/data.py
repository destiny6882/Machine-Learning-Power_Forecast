"""Data utilities for household electric power forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

SUM_COLS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
MEAN_COLS = ["voltage", "global_intensity"]
WEATHER_COLS = ["rr", "nbjrr1", "nbjrr5", "nbjrr10", "nbjbrou"]
TARGET_COL = "global_active_power"


def _standardize_col_name(col: str) -> str:
    return col.strip().replace(" ", "_").lower()


def read_power_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file robustly.

    It supports comma/semicolon separated files, Date+Time columns, and common
    missing value markers such as '?'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data file: {path}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        na_values=["?", "", "nan", "NaN", "NULL", "null"],
    )
    df.columns = [_standardize_col_name(c) for c in df.columns]

    # Parse datetime. The UCI raw file usually has Date and Time.
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    elif "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce", dayfirst=True)
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    else:
        raise ValueError(
            "No datetime column found. Please include 'Date' + 'Time', 'datetime', or 'date'."
        )

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # Convert numeric feature columns.
    for col in df.columns:
        if col not in {"datetime", "date", "time", "date_time"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _first_non_null(x: pd.Series) -> float:
    y = x.dropna()
    return float(y.iloc[0]) if len(y) else np.nan


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level records to daily records according to the assignment."""
    df = df.copy()
    df = df.set_index("datetime").sort_index()

    agg: Dict[str, object] = {}
    for col in SUM_COLS:
        if col in df.columns:
            agg[col] = "sum"
    for col in MEAN_COLS:
        if col in df.columns:
            agg[col] = "mean"
    for col in WEATHER_COLS:
        if col in df.columns:
            agg[col] = _first_non_null

    if TARGET_COL not in agg:
        raise ValueError(f"Target column '{TARGET_COL}' is required.")

    daily = df.resample("D").agg(agg)

    # Optional derived feature: unmetered consumption remainder in Wh.
    needed = ["global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"]
    if all(c in daily.columns for c in needed):
        daily["sub_metering_remainder"] = (
            daily["global_active_power"] * 1000.0 / 60.0
            - daily["sub_metering_1"]
            - daily["sub_metering_2"]
            - daily["sub_metering_3"]
        )

    # Fill missing values: time interpolation first, then fallback to ffill/bfill.
    daily = daily.interpolate(method="time", limit_direction="both")
    daily = daily.ffill().bfill()
    daily = daily.dropna(axis=1, how="all")

    # Ensure target is the first feature for convenient inverse transformation.
    feature_cols = [TARGET_COL] + [c for c in daily.columns if c != TARGET_COL]
    daily = daily[feature_cols]
    return daily


@dataclass
class PreparedData:
    train_daily: pd.DataFrame
    test_daily: pd.DataFrame
    feature_cols: List[str]
    scaler: StandardScaler
    target_mean: float
    target_scale: float


class WindowDataset(Dataset):
    """Sliding-window dataset.

    X shape: [input_len, feature_dim]
    y shape: [horizon]
    """

    def __init__(self, values_scaled: np.ndarray, input_len: int, horizon: int, stride: int = 1):
        if values_scaled.ndim != 2:
            raise ValueError("values_scaled must be [num_days, feature_dim].")
        self.values = values_scaled.astype(np.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.stride = stride
        max_start = len(values_scaled) - input_len - horizon
        self.starts = list(range(0, max_start + 1, stride)) if max_start >= 0 else []

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        x = self.values[s : s + self.input_len]
        y = self.values[s + self.input_len : s + self.input_len + self.horizon, 0]
        return x, y


def prepare_data(
    train_csv: str | Path,
    test_csv: str | Path,
) -> PreparedData:
    train_raw = read_power_csv(train_csv)
    test_raw = read_power_csv(test_csv)
    train_daily = aggregate_to_daily(train_raw)
    test_daily = aggregate_to_daily(test_raw)

    # Align columns. Missing test columns are filled from train medians; extra test columns are ignored.
    feature_cols = list(train_daily.columns)
    for col in feature_cols:
        if col not in test_daily.columns:
            test_daily[col] = train_daily[col].median()
    test_daily = test_daily[feature_cols]

    scaler = StandardScaler()
    scaler.fit(train_daily.values)
    target_mean = float(scaler.mean_[0])
    target_scale = float(scaler.scale_[0])
    return PreparedData(train_daily, test_daily, feature_cols, scaler, target_mean, target_scale)


def make_train_val_test_datasets(
    prepared: PreparedData,
    input_len: int,
    horizon: int,
    val_ratio: float = 0.2,
    stride: int = 1,
) -> Tuple[WindowDataset, WindowDataset, WindowDataset]:
    train_scaled = prepared.scaler.transform(prepared.train_daily.values)

    # Time-ordered split over daily records. Validation uses the tail of training period.
    split = int(len(train_scaled) * (1.0 - val_ratio))
    # Keep enough context for validation windows.
    train_part = train_scaled[:split]
    val_start = max(0, split - input_len)
    val_part = train_scaled[val_start:]

    train_ds = WindowDataset(train_part, input_len=input_len, horizon=horizon, stride=stride)
    val_ds = WindowDataset(val_part, input_len=input_len, horizon=horizon, stride=stride)

    # For test, prepend last input_len days of training data so the first test prediction
    # can use the immediate historical context.
    combined_test_daily = pd.concat([prepared.train_daily.tail(input_len), prepared.test_daily], axis=0)
    combined_test_scaled = prepared.scaler.transform(combined_test_daily.values)
    test_ds = WindowDataset(combined_test_scaled, input_len=input_len, horizon=horizon, stride=stride)

    if len(train_ds) == 0:
        raise ValueError(
            f"Training set too short for input_len={input_len}, horizon={horizon}. "
            f"Need at least {input_len + horizon} daily points."
        )
    if len(test_ds) == 0:
        raise ValueError(
            f"Test set too short for input_len={input_len}, horizon={horizon}. "
            f"Need at least {horizon} daily test points plus {input_len} history days."
        )
    return train_ds, val_ds, test_ds


def inverse_target(y_scaled: np.ndarray, target_mean: float, target_scale: float) -> np.ndarray:
    return y_scaled * target_scale + target_mean
