"""Prepare train/test CSV files from the original UCI household power file.

This version supports optional Météo-France monthly weather data.

Examples:
    # Power data only
    python -m src.prepare_split \
        --raw-csv data/household_power_consumption.txt \
        --test-days 365

    # Power + weather data
    python -m src.prepare_split \
        --raw-csv data/household_power_consumption.txt \
        --weather-csv data/MENSQ_92_previous-1950-2024.csv \
        --test-days 365

The script first aggregates the minute-level power data to daily records. If a
monthly weather CSV is provided, it selects the course-required variables and
copies each monthly weather value to all days in that month. Finally, it uses a
chronological split: the last N days are test.csv and the earlier days are
train.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import aggregate_to_daily, read_power_csv

WEATHER_COLS_UPPER = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
WEATHER_STATION_CANDIDATES = ["NUM_POSTE", "POSTE", "ID_POSTE", "IDPOSTE", "STAID", "STATION"]
WEATHER_MONTH_CANDIDATES = [
    "AAAAMM",
    "YYYYMM",
    "DATE",
    "DATE_MENS",
    "DATE_MENSUELLE",
    "MOIS",
    "MONTH",
]


def _read_weather_csv(weather_path: Path) -> pd.DataFrame:
    """Read Météo-France monthly climate CSV robustly.

    The file is usually semicolon-separated and may use decimal commas.
    """
    df = pd.read_csv(
        weather_path,
        sep=None,
        engine="python",
        decimal=",",
        na_values=["", "?", "mq", "MQ", "nan", "NaN"],
    )
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _find_month_column(df: pd.DataFrame) -> str | None:
    for c in WEATHER_MONTH_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _parse_weather_month(df: pd.DataFrame) -> pd.Series:
    month_col = _find_month_column(df)
    if month_col is not None:
        s = df[month_col].astype(str).str.strip()
        numeric_like = s.str.replace(r"[^0-9]", "", regex=True)
        parsed = pd.to_datetime(numeric_like.str[:6], format="%Y%m", errors="coerce")
        fallback = pd.to_datetime(s, errors="coerce", dayfirst=True)
        parsed = parsed.fillna(fallback)
        return parsed.dt.to_period("M").dt.to_timestamp()

    year_candidates = ["ANNEE", "YEAR", "YYYY"]
    month_candidates = ["MOIS", "MONTH", "MM"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    mon_col = next((c for c in month_candidates if c in df.columns), None)
    if year_col is not None and mon_col is not None:
        y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64").astype(str)
        m = pd.to_numeric(df[mon_col], errors="coerce").astype("Int64").astype(str).str.zfill(2)
        return pd.to_datetime(y + m, format="%Y%m", errors="coerce").dt.to_period("M").dt.to_timestamp()

    raise ValueError(
        "Could not find a month column in the weather file. Expected one of: "
        f"{WEATHER_MONTH_CANDIDATES}, or separate ANNEE/MOIS columns."
    )


def _choose_station(df: pd.DataFrame, station_id: str | None) -> tuple[pd.DataFrame, str | None, str | None]:
    station_col = next((c for c in WEATHER_STATION_CANDIDATES if c in df.columns), None)
    if station_col is None:
        return df, None, None

    if station_id:
        station_id_str = str(station_id)
        filtered = df[df[station_col].astype(str) == station_id_str]
        if filtered.empty:
            raise ValueError(f"No rows found for station_id={station_id} in column {station_col}.")
        return filtered, station_col, station_id_str

    available_weather_cols = [c for c in WEATHER_COLS_UPPER if c in df.columns]
    if not available_weather_cols:
        return df, station_col, None

    coverage = df.groupby(station_col)[available_weather_cols].count().sum(axis=1)
    best_station = str(coverage.sort_values(ascending=False).index[0])
    filtered = df[df[station_col].astype(str) == best_station]
    return filtered, station_col, best_station


def build_monthly_weather(weather_path: Path, station_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return monthly weather features: date_month + RR/NBJRR*/NBJBROU.

    If multiple stations are included, the default is selecting the station with
    the largest non-missing coverage on the required weather variables.
    """
    df = _read_weather_csv(weather_path)
    missing_weather = [c for c in WEATHER_COLS_UPPER if c not in df.columns]
    if missing_weather:
        raise ValueError(
            f"Missing weather columns {missing_weather}. Available columns include: {list(df.columns)[:30]}"
        )

    df["date_month"] = _parse_weather_month(df)
    df = df.dropna(subset=["date_month"]).copy()
    df, station_col, selected_station = _choose_station(df, station_id)

    for c in WEATHER_COLS_UPPER:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Assignment note: RR is recorded in tenths of millimeters, so divide by 10.
    df["RR"] = df["RR"] / 10.0

    monthly = (
        df.groupby("date_month", as_index=False)[WEATHER_COLS_UPPER]
        .mean()
        .sort_values("date_month")
        .reset_index(drop=True)
    )
    monthly[WEATHER_COLS_UPPER] = monthly[WEATHER_COLS_UPPER].interpolate(limit_direction="both").ffill().bfill()

    info = {
        "weather_path": str(weather_path),
        "weather_rows_after_station_filter": int(len(df)),
        "weather_month_rows": int(len(monthly)),
        "weather_month_start": str(monthly["date_month"].min().date()) if len(monthly) else None,
        "weather_month_end": str(monthly["date_month"].max().date()) if len(monthly) else None,
        "weather_station_column": station_col,
        "weather_station_selected": selected_station,
        "weather_features": WEATHER_COLS_UPPER,
        "rr_note": "RR has been divided by 10 according to the assignment description.",
    }
    return monthly, info


def _daily_index_to_date_column(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"])
        return daily.sort_values("date").reset_index(drop=True)

    idx_name = daily.index.name or "date"
    daily = daily.reset_index().rename(columns={idx_name: "date"})
    if "datetime" in daily.columns and "date" not in daily.columns:
        daily = daily.rename(columns={"datetime": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def merge_weather_to_daily(daily: pd.DataFrame, weather_monthly: pd.DataFrame) -> pd.DataFrame:
    """Expand monthly weather values to daily rows by year-month matching."""
    daily = _daily_index_to_date_column(daily)
    daily["date_month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    merged = daily.merge(weather_monthly, on="date_month", how="left")
    merged = merged.drop(columns=["date_month"])
    merged[WEATHER_COLS_UPPER] = merged[WEATHER_COLS_UPPER].interpolate(limit_direction="both").ffill().bfill()
    return merged


def chronological_split_raw(
    raw_csv: str | Path,
    train_out: str | Path = "data/train.csv",
    test_out: str | Path = "data/test.csv",
    test_days: int = 365,
    daily_preview_out: str | Path = "data/daily_all.csv",
    info_out: str | Path = "data/split_info.json",
    weather_csv: str | Path | None = None,
    station_id: str | None = None,
) -> dict:
    raw_csv = Path(raw_csv)
    train_out = Path(train_out)
    test_out = Path(test_out)
    daily_preview_out = Path(daily_preview_out)
    info_out = Path(info_out)

    df = read_power_csv(raw_csv)
    daily = aggregate_to_daily(df)
    daily = _daily_index_to_date_column(daily)

    weather_info = None
    if weather_csv:
        weather_monthly, weather_info = build_monthly_weather(Path(weather_csv), station_id=station_id)
        weather_monthly_out = daily_preview_out.parent / "weather_monthly_selected.csv"
        weather_monthly_out.parent.mkdir(parents=True, exist_ok=True)
        weather_monthly.to_csv(weather_monthly_out, index=False)
        daily = merge_weather_to_daily(daily, weather_monthly)

    if len(daily) <= test_days + 90:
        raise ValueError(
            f"Daily data is too short: {len(daily)} days. Need more than test_days + input_len = {test_days + 90}."
        )

    # Use the last `test_days` daily rows as the held-out test period.
    train_daily = daily.iloc[:-test_days].copy().reset_index(drop=True)
    test_daily = daily.iloc[-test_days:].copy().reset_index(drop=True)

    if train_daily.empty or test_daily.empty:
        raise ValueError("Split produced an empty train or test set. Please check the date range.")

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    daily_preview_out.parent.mkdir(parents=True, exist_ok=True)
    info_out.parent.mkdir(parents=True, exist_ok=True)

    train_daily.to_csv(train_out, index=False)
    test_daily.to_csv(test_out, index=False)
    daily.to_csv(daily_preview_out, index=False)

    info = {
        "raw_csv": str(raw_csv),
        "weather_used": bool(weather_csv),
        "weather_csv": str(weather_csv) if weather_csv else None,
        "split_rule": "chronological; last N daily rows as test",
        "test_days_requested": test_days,
        "daily_rows": int(len(daily)),
        "train_rows": int(len(train_daily)),
        "test_rows": int(len(test_daily)),
        "train_date_min": str(pd.to_datetime(train_daily["date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train_daily["date"]).max().date()),
        "test_date_min": str(pd.to_datetime(test_daily["date"]).min().date()),
        "test_date_max": str(pd.to_datetime(test_daily["date"]).max().date()),
        "columns": list(daily.columns),
        "weather_info": weather_info,
        "note": "Scaler is fitted only on train.csv in src/data.py to avoid test leakage.",
    }
    with open(info_out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=str, default="data/household_power_consumption.txt")
    parser.add_argument("--weather-csv", type=str, default=None, help="Optional Météo-France monthly MENSQ CSV file.")
    parser.add_argument("--station-id", type=str, default=None, help="Optional weather station id. Default selects best coverage station.")
    parser.add_argument("--train-out", type=str, default="data/train.csv")
    parser.add_argument("--test-out", type=str, default="data/test.csv")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--daily-preview-out", type=str, default="data/daily_all.csv")
    parser.add_argument("--info-out", type=str, default="data/split_info.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chronological_split_raw(
        raw_csv=args.raw_csv,
        train_out=args.train_out,
        test_out=args.test_out,
        test_days=args.test_days,
        daily_preview_out=args.daily_preview_out,
        info_out=args.info_out,
        weather_csv=args.weather_csv,
        station_id=args.station_id,
    )
