"""Summarize repeated experiments and create report-ready table image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def collect_metrics(outputs: Path) -> pd.DataFrame:
    rows = []
    for path in outputs.glob("*/metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {outputs}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["horizon", "model"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
    )
    return summary


def save_table_image(summary: pd.DataFrame, out_png: Path) -> None:
    show = summary.copy()
    for col in ["mse_mean", "mse_std", "mae_mean", "mae_std"]:
        show[col] = show[col].map(lambda x: f"{x:.4f}")
    show.columns = ["Horizon", "Model", "MSE mean", "MSE std", "MAE mean", "MAE std"]
    fig_h = max(2.6, 0.45 * len(show) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.25)
    ax.set_title("Forecasting results over five runs", pad=14, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.outputs)
    df = collect_metrics(out_dir)
    df.to_csv(out_dir / "all_metrics.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    save_table_image(summary, out_dir / "summary_table.png")
    print(summary)
    print(f"Saved: {out_dir / 'summary.csv'}")
    print(f"Saved: {out_dir / 'summary_table.png'}")
