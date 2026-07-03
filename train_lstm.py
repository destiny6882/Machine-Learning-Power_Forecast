"""Run only the LSTM experiments in the old power_forecast project.

Examples:
    python train_lstm.py --horizon 90 --epochs 80 --seeds 0 1 2 3 4
    python train_lstm.py --horizon 365 --epochs 100 --seeds 0 1 2 3 4
    python train_lstm.py --horizon all --epochs-short 80 --epochs-long 100

Weather preprocessing example:
    python train_lstm.py --horizon 90 --weather-csv data/MENSQ_92_previous-1950-2024.csv --force-split
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.prepare_split import chronological_split_raw

MODEL_NAME = "lstm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only LSTM experiments.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument(
        "--raw-csv",
        default="data/household_power_consumption.txt",
        help="Original UCI raw file. Used when train/test do not exist or --force-split is set.",
    )
    parser.add_argument("--weather-csv", default=None, help="Optional Météo-France monthly MENSQ CSV file.")
    parser.add_argument("--station-id", default=None, help="Optional weather station id. Default selects best coverage station.")
    parser.add_argument("--force-split", action="store_true", help="Regenerate train/test even if they already exist.")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--horizon", default="all", choices=["90", "365", "all"])
    parser.add_argument("--epochs", type=int, default=None, help="Epochs used when --horizon is 90 or 365.")
    parser.add_argument("--epochs-short", type=int, default=80, help="Epochs for horizon=90 when --horizon all.")
    parser.add_argument("--epochs-long", type=int, default=100, help="Epochs for horizon=365 when --horizon all.")
    parser.add_argument("--batch-size-short", type=int, default=32)
    parser.add_argument("--batch-size-long", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def ensure_split(args: argparse.Namespace) -> None:
    train_exists = Path(args.train_csv).exists()
    test_exists = Path(args.test_csv).exists()
    if train_exists and test_exists and not args.force_split:
        if args.weather_csv:
            print(
                "[Info] Existing train/test files are used. "
                "If they were generated without weather, rerun with --force-split."
            )
        return

    if not Path(args.raw_csv).exists():
        raise FileNotFoundError(
            f"Cannot find raw file: {args.raw_csv}. "
            "Please download household_power_consumption.txt into data/ or specify --raw-csv."
        )

    chronological_split_raw(
        raw_csv=args.raw_csv,
        train_out=args.train_csv,
        test_out=args.test_csv,
        test_days=args.test_days,
        weather_csv=args.weather_csv,
        station_id=args.station_id,
    )


def run_one(args: argparse.Namespace, horizon: int, seed: int) -> None:
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = args.epochs_short if horizon == 90 else args.epochs_long
    batch_size = args.batch_size_short if horizon == 90 else args.batch_size_long

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--train-csv",
        args.train_csv,
        "--test-csv",
        args.test_csv,
        "--model",
        MODEL_NAME,
        "--input-len",
        str(args.input_len),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--val-ratio",
        str(args.val_ratio),
        "--stride",
        str(args.stride),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        args.output_dir,
    ]
    if args.cpu:
        cmd.append("--cpu")

    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_split(args)
    horizons = [90, 365] if args.horizon == "all" else [int(args.horizon)]
    for horizon in horizons:
        for seed in args.seeds:
            run_one(args, horizon, seed)


if __name__ == "__main__":
    main()
