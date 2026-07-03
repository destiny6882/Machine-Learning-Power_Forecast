"""Run all required experiments: 3 models x 2 horizons x 5 seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .prepare_split import chronological_split_raw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--raw-csv", default="data/household_power_consumption.txt", help="Original UCI raw file. Used when train/test do not exist.")
    parser.add_argument("--test-days", type=int, default=365, help="Number of last days used as test if splitting raw data.")
    parser.add_argument("--epochs-short", type=int, default=60)
    parser.add_argument("--epochs-long", type=int, default=80)
    parser.add_argument("--batch-size-short", type=int, default=32)
    parser.add_argument("--batch-size-long", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (not Path(args.train_csv).exists()) or (not Path(args.test_csv).exists()):
        if not Path(args.raw_csv).exists():
            raise FileNotFoundError(
                f"Cannot find train/test files and raw file is also missing: {args.raw_csv}. "
                "Please download household_power_consumption.txt into data/ or specify --raw-csv."
            )
        chronological_split_raw(
            raw_csv=args.raw_csv,
            train_out=args.train_csv,
            test_out=args.test_csv,
            test_days=args.test_days,
        )

    models = ["lstm", "transformer", "mtcformer"]
    horizons = [90, 365]
    for horizon in horizons:
        for model in models:
            for seed in args.seeds:
                cmd = [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--train-csv",
                    args.train_csv,
                    "--test-csv",
                    args.test_csv,
                    "--model",
                    model,
                    "--horizon",
                    str(horizon),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs_short if horizon == 90 else args.epochs_long),
                    "--batch-size",
                    str(args.batch_size_short if horizon == 90 else args.batch_size_long),
                    "--output-dir",
                    args.output_dir,
                ]
                if args.cpu:
                    cmd.append("--cpu")
                print("\n" + "=" * 90)
                print("Running:", " ".join(cmd))
                print("=" * 90)
                subprocess.run(cmd, check=True)
