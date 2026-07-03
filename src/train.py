"""Train and evaluate one forecasting model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import inverse_target, make_train_val_test_datasets, prepare_data
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def evaluate(model, loader, device, target_mean: float, target_scale: float) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            gts.append(y.cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    gt_scaled = np.concatenate(gts, axis=0)
    pred_raw = inverse_target(pred_scaled, target_mean, target_scale)
    gt_raw = inverse_target(gt_scaled, target_mean, target_scale)
    mse = float(np.mean((pred_raw - gt_raw) ** 2))
    mae = float(np.mean(np.abs(pred_raw - gt_raw)))
    return {"mse": mse, "mae": mae, "pred_raw": pred_raw, "gt_raw": gt_raw}


def save_curve(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 4.8))
    x = np.arange(1, len(gt) + 1)
    plt.plot(x, gt, label="Ground Truth")
    plt.plot(x, pred, label="Prediction")
    plt.xlabel("Future day")
    plt.ylabel("Daily global active power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_loss_curve(train_losses, val_losses, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.2))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss on scaled target")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    prepared = prepare_data(args.train_csv, args.test_csv)
    train_ds, val_ds, test_ds = make_train_val_test_datasets(
        prepared,
        input_len=args.input_len,
        horizon=args.horizon,
        val_ratio=args.val_ratio,
        stride=args.stride,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, input_dim=len(prepared.feature_cols), input_len=args.input_len, horizon=args.horizon)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    out_dir = Path(args.output_dir) / f"{args.model}_h{args.horizon}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"{args.model} H={args.horizon} seed={args.seed} epoch={epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(running))
        scheduler.step()

        train_loss = float(np.mean(running))
        train_losses.append(train_loss)

        model.eval()
        val_running = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_running.append(float(criterion(model(x), y).item()))
        val_loss = float(np.mean(val_running)) if val_running else train_loss
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    test_result = evaluate(model, test_loader, device, prepared.target_mean, prepared.target_scale)
    pred_raw = test_result.pop("pred_raw")
    gt_raw = test_result.pop("gt_raw")

    # Save the first rolling-window prediction for report visualization.
    first_pred = pred_raw[0]
    first_gt = gt_raw[0]
    save_curve(
        first_pred,
        first_gt,
        out_dir / "curve.png",
        title=f"{args.model} prediction vs ground truth, horizon={args.horizon}",
    )
    save_loss_curve(
        train_losses,
        val_losses,
        out_dir / "loss.png",
        title=f"{args.model} training curve, horizon={args.horizon}, seed={args.seed}",
    )

    np.savetxt(out_dir / "prediction_first_window.csv", np.vstack([first_gt, first_pred]).T, delimiter=",", header="ground_truth,prediction", comments="")

    metrics = {
        "model": args.model,
        "horizon": args.horizon,
        "seed": args.seed,
        "mse": test_result["mse"],
        "mae": test_result["mae"],
        "best_val_loss": best_val,
        "input_len": args.input_len,
        "feature_cols": prepared.feature_cols,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_test_samples": len(test_ds),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--test-csv", type=str, default="data/test.csv")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer", "mtcformer"])
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=90, choices=[90, 365])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
