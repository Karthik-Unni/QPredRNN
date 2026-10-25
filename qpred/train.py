from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from qpred.data.dataset import FrameSequenceDataset
from qpred.models import build_model


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Run one epoch of training and return the average MSE loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at batch {batch_idx}: {loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training loader yielded zero batches; check dataset and batch_size.")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Evaluate model on validation loader and return average MSE loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation loader yielded zero batches; check split.")

    return total_loss / num_batches


def run_training(
    config_path: str | Path,
    data_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    device_name: str | None = None,
) -> dict:
    """Run full training pipeline as specified by configuration."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve output directory and data root
    out_dir = Path(run_dir) if run_dir else Path(config.get("run_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(data_root) if data_root else Path(config.get("data_root", "data/toy"))
    if not root.exists():
        raise FileNotFoundError(f"Data root directory not found: {root}")

    # Resolve device
    if device_name:
        chosen_device = device_name
    else:
        chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(chosen_device)
    print(f"[QPred] Device: {device}")

    # Seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"[QPred] Random seed: {seed}")

    # Config sections
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    image_size = data_cfg.get("image_size", 64)
    input_frames = data_cfg.get("input_frames", 5)
    output_frames = data_cfg.get("output_frames", 5)
    stride = data_cfg.get("stride", 1)

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 0)
    epochs = train_cfg.get("epochs", 1)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_train_batches = train_cfg.get("max_train_batches", None)
    max_val_batches = train_cfg.get("max_val_batches", None)

    # Datasets
    print(f"[QPred] Loading train and validation datasets from {root}...")
    train_ds = FrameSequenceDataset(
        root=root,
        split="train",
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
        image_size=image_size,
        augment=True,
    )
    val_ds = FrameSequenceDataset(
        root=root,
        split="val",
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Build model
    model = build_model(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[QPred] Model '{config.get('model', {}).get('name')}' initialized with {total_params:,} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: list[dict] = []

    # Save resolved configuration copy
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])

    print(f"[QPred] Starting training for {epochs} epoch(s)...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            ckpt_best = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": config,
                "seed": seed,
            }
            torch.save(ckpt_best, out_dir / "checkpoint_best.pt")

        ckpt_last = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "config": config,
            "seed": seed,
        }
        torch.save(ckpt_last, out_dir / "checkpoint_last.pt")

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "is_best": is_best,
        }
        history.append(record)
        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_val_loss:.6f}"])
        csv_file.flush()

        best_mark = " (NEW BEST)" if is_best else ""
        print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}{best_mark}")

    csv_file.close()

    # Save complete JSON history
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Verification: Load best checkpoint and verify forward pass
    print(f"[QPred] Verifying checkpoint reload from {out_dir / 'checkpoint_best.pt'}...")
    checkpoint = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    eval_model = build_model(checkpoint["config"])
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.to(device)
    eval_model.eval()

    sample_x, sample_y = next(iter(val_loader))
    with torch.no_grad():
        preds = eval_model(sample_x.to(device))
    assert preds.shape == sample_y.shape, f"Shape mismatch: {preds.shape} vs {sample_y.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain non-finite numbers!"
    print(f"[QPred] Checkpoint verified! Predictions shape: {tuple(preds.shape)}, finite: True")

    print(f"[QPred] Training finished. Artifacts saved in {out_dir}")
    return {
        "out_dir": str(out_dir),
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QPredRNN classical model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--data-root", default=None, help="Root folder of processed dataset.")
    parser.add_argument("--run-dir", default=None, help="Directory to save logs and checkpoints.")
    parser.add_argument("--device", default=None, help="Device to use: cpu or cuda.")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        data_root=args.data_root,
        run_dir=args.run_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
