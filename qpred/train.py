from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from qpred.data.dataset import FrameSequenceDataset
from qpred.models import (
    build_model,
    reserve_schedule_sampling_exp,
    generate_scheduled_sampling_masks,
)


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict:
    """Capture current RNG states across Python, NumPy, and PyTorch CPU/CUDA."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict | None) -> None:
    """Restore RNG states across Python, NumPy, and PyTorch CPU/CUDA."""
    if states is None:
        warnings.warn(
            "[QPred] Resumed checkpoint does not contain 'rng_states'. "
            "Random number generator states were not restored.",
            UserWarning,
            stacklevel=2,
        )
        return

    if "python" in states and states["python"] is not None:
        random.setstate(states["python"])
    if "numpy" in states and states["numpy"] is not None:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states and states["torch_cpu"] is not None:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and states["torch_cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["torch_cuda"])
        except Exception as e:
            warnings.warn(f"[QPred] Could not restore CUDA RNG state: {e}", UserWarning, stacklevel=2)


def validate_resume_config(current_config: dict, saved_config: dict | None) -> None:
    """Validate compatibility between current config and checkpoint's config."""
    if saved_config is None:
        warnings.warn(
            "[QPred] Resumed checkpoint does not contain 'config'. "
            "Configuration compatibility validation skipped.",
            UserWarning,
            stacklevel=2,
        )
        return

    curr_model = current_config.get("model", {})
    saved_model = saved_config.get("model", {})
    for key in ["name", "hidden_channels", "num_layers", "image_channels", "kernel_size"]:
        if key in curr_model and key in saved_model:
            if curr_model[key] != saved_model[key]:
                raise ValueError(
                    f"Incompatible model parameter '{key}': current config has '{curr_model[key]}', "
                    f"but checkpoint was trained with '{saved_model[key]}'."
                )

    curr_data = current_config.get("data", {})
    saved_data = saved_config.get("data", {})
    for key in ["image_size", "input_frames", "output_frames", "stride"]:
        if key in curr_data and key in saved_data:
            if curr_data[key] != saved_data[key]:
                raise ValueError(
                    f"Incompatible data parameter '{key}': current config has {curr_data[key]}, "
                    f"but checkpoint was trained with {saved_data[key]}."
                )

    curr_train = current_config.get("training", {})
    saved_train = saved_config.get("training", {})
    if curr_train.get("batch_size") != saved_train.get("batch_size"):
        raise ValueError(
            f"Incompatible batch size: current config has {curr_train.get('batch_size')}, "
            f"but checkpoint was trained with {saved_train.get('batch_size')}."
        )

    curr_rss = curr_train.get("reverse_scheduled_sampling", {})
    saved_rss = saved_train.get("reverse_scheduled_sampling", {})
    if curr_rss.get("enabled", False) != saved_rss.get("enabled", False):
        raise ValueError(
            f"Incompatible reverse_scheduled_sampling enabled status: current config has "
            f"{curr_rss.get('enabled', False)}, but checkpoint was trained with "
            f"{saved_rss.get('enabled', False)}."
        )
    if curr_rss.get("enabled", False):
        for key in ["r_sampling_step_1", "r_sampling_step_2", "r_exp_alpha"]:
            if key in curr_rss and key in saved_rss:
                if curr_rss[key] != saved_rss[key]:
                    raise ValueError(
                        f"Incompatible reverse_scheduled_sampling parameter '{key}': "
                        f"current config has {curr_rss[key]}, but checkpoint had {saved_rss[key]}."
                    )


def get_rss_probabilities(global_step: int, rss_cfg: dict | None) -> tuple[float, float]:
    """Compute context and future sampling probabilities based on training iteration."""
    if not rss_cfg or not rss_cfg.get("enabled", False):
        return 1.0, 0.0  # 1.0 context (100% ground truth), 0.0 future (100% autoregressive)

    step_1 = int(rss_cfg.get("r_sampling_step_1", 25000))
    step_2 = int(rss_cfg.get("r_sampling_step_2", 50000))
    alpha = float(rss_cfg.get("r_exp_alpha", 5000.0))

    return reserve_schedule_sampling_exp(
        itr=global_step,
        r_sampling_step_1=step_1,
        r_sampling_step_2=step_2,
        r_exp_alpha=alpha,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    decouple_weight: float = 0.0,
    rss_cfg: dict | None = None,
    global_step: int = 0,
) -> tuple[float, float, float, float, int]:
    """Run one epoch of training and return average (mse_loss, decouple_loss, r_eta, eta, updated_global_step)."""
    model.train()
    total_mse = 0.0
    total_decouple = 0.0
    num_batches = 0
    latest_r_eta, latest_eta = 1.0, 0.0

    supports_decouple = getattr(model, "has_decouple_loss", False)

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        r_eta, eta = get_rss_probabilities(global_step, rss_cfg)
        latest_r_eta, latest_eta = r_eta, eta

        optimizer.zero_grad()

        if supports_decouple:
            mask_context, mask_future = None, None
            if rss_cfg and rss_cfg.get("enabled", False):
                mask_context, mask_future = generate_scheduled_sampling_masks(
                    batch_size=x.size(0),
                    input_frames=x.size(1),
                    output_frames=y.size(1),
                    r_eta=r_eta,
                    eta=eta,
                    device=device,
                )

            y_pred, decouple_loss = model(
                x,
                target=y,
                mask_context=mask_context,
                mask_future=mask_future,
                return_decouple_loss=True,
            )
            mse_loss = criterion(y_pred, y)
            total_loss = mse_loss + (decouple_weight * decouple_loss if decouple_weight > 0.0 else 0.0)
        else:
            y_pred = model(x)
            mse_loss = criterion(y_pred, y)
            decouple_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss detected at batch {batch_idx}: total={total_loss.item()}, mse={mse_loss.item()}"
            )

        total_loss.backward()
        optimizer.step()

        total_mse += mse_loss.item()
        total_decouple += decouple_loss.item()
        num_batches += 1
        global_step += 1

    if num_batches == 0:
        raise ValueError("Training loader yielded zero batches; check dataset and batch_size.")

    return total_mse / num_batches, total_decouple / num_batches, latest_r_eta, latest_eta, global_step


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

        # Clean evaluation: purely autoregressive prediction from x without ground-truth masks
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
    resume_path: str | Path | None = None,
) -> dict:
    """Run full training pipeline as specified by configuration, with optional checkpoint resume."""
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
    model_cfg = config.get("model", {})

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

    # Auxiliary loss and sampling configurations
    decouple_weight = float(model_cfg.get("decouple_loss_weight", 0.0))
    rss_cfg = train_cfg.get("reverse_scheduled_sampling", {})

    # Checkpoint Resume Early Validation & Loading
    checkpoint = None
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Checkpoint file to resume not found: {resume_file}")

        print(f"[QPred] Loading checkpoint from {resume_file} to resume training...")
        checkpoint = torch.load(resume_file, map_location=device, weights_only=False)

        # Validate configuration compatibility early
        validate_resume_config(current_config=config, saved_config=checkpoint.get("config"))

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
    print(f"[QPred] Model '{model_cfg.get('name')}' initialized with {total_params:,} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: list[dict] = []
    global_step = 0
    start_epoch = 1

    # Checkpoint State Restoration
    if checkpoint is not None:
        # Restore model weights
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {resume_file} is missing 'model_state_dict'.")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[QPred] Restored model weights from checkpoint.")

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print("[QPred] Restored optimizer state from checkpoint.")
            except Exception as e:
                warnings.warn(f"[QPred] Could not restore optimizer state: {e}", UserWarning, stacklevel=2)

        saved_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = saved_epoch + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))

        # Restore RNG states
        set_rng_states(checkpoint.get("rng_states"))

        # History restoration
        if (out_dir / "history.json").exists():
            try:
                existing_hist = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
                history = [h for h in existing_hist if h.get("epoch", 0) <= saved_epoch]
            except Exception:
                history = []
        elif "history" in checkpoint and isinstance(checkpoint["history"], list):
            history = [h for h in checkpoint["history"] if h.get("epoch", 0) <= saved_epoch]
        else:
            history = []

        # If checkpoint_best.pt does not exist in out_dir, preserve best checkpoint into out_dir
        best_target = out_dir / "checkpoint_best.pt"
        if not best_target.exists():
            candidate_best = resume_file.parent / "checkpoint_best.pt"
            if candidate_best.exists():
                shutil.copy2(candidate_best, best_target)
            else:
                torch.save(checkpoint, best_target)

        print(
            f"[QPred] Checkpoint resume successful: starting at epoch {start_epoch}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.6f}."
        )

    # Save resolved configuration copy
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    # CSV metrics handling
    csv_path = out_dir / "metrics.csv"
    if resume_path and csv_path.exists():
        preserved_rows = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    preserved_rows.append(header)
                for row in reader:
                    if row and row[0].isdigit() and int(row[0]) <= (start_epoch - 1):
                        preserved_rows.append(row)
        except Exception:
            preserved_rows = []

        if preserved_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(preserved_rows)
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
        else:
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["epoch", "train_mse", "train_decouple", "r_eta", "eta", "val_mse", "best_val_mse"])
    else:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_mse", "train_decouple", "r_eta", "eta", "val_mse", "best_val_mse"])
        if history:
            for rec in history:
                csv_writer.writerow([
                    rec["epoch"],
                    f"{rec['train_loss']:.6f}",
                    f"{rec.get('decouple_loss', 0.0):.6f}",
                    f"{rec.get('r_eta', 1.0):.4f}",
                    f"{rec.get('eta', 0.0):.4f}",
                    f"{rec['val_loss']:.6f}",
                    f"{rec.get('best_val_loss', rec['val_loss']):.6f}",
                ])
            csv_file.flush()

    if start_epoch > epochs:
        print(
            f"[QPred] Resumed checkpoint at epoch {start_epoch - 1} has already reached "
            f"or exceeded configured epochs ({epochs}). Training complete."
        )
        csv_file.close()
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    else:
        print(f"[QPred] Starting training from epoch {start_epoch} to {epochs}...")
        for epoch in range(start_epoch, epochs + 1):
            train_mse, train_decouple, r_eta, eta, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=max_train_batches,
                decouple_weight=decouple_weight,
                rss_cfg=rss_cfg,
                global_step=global_step,
            )

            val_mse = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=max_val_batches,
            )

            is_best = val_mse < best_val_loss
            if is_best:
                best_val_loss = val_mse

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(train_mse),
                "decouple_loss": float(train_decouple),
                "r_eta": float(r_eta),
                "eta": float(eta),
                "val_loss": float(val_mse),
                "best_val_loss": float(best_val_loss),
                "is_best": is_best,
            }
            history.append(record)

            if is_best:
                ckpt_best = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_mse,
                    "best_val_loss": best_val_loss,
                    "train_loss": train_mse,
                    "decouple_loss": train_decouple,
                    "r_eta": r_eta,
                    "eta": eta,
                    "config": config,
                    "seed": seed,
                    "rng_states": get_rng_states(),
                    "history": history,
                }
                torch.save(ckpt_best, out_dir / "checkpoint_best.pt")

            ckpt_last = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_mse,
                "best_val_loss": best_val_loss,
                "train_loss": train_mse,
                "decouple_loss": train_decouple,
                "r_eta": r_eta,
                "eta": eta,
                "config": config,
                "seed": seed,
                "rng_states": get_rng_states(),
                "history": history,
            }
            torch.save(ckpt_last, out_dir / "checkpoint_last.pt")

            csv_writer.writerow([
                epoch,
                f"{train_mse:.6f}",
                f"{train_decouple:.6f}",
                f"{r_eta:.4f}",
                f"{eta:.4f}",
                f"{val_mse:.6f}",
                f"{best_val_loss:.6f}",
            ])
            csv_file.flush()

            # Save JSON history after every epoch
            with open(out_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            best_mark = " (NEW BEST)" if is_best else ""
            if decouple_weight > 0.0:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_mse:.6f} | "
                    f"Decouple: {train_decouple:.6f} (r_eta: {r_eta:.2f}, eta: {eta:.2f}) | Val MSE: {val_mse:.6f}{best_mark}"
                )
            else:
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}{best_mark}")

        csv_file.close()

    # Verification: Load best checkpoint and verify forward pass
    if (out_dir / "checkpoint_best.pt").exists():
        print(f"[QPred] Verifying checkpoint reload from {out_dir / 'checkpoint_best.pt'}...")
        checkpoint = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
        eval_model = build_model(checkpoint.get("config", config))
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
        "epochs_completed": max(epochs, start_epoch - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QPredRNN models.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--data-root", default=None, help="Root folder of processed dataset.")
    parser.add_argument("--run-dir", default=None, help="Directory to save logs and checkpoints.")
    parser.add_argument("--device", default=None, help="Device to use: cpu or cuda.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        data_root=args.data_root,
        run_dir=args.run_dir,
        device_name=args.device,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
