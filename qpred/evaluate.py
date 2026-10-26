from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qpred.models import build_model
from qpred.data.dataset import FrameSequenceDataset
from qpred.metrics import compute_sequence_metrics


def load_checkpoint_for_eval(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load model and metadata dictionary from a saved checkpoint for evaluation."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    dev = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)

    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint at {ckpt_path} missing config metadata.")

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()
    return model, checkpoint


def evaluate_windows(
    predictor: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    dataset: FrameSequenceDataset,
    num_windows: int | None = None,
    device: torch.device | str = "cpu",
    output_frames: int = 10,
) -> dict[str, Any]:
    """Evaluate predictor or baseline callable on a sequence dataset across windows.

    Args:
        predictor: PyTorch Module or callable taking (1, T_in, C, H, W) -> (1, T_out, C, H, W).
                   Can also be string 'persistence' or 'mean_context'.
        dataset: FrameSequenceDataset providing (x, y) pairs.
        num_windows: Number of temporal sliding windows to evaluate (None for full dataset).
        device: Execution device for neural models.
        output_frames: Expected horizon length T_out.

    Returns:
        Dictionary of aggregated metrics, per-horizon degradation arrays, and tensor distribution.
    """
    dev = torch.device(device)
    total_samples = len(dataset)
    n_eval = min(num_windows, total_samples) if num_windows is not None else total_samples

    all_mse: list[float] = []
    all_mae: list[float] = []
    all_psnr: list[float] = []
    all_ssim: list[float] = []

    horizon_mse: list[list[float]] = [[] for _ in range(output_frames)]
    horizon_mae: list[list[float]] = [[] for _ in range(output_frames)]
    horizon_psnr: list[list[float]] = [[] for _ in range(output_frames)]
    horizon_ssim: list[list[float]] = [[] for _ in range(output_frames)]

    preds_min = float("inf")
    preds_max = float("-inf")
    preds_sum = 0.0
    preds_sq_sum = 0.0
    preds_count = 0
    all_finite = True

    is_module = isinstance(predictor, nn.Module)
    if is_module:
        predictor.eval()

    for idx in range(n_eval):
        x, y = dataset[idx]  # x: (T_in, C, H, W), y: (T_out, C, H, W)

        if predictor == "persistence":
            # Baseline: Repeat last context frame for all future steps
            pred = x[-1:].repeat(output_frames, 1, 1, 1)
        elif predictor == "mean_context":
            # Baseline: Repeat average context frame for all future steps
            pred = x.mean(dim=0, keepdim=True).repeat(output_frames, 1, 1, 1)
        elif callable(predictor):
            x_b = x.unsqueeze(0).to(dev)
            with torch.no_grad():
                out_b = predictor(x_b)
            pred = out_b[0].detach().cpu()
        else:
            raise ValueError(f"Unknown predictor type: {predictor}")

        # Finite and distribution checks
        if not torch.isfinite(pred).all():
            all_finite = False

        p_min = float(pred.min())
        p_max = float(pred.max())
        preds_min = min(preds_min, p_min)
        preds_max = max(preds_max, p_max)

        p_numel = pred.numel()
        preds_sum += float(pred.sum())
        preds_sq_sum += float((pred ** 2).sum())
        preds_count += p_numel

        # Compute sequence metrics
        met = compute_sequence_metrics(pred, y)
        all_mse.append(met["mse"])
        all_mae.append(met["mae"])
        all_psnr.append(met["psnr"])
        all_ssim.append(met["ssim"])

        for h_step in range(min(output_frames, len(met["per_frame"]["mse"]))):
            horizon_mse[h_step].append(met["per_frame"]["mse"][h_step])
            horizon_mae[h_step].append(met["per_frame"]["mae"][h_step])
            horizon_psnr[h_step].append(met["per_frame"]["psnr"][h_step])
            horizon_ssim[h_step].append(met["per_frame"]["ssim"][h_step])

    pred_mean = preds_sum / max(preds_count, 1)
    pred_variance = max(0.0, (preds_sq_sum / max(preds_count, 1)) - (pred_mean ** 2))
    pred_std = float(np.sqrt(pred_variance))

    return {
        "num_windows": n_eval,
        "is_finite": all_finite,
        "mean_mse": float(np.mean(all_mse)),
        "mean_mae": float(np.mean(all_mae)),
        "mean_psnr": float(np.mean(all_psnr)),
        "mean_ssim": float(np.mean(all_ssim)),
        "distribution": {
            "min": float(preds_min),
            "max": float(preds_max),
            "mean": float(pred_mean),
            "std": float(pred_std),
        },
        "per_horizon": {
            "mse": [float(np.mean(h)) for h in horizon_mse],
            "mae": [float(np.mean(h)) for h in horizon_mae],
            "psnr": [float(np.mean(h)) for h in horizon_psnr],
            "ssim": [float(np.mean(h)) for h in horizon_ssim],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video prediction checkpoints on dataset splits.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--data-root", default="data/processed/kth64", help="Path to processed dataset root.")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"], help="Split to evaluate on.")
    parser.add_argument("--num-windows", type=int, default=100, help="Number of sliding windows to evaluate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda).")
    args = parser.parse_args()

    model, ckpt = load_checkpoint_for_eval(args.checkpoint, device=args.device)
    ds = FrameSequenceDataset(Path(args.data_root), split=args.split, input_frames=10, output_frames=10, stride=2)

    res = evaluate_windows(model, ds, num_windows=args.num_windows, device=args.device)
    print(f"Evaluation on split='{args.split}' ({res['num_windows']} windows):")
    print(f"  MSE:  {res['mean_mse']:.6f}")
    print(f"  MAE:  {res['mean_mae']:.6f}")
    print(f"  PSNR: {res['mean_psnr']:.2f} dB")
    print(f"  SSIM: {res['mean_ssim']:.4f}")
    print(f"  Distribution: [{res['distribution']['min']:.4f}, {res['distribution']['max']:.4f}], mean={res['distribution']['mean']:.4f}, std={res['distribution']['std']:.4f}")


if __name__ == "__main__":
    main()
