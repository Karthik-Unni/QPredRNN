"""Evaluation script for Colab TPU v5e-1 50-epoch benchmark runs on held-out KTH test split.

Evaluates imported checkpoints from Google Colab against non-parametric baselines
(Persistence and Mean Context) on identical held-out test windows.

Outputs:
    docs/tpu_test_evaluation.json

Usage:
    python scripts/evaluate_tpu_results.py [--num-windows 100] [--device cuda|cpu]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import torch
from qpred.data.dataset import FrameSequenceDataset
from qpred.evaluate import load_checkpoint_for_eval, evaluate_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Colab TPU 50-epoch checkpoints on held-out test split.")
    parser.add_argument("--data-root", default="data/processed/kth64", help="Path to processed dataset root.")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"], help="Dataset split to evaluate.")
    parser.add_argument("--num-windows", type=int, default=100, help="Number of sliding windows to evaluate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda).")
    parser.add_argument("--output-json", default="docs/tpu_test_evaluation.json", help="Path to save evaluation results JSON.")
    args = parser.parse_args()

    print("=" * 80)
    print("FINAL 50-EPOCH TPU BENCHMARK EVALUATION (Colab TPU v5e-1)")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Dataset not found at {data_root.resolve()}")
        sys.exit(1)

    print(f"\nLoading KTH '{args.split}' dataset (evaluating {args.num_windows} windows)...")
    ds = FrameSequenceDataset(data_root, split=args.split, input_frames=10, output_frames=10, stride=2)
    print(f"Total {args.split} windows available in dataset: {len(ds):,}")

    tpu_models = [
        {
            "name": "ConvLSTM (TPU 50ep)",
            "checkpoint": Path("runs/kth_classical_tpu_50ep/checkpoint_best.pt"),
            "training_budget": "50 epochs (TPU v5e-1)",
            "architecture": "2-Layer Classical ConvLSTM Baseline",
        },
        {
            "name": "PredRNN-V2 (TPU 50ep)",
            "checkpoint": Path("runs/kth_predrnn_v2_tpu_50ep/checkpoint_best.pt"),
            "training_budget": "50 epochs (TPU v5e-1)",
            "architecture": "Classical Dual-Memory ST-LSTM + Decouple Loss + RSS",
        },
        {
            "name": "QPredRNN (TPU 50ep)",
            "checkpoint": Path("runs/kth_qpredrnn_tpu_50ep/checkpoint_best.pt"),
            "training_budget": "50 epochs (TPU v5e-1)",
            "architecture": "4-qubit VQC + ST-LSTM + Decouple Loss + RSS",
        },
    ]

    results = []

    # 1. Non-parametric Baselines
    print("\n--- Evaluating Non-Parametric Baselines ---")
    t0 = time.perf_counter()
    res_pers = evaluate_windows("persistence", ds, num_windows=args.num_windows, device=device)
    t_pers = time.perf_counter() - t0
    results.append({
        "Model": "Persistence Baseline (Last Frame)",
        "Architecture": "Non-parametric (repeat t=10)",
        "Training Budget": "0 parameters / 0 steps",
        "MSE": res_pers["mean_mse"],
        "MAE": res_pers["mean_mae"],
        "PSNR (dB)": res_pers["mean_psnr"],
        "SSIM": res_pers["mean_ssim"],
        "Distribution": res_pers["distribution"],
        "Per-Horizon MSE": res_pers["per_horizon"]["mse"],
        "Per-Horizon PSNR": res_pers["per_horizon"]["psnr"],
        "Per-Horizon SSIM": res_pers["per_horizon"]["ssim"],
        "Latency_s": t_pers,
    })
    print(f"Persistence: MSE={res_pers['mean_mse']:.6f}, MAE={res_pers['mean_mae']:.6f}, PSNR={res_pers['mean_psnr']:.2f} dB, SSIM={res_pers['mean_ssim']:.4f}")

    t0 = time.perf_counter()
    res_mean = evaluate_windows("mean_context", ds, num_windows=args.num_windows, device=device)
    t_mean = time.perf_counter() - t0
    results.append({
        "Model": "Mean Context Baseline",
        "Architecture": "Non-parametric (temporal mean t=1..10)",
        "Training Budget": "0 parameters / 0 steps",
        "MSE": res_mean["mean_mse"],
        "MAE": res_mean["mean_mae"],
        "PSNR (dB)": res_mean["mean_psnr"],
        "SSIM": res_mean["mean_ssim"],
        "Distribution": res_mean["distribution"],
        "Per-Horizon MSE": res_mean["per_horizon"]["mse"],
        "Per-Horizon PSNR": res_mean["per_horizon"]["psnr"],
        "Per-Horizon SSIM": res_mean["per_horizon"]["ssim"],
        "Latency_s": t_mean,
    })
    print(f"Mean Context: MSE={res_mean['mean_mse']:.6f}, MAE={res_mean['mean_mae']:.6f}, PSNR={res_mean['mean_psnr']:.2f} dB, SSIM={res_mean['mean_ssim']:.4f}")

    # 2. Checkpoint Evaluations
    for entry in tpu_models:
        ckpt_p = entry["checkpoint"]
        print(f"\n--- Evaluating {entry['name']} ---")
        if not ckpt_p.exists():
            print(f"  ⚠️ Checkpoint not found at: {ckpt_p}")
            print(f"     Run 'python scripts/import_colab_results.py --source-dir <colab_dir>' once Colab training completes.")
            continue

        try:
            model, ckpt_meta = load_checkpoint_for_eval(ckpt_p, device=device)
            t0 = time.perf_counter()
            res_m = evaluate_windows(model, ds, num_windows=args.num_windows, device=device)
            t_eval = time.perf_counter() - t0

            saved_ep = ckpt_meta.get("epoch")
            saved_step = ckpt_meta.get("global_step")
            print(f"  Best Epoch: {saved_ep}, Global Step: {saved_step}")
            print(f"  MSE:  {res_m['mean_mse']:.6f}")
            print(f"  MAE:  {res_m['mean_mae']:.6f}")
            print(f"  PSNR: {res_m['mean_psnr']:.2f} dB")
            print(f"  SSIM: {res_m['mean_ssim']:.4f}")
            print(f"  Range: [{res_m['distribution']['min']:.4f}, {res_m['distribution']['max']:.4f}], mean={res_m['distribution']['mean']:.4f}, std={res_m['distribution']['std']:.4f}")
            print(f"  Finite: {res_m['is_finite']}")

            results.append({
                "Model": entry["name"],
                "Architecture": entry["architecture"],
                "Training Budget": entry["training_budget"],
                "Checkpoint": str(ckpt_p),
                "Best Epoch": saved_ep,
                "Global Step": saved_step,
                "MSE": res_m["mean_mse"],
                "MAE": res_m["mean_mae"],
                "PSNR (dB)": res_m["mean_psnr"],
                "SSIM": res_m["mean_ssim"],
                "Distribution": res_m["distribution"],
                "Per-Horizon MSE": res_m["per_horizon"]["mse"],
                "Per-Horizon PSNR": res_m["per_horizon"]["psnr"],
                "Per-Horizon SSIM": res_m["per_horizon"]["ssim"],
                "Latency_s": t_eval,
            })
        except Exception as err:
            print(f"  ❌ Error evaluating {entry['name']}: {err}")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEvaluation successfully written to {out_json.resolve()}")


if __name__ == "__main__":
    main()
