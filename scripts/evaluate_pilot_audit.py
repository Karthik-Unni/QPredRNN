from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import torch
from qpred.evaluate import load_checkpoint_for_eval, evaluate_windows
from qpred.data.dataset import FrameSequenceDataset


def main():
    print("=" * 75)
    print("POST-TRAINING EVALUATION AUDIT: QPredRNN PILOT VS BASELINES")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path("data/processed/kth64")
    assert data_root.exists(), f"Missing dataset at {data_root}"

    # Load Test dataset (held-out subjects 17-25)
    num_eval_windows = 100
    print(f"\nLoading KTH 'test' dataset (evaluating first {num_eval_windows} windows)...")
    ds_test = FrameSequenceDataset(data_root, split="test", input_frames=10, output_frames=10, stride=2)
    print(f"Total test windows available in dataset: {len(ds_test):,}")

    models_to_evaluate = [
        {
            "name": "QPredRNN (Pilot)",
            "checkpoint": Path("runs/kth_qpredrnn_pilot_gpu/checkpoint_best.pt"),
            "training_budget": "10 epochs (4,500 steps)",
            "architecture": "4-qubit VQC + ST-LSTM + Decouple Loss + RSS",
        },
        {
            "name": "PredRNN-V2 (Overnight)",
            "checkpoint": Path("runs/kth_predrnn_v2_overnight_gpu/checkpoint_best.pt"),
            "training_budget": "10 epochs (5,000 steps)",
            "architecture": "Classical Dual-Memory ST-LSTM + Decouple Loss + RSS",
        },
        {
            "name": "ConvLSTM (Pilot Baseline)",
            "checkpoint": Path("runs/kth_classical_gpu/checkpoint_best.pt"),
            "training_budget": "1 epoch (100 steps)",
            "architecture": "2-Layer ConvLSTM Baseline",
        },
    ]

    results = []

    # 1. Non-parametric Baselines
    print("\n--- Evaluating Constant Baselines ---")
    t0 = time.perf_counter()
    res_pers = evaluate_windows("persistence", ds_test, num_windows=num_eval_windows, device=device)
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
    res_mean = evaluate_windows("mean_context", ds_test, num_windows=num_eval_windows, device=device)
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
    for entry in models_to_evaluate:
        ckpt_p = entry["checkpoint"]
        print(f"\n--- Evaluating {entry['name']} ---")
        if not ckpt_p.exists():
            print(f"Checkpoint not found: {ckpt_p}")
            continue

        model, ckpt_meta = load_checkpoint_for_eval(ckpt_p, device=device)
        t0 = time.perf_counter()
        res_m = evaluate_windows(model, ds_test, num_windows=num_eval_windows, device=device)
        t_eval = time.perf_counter() - t0

        print(f"{entry['name']}:")
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
            "Best Epoch": ckpt_meta.get("epoch"),
            "Global Step": ckpt_meta.get("global_step"),
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

    # Save complete evaluation artifact
    out_json = Path("docs/pilot_test_evaluation.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEvaluation successfully saved to {out_json}")


if __name__ == "__main__":
    main()
