"""Rigorous Held-Out Test Evaluation Suite for QPredRNN, PredRNN-V2, and ConvLSTM.

Evaluates all primary candidate checkpoints under strictly identical conditions:
- Same held-out test split of KTH-64
- Same input context (10 frames) and forecast horizon (10 frames)
- Identical sliding window protocol (stride=2)
- Fixed number of test windows (e.g., 100 windows)
- Real metric calculation: MSE, MAE, PSNR (peak=1.0), SSIM
- Strict provenance tracking with checkpoint SHA-256 fingerprints
- Saves machine-readable results to docs/verified_evaluation_results.json and docs/VERIFIED_BENCHMARKS.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[1].resolve()
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
DATA_ROOT = ROOT / "data" / "processed" / "kth64"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpred.data.dataset import FrameSequenceDataset
from qpred.evaluate import load_checkpoint_for_eval, evaluate_windows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


CANDIDATE_RUNS = [
    {
        "model_id": "persistence",
        "model_name": "Persistence Baseline",
        "architecture_id": "baseline_persistence",
        "architecture_display": "Non-parametric (Repeat t=10)",
        "run_id": "baseline_persistence",
        "checkpoint_path": None,
        "is_baseline": True,
        "baseline_type": "persistence",
        "notes": "Static baseline repeating the final observed context frame",
    },
    {
        "model_id": "mean_context",
        "model_name": "Mean Context Baseline",
        "architecture_id": "baseline_mean",
        "architecture_display": "Non-parametric (Temporal Mean)",
        "run_id": "baseline_mean",
        "checkpoint_path": None,
        "is_baseline": True,
        "baseline_type": "mean_context",
        "notes": "Static baseline repeating temporal mean of context frames",
    },
    {
        "model_id": "convlstm_pilot",
        "model_name": "ConvLSTM — 10-Epoch Pilot",
        "architecture_id": "convlstm",
        "architecture_display": "ConvLSTM (VideoConvLSTM)",
        "run_id": "kth_classical_pilot_gpu",
        "checkpoint_path": ROOT / "runs" / "kth_classical_pilot_gpu" / "checkpoint_best.pt",
        "is_baseline": False,
        "notes": "Classical multi-layer 2D ConvLSTM baseline (10 epochs)",
    },
    {
        "model_id": "predrnn_v2_main",
        "model_name": "PredRNN-V2 — 10-Epoch Run",
        "architecture_id": "predrnn_v2",
        "architecture_display": "PredRNN-V2 (Dual-Memory ST-LSTM)",
        "run_id": "kth_predrnn_v2_overnight_gpu",
        "checkpoint_path": ROOT / "runs" / "kth_predrnn_v2_overnight_gpu" / "checkpoint_best.pt",
        "is_baseline": False,
        "notes": "Official PredRNN-V2 architecture with decoupling loss and reverse scheduled sampling",
    },
    {
        "model_id": "qpredrnn_pilot",
        "model_name": "QPredRNN — 10-Epoch Pilot",
        "architecture_id": "qpredrnn",
        "architecture_display": "QPredRNN (Hybrid VQC + ST-LSTM)",
        "run_id": "kth_qpredrnn_pilot_gpu",
        "checkpoint_path": ROOT / "runs" / "kth_qpredrnn_pilot_gpu" / "checkpoint_best.pt",
        "is_baseline": False,
        "notes": "Quantum-enhanced predictive RNN with 4-qubit VQC context modulator (10 epochs)",
    },
    {
        "model_id": "qpredrnn_50ep",
        "model_name": "QPredRNN — 50-Epoch Local Run",
        "architecture_id": "qpredrnn",
        "architecture_display": "QPredRNN (Hybrid VQC + ST-LSTM)",
        "run_id": "kth_qpredrnn_local_50ep",
        "checkpoint_path": ROOT / "runs" / "kth_qpredrnn_local_50ep" / "checkpoint_best.pt",
        "is_baseline": False,
        "notes": "Full 50-epoch converged QPredRNN model (best validation at epoch 30)",
    },
]


def run_evaluation(num_windows: int = 100, device_str: str = "cuda") -> list[dict[str, Any]]:
    print("=" * 75)
    print("QPredRNN COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 75)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if device.type == "cuda":
        print(f"[*] GPU device name: {torch.cuda.get_device_name(0)}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"KTH-64 dataset not found at {DATA_ROOT}")

    print(f"[*] Loading KTH test split (evaluating first {num_windows} temporal windows)...")
    ds_test = FrameSequenceDataset(DATA_ROOT, split="test", input_frames=10, output_frames=10, stride=2)
    print(f"[*] Total test windows available: {len(ds_test):,}")

    results = []

    for item in CANDIDATE_RUNS:
        m_id = item["model_id"]
        m_name = item["model_name"]
        print(f"\n--> Evaluating {m_name}...")

        t0 = time.perf_counter()
        ckpt_hash = None
        ckpt_epoch = None
        eval_ok = True
        err_msg = None

        if item["is_baseline"]:
            try:
                res = evaluate_windows(
                    item["baseline_type"],
                    ds_test,
                    num_windows=num_windows,
                    device=device,
                    output_frames=10,
                )
            except Exception as e:
                eval_ok = False
                err_msg = str(e)
                res = {}
        else:
            ckpt_path = item["checkpoint_path"]
            if not ckpt_path.exists():
                print(f"[!] Checkpoint missing: {ckpt_path}")
                continue
            ckpt_hash = sha256_file(ckpt_path)

            try:
                model, ckpt_meta = load_checkpoint_for_eval(str(ckpt_path), device=device)
                ckpt_epoch = ckpt_meta.get("epoch")
                res = evaluate_windows(
                    model,
                    ds_test,
                    num_windows=num_windows,
                    device=device,
                    output_frames=10,
                )
            except Exception as e:
                eval_ok = False
                err_msg = str(e)
                res = {}

        elapsed = time.perf_counter() - t0
        latency_per_sample_ms = (elapsed / num_windows) * 1000.0

        if eval_ok:
            print(f"    Done: MSE={res['mean_mse']:.6f} | MAE={res['mean_mae']:.6f} | PSNR={res['mean_psnr']:.2f} dB | SSIM={res['mean_ssim']:.4f} | Latency={latency_per_sample_ms:.1f}ms/clip")
            res_entry = {
                "model_id": m_id,
                "display_name": m_name,
                "architecture_id": item["architecture_id"],
                "architecture": item["architecture_display"],
                "run_id": item["run_id"],
                "checkpoint_path": item["checkpoint_path"].relative_to(ROOT).as_posix() if item["checkpoint_path"] else None,
                "checkpoint_sha256": ckpt_hash,
                "checkpoint_epoch": ckpt_epoch,
                "evaluation_split": "test",
                "evaluated_windows": num_windows,
                "total_available_windows": len(ds_test),
                "device": str(device),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "metrics": {
                    "mse": round(float(res["mean_mse"]), 8),
                    "mae": round(float(res["mean_mae"]), 8),
                    "psnr": round(float(res["mean_psnr"]), 4),
                    "ssim": round(float(res["mean_ssim"]), 6),
                    "latency_ms_per_window": round(latency_per_sample_ms, 2),
                    "total_eval_time_s": round(elapsed, 2),
                },
                "per_horizon": {
                    "mse": [round(float(v), 8) for v in res["per_horizon"]["mse"]],
                    "mae": [round(float(v), 8) for v in res["per_horizon"]["mae"]],
                    "psnr": [round(float(v), 4) for v in res["per_horizon"]["psnr"]],
                    "ssim": [round(float(v), 6) for v in res["per_horizon"]["ssim"]],
                },
                "distribution": res.get("distribution", {}),
                "status": "Verified Complete",
                "notes": item["notes"],
            }
        else:
            print(f"    Failed: {err_msg}")
            res_entry = {
                "model_id": m_id,
                "display_name": m_name,
                "architecture_id": item["architecture_id"],
                "architecture": item["architecture_display"],
                "run_id": item["run_id"],
                "checkpoint_path": item["checkpoint_path"].relative_to(ROOT).as_posix() if item["checkpoint_path"] else None,
                "checkpoint_sha256": ckpt_hash,
                "status": "Evaluation Failed",
                "error": err_msg,
                "notes": item["notes"],
            }

        results.append(res_entry)

    # Save JSON
    out_json = DOCS_DIR / "verified_evaluation_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[+] Saved detailed evaluation JSON to {out_json}")

    # Save CSV
    out_csv = DOCS_DIR / "VERIFIED_BENCHMARKS.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "display_name",
            "architecture",
            "run_id",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "test_mse",
            "test_mae",
            "test_psnr_db",
            "test_ssim",
            "latency_ms_per_window",
            "evaluated_windows",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            if r["status"] == "Verified Complete":
                writer.writerow({
                    "display_name": r["display_name"],
                    "architecture": r["architecture"],
                    "run_id": r["run_id"],
                    "checkpoint_epoch": r["checkpoint_epoch"],
                    "checkpoint_sha256": r["checkpoint_sha256"][:12] if r["checkpoint_sha256"] else "N/A",
                    "test_mse": r["metrics"]["mse"],
                    "test_mae": r["metrics"]["mae"],
                    "test_psnr_db": r["metrics"]["psnr"],
                    "test_ssim": r["metrics"]["ssim"],
                    "latency_ms_per_window": r["metrics"]["latency_ms_per_window"],
                    "evaluated_windows": r["evaluated_windows"],
                    "status": r["status"],
                })
            else:
                writer.writerow({
                    "display_name": r["display_name"],
                    "architecture": r["architecture"],
                    "run_id": r["run_id"],
                    "status": r["status"],
                })
    print(f"[+] Saved benchmark CSV to {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_evaluation(num_windows=args.num_windows, device_str=args.device)
