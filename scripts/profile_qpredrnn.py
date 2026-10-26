import sys
import os
import time
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
import yaml

from qpred.train import run_training
from qpred.models import build_model
from qpred.data.dataset import FrameSequenceDataset
from qpred.metrics import compute_sequence_metrics

def main():
    print("=" * 70)
    print("STAGE 1: QPredRNN Profiling Run (100 train batches, 20 val batches)")
    print("=" * 70)

    config_path = Path("configs/kth_qpredrnn_profile.yaml")
    data_root = Path("data/processed/kth64")
    run_dir = Path("runs/kth_qpredrnn_profile_gpu")

    assert config_path.exists(), f"Missing config: {config_path}"
    assert data_root.exists(), f"Missing data: {data_root}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t_start = time.perf_counter()
    res = run_training(
        config_path=config_path,
        data_root=data_root,
        run_dir=run_dir,
        device_name=str(device),
    )
    t_end = time.perf_counter()
    wall_clock_sec = t_end - t_start

    peak_cuda_mb = 0.0
    if device.type == "cuda":
        peak_cuda_bytes = torch.cuda.max_memory_allocated()
        peak_cuda_mb = peak_cuda_bytes / (1024.0 * 1024.0)

    print("\n" + "=" * 70)
    print("STAGE 2: Profiling Measurement & Artifact Verification")
    print("=" * 70)

    # 1. Inspect history.json
    history_file = run_dir / "history.json"
    assert history_file.exists(), f"history.json not found in {run_dir}"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    ep_entry = history[0]

    train_mse = ep_entry["train_loss"]
    val_mse = ep_entry["val_loss"]
    decouple_loss = ep_entry.get("decouple_loss", 0.0)
    global_step = ep_entry.get("global_step", 100)

    train_steps = global_step
    steps_per_sec = train_steps / wall_clock_sec if wall_clock_sec > 0 else 0.0
    sec_per_step = wall_clock_sec / train_steps if train_steps > 0 else 0.0

    print(f"Wall-clock Time:       {wall_clock_sec:.2f} seconds ({wall_clock_sec/60.0:.2f} minutes)")
    print(f"Completed Steps:       {train_steps} training batches + 20 validation batches")
    print(f"Throughput:            {steps_per_sec:.3f} steps/second ({sec_per_step:.3f} seconds/step)")
    print(f"Peak CUDA Memory:      {peak_cuda_mb:.2f} MB")
    print(f"Training Loss (MSE):   {train_mse:.6f}")
    print(f"Validation Loss (MSE): {val_mse:.6f}")
    print(f"Decouple Loss:         {decouple_loss:.6f}")

    # 2. Checkpoints Verification
    ckpt_best_path = run_dir / "checkpoint_best.pt"
    ckpt_last_path = run_dir / "checkpoint_last.pt"
    assert ckpt_best_path.exists(), "checkpoint_best.pt missing"
    assert ckpt_last_path.exists(), "checkpoint_last.pt missing"

    ckpt_best = torch.load(ckpt_best_path, map_location=device, weights_only=False)
    ckpt_last = torch.load(ckpt_last_path, map_location=device, weights_only=False)

    print("\n--- Checkpoint Reload Verification ---")
    print(f"checkpoint_best epoch: {ckpt_best.get('epoch')}, step: {ckpt_best.get('global_step')}, val_loss: {ckpt_best.get('best_val_loss'):.6f}")
    print(f"checkpoint_last epoch: {ckpt_last.get('epoch')}, step: {ckpt_last.get('global_step')}, val_loss: {ckpt_last.get('val_loss'):.6f}")

    # 3. Model Reload and Prediction Distribution on Validation Sequence
    model = build_model(ckpt_best["config"]).to(device)
    model.load_state_dict(ckpt_best["model_state_dict"])
    model.eval()

    ds_val = FrameSequenceDataset(data_root, split="val", input_frames=10, output_frames=10, stride=2)
    x_val, y_val = ds_val[0]
    with torch.no_grad():
        pred_val = model(x_val.unsqueeze(0).to(device))[0].cpu()

    metrics = compute_sequence_metrics(pred_val, y_val)

    print("\n--- Model Output Distribution on Validation Window 0 ---")
    print(f"Input  x: range [{x_val.min():.4f}, {x_val.max():.4f}], mean={x_val.mean():.4f}, std={x_val.std():.4f}, is_finite={torch.isfinite(x_val).all().item()}")
    print(f"Target y: range [{y_val.min():.4f}, {y_val.max():.4f}], mean={y_val.mean():.4f}, std={y_val.std():.4f}, is_finite={torch.isfinite(y_val).all().item()}")
    print(f"Pred   y^: range [{pred_val.min():.4f}, {pred_val.max():.4f}], mean={pred_val.mean():.4f}, std={pred_val.std():.4f}, is_finite={torch.isfinite(pred_val).all().item()}")
    print(f"Validation Metrics: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")

    print("\nPer-frame Prediction Statistics across Horizon:")
    for t in range(10):
        print(f"  Horizon t={t+1:2d}: mean={pred_val[t].mean():.5f}, std={pred_val[t].std():.5f}, min={pred_val[t].min():.5f}, max={pred_val[t].max():.5f}")

    # 4. Verify Gradient Reach to Quantum Circuit & Classical Weights
    print("\n--- Gradient Reachability Verification ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    x_sample = x_val.unsqueeze(0).to(device)
    y_sample = y_val.unsqueeze(0).to(device)

    pred_train, d_loss = model(x_sample, target=y_sample, return_decouple_loss=True)
    loss = torch.nn.functional.mse_loss(pred_train, y_sample) + 0.1 * d_loss
    loss.backward()

    # Quantum parameters
    vqc_weights = model.quantum_modulator.vqc.weights
    vqc_grad = vqc_weights.grad
    assert vqc_grad is not None, "Quantum VQC layer received no gradients!"
    assert torch.isfinite(vqc_grad).all(), "Quantum VQC gradients contain non-finite numbers!"
    vqc_grad_norm = float(vqc_grad.norm().item())

    # Classical ST-LSTM cell parameters
    st_cell_conv = model.cells[0].conv_x.weight
    assert st_cell_conv.grad is not None, "Classical ST-LSTM conv_x received no gradients!"
    cell_grad_norm = float(st_cell_conv.grad.norm().item())

    # Readout parameters
    readout_w = model.readout[0].weight
    assert readout_w.grad is not None, "Readout conv received no gradients!"
    readout_grad_norm = float(readout_w.grad.norm().item())

    print(f"Quantum VQC parameter grad norm:    {vqc_grad_norm:.6f} (is_finite={torch.isfinite(vqc_grad).all().item()})")
    print(f"ST-LSTM cell_0 conv_x grad norm:   {cell_grad_norm:.6f} (is_finite={torch.isfinite(st_cell_conv.grad).all().item()})")
    print(f"Readout conv parameter grad norm:   {readout_grad_norm:.6f} (is_finite={torch.isfinite(readout_w.grad).all().item()})")

    # 5. Check Non-Modification of Previous Artifacts
    print("\n--- Previous Artifacts Integrity Check ---")
    prev_artifacts = [
        Path("runs/kth_classical_gpu/checkpoint_best.pt"),
        Path("runs/kth_predrnn_v2_overnight_gpu/checkpoint_best.pt"),
        Path("runs/kth_qpredrnn_smoke_gpu/checkpoint_best.pt"),
    ]
    for pa in prev_artifacts:
        assert pa.exists(), f"Previous artifact was deleted or moved: {pa}"
        print(f"Confirmed intact: {pa} (size: {pa.stat().st_size:,} bytes)")

    # Save summary profile report
    profile_summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "wall_clock_sec": wall_clock_sec,
        "completed_train_steps": train_steps,
        "steps_per_sec": steps_per_sec,
        "sec_per_step": sec_per_step,
        "peak_cuda_mb": peak_cuda_mb,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "decouple_loss": decouple_loss,
        "pred_val_stats": {
            "min": float(pred_val.min()),
            "max": float(pred_val.max()),
            "mean": float(pred_val.mean()),
            "std": float(pred_val.std()),
        },
        "gradient_norms": {
            "vqc_weights": vqc_grad_norm,
            "st_lstm_conv_x": cell_grad_norm,
            "readout": readout_grad_norm,
        },
    }
    (run_dir / "profile_summary.json").write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
    print(f"\nProfile summary successfully recorded in {run_dir / 'profile_summary.json'}")

if __name__ == "__main__":
    main()
