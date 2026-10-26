# QPredRNN Canonical Experiment Registry

This document establishes the canonical mapping between original file-system run directories and human-readable experiment display names across the research platform.

---

## 1. Primary Research & Benchmark Experiments

| Display Name | Original Run ID | Model Architecture | Run Category | Configured Epochs | Completed Epochs | Best Val Loss (MSE) | Best Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|
| **QPredRNN — 50-Epoch Local Run** | `kth_qpredrnn_local_50ep` | `qpredrnn` | Primary Research | 50 | 50 | **0.001136** (ep 30) | `checkpoint_best.pt` (`647244761c...`) | Completed (Verified) |
| **QPredRNN — 10-Epoch Pilot Run** | `kth_qpredrnn_pilot_gpu` | `qpredrnn` | Pilot Study | 10 | 10 | **0.001231** (ep 9) | `checkpoint_best.pt` (`0e2174c5a2...`) | Completed (Verified) |
| **PredRNN-V2 — 10-Epoch Overnight Run** | `kth_predrnn_v2_overnight_gpu` | `predrnn_v2` | Primary Baseline | 10 | 10 | **0.001277** (ep 10) | `checkpoint_best.pt` (`922e43f095...`) | Completed (Verified) |
| **ConvLSTM — 10-Epoch Pilot Run** | `kth_classical_pilot_gpu` | `convlstm` | Pilot Baseline | 10 | 10 | **0.001276** (ep 6) | `checkpoint_best.pt` (`c216a746ab...`) | Completed (Verified) |

---

## 2. Experimental, Smoke, & Diagnostic Runs

These artifacts are preserved for provenance, debugging, and continuous integration, but are separated from the primary benchmark rankings:

| Display Name | Original Run ID | Architecture | Category | Completed Steps / Epochs | Notes |
|---|---|---|---|---|---|
| **ConvLSTM — Early 3-Epoch Run** | `kth_classical_gpu` | `convlstm` | Experimental | 3 epochs | Replaced by full 10-epoch pilot run |
| **PredRNN-V2 — 1-Epoch Smoke Test** | `kth_predrnn_v2_smoke` | `predrnn_v2` | Smoke Test | 1 epoch (1 batch) | Pipeline test fixture |
| **QPredRNN — GPU Smoke Test** | `kth_qpredrnn_smoke_gpu` | `qpredrnn` | Smoke Test | 1 epoch (1 batch) | Verified GPU execution of PennyLane VQC |
| **QPredRNN — Manual Smoke Test** | `kth_qpredrnn_smoke_manual` | `qpredrnn` | Smoke Test | 1 epoch (1 batch) | Manual parameter check |
| **QPredRNN — Profiling Run** | `kth_qpredrnn_profile_gpu` | `qpredrnn` | Profiling | 1 epoch (1 batch) | Generated `profile_summary.json` |
| **KTH Smoke Test (CPU)** | `kth_smoke` | `convlstm` | Smoke Test | 1 epoch | CPU baseline sanity check |
| **KTH Smoke Test (GPU)** | `kth_smoke_gpu` | `convlstm` | Smoke Test | 1 epoch | GPU baseline sanity check |
| **Toy Synthetic Smoke Test** | `smoke` | `convlstm` | Smoke Test | 1 epoch | Toy moving-squares test |

---

## 3. Associated Artifact Paths

Every run directory contains:
- `config.yaml`: Model architecture, training hyperparameters, dataset paths.
- `checkpoint_best.pt`: Lowest validation loss model weights.
- `checkpoint_last.pt`: Final epoch training state (model + optimizer + scheduler).
- `history.json`: Array of per-epoch logs (`train_loss`, `val_loss`, `decouple_loss`, `r_eta`, `eta`).
- `metrics.csv`: Tabular copy of training progression.
- `training_console.log`: Console stdout/stderr stream from the training job.
