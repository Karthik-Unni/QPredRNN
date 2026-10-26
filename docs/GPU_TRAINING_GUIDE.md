# Collaborative GPU Training Guide

**Branch**: `main`
**Repository**: `https://github.com/Karthik-Unni/QPredRNN.git`
**Target Environment**: Linux or Windows workstation with high-performance NVIDIA GPU (e.g., RTX 3080/3090/4090, A100, H100)

This guide provides instructions for training all three spatiotemporal video prediction models (**ConvLSTM**, **PredRNN-V2**, and **QPredRNN**) on a high-performance GPU machine.

---

## 1. Clone the `main` Branch

Clone the repository and switch to the `main` branch:

```bash
git clone https://github.com/Karthik-Unni/QPredRNN.git
cd QPredRNN
git checkout main
```

> [!NOTE]
> This branch contains **no `.gitignore` file**. To protect your local environment and dataset from accidental commits, never run `git add .`. Always stage specific files explicitly (e.g., `git add configs/my_experiment.yaml`).

---

## 2. Environment Setup

Always create an isolated Python 3.11 (or 3.10) virtual environment:

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install PyTorch with GPU Support & Dependencies

### Step A: Install PyTorch with CUDA
Select the CUDA build matching your driver. For modern NVIDIA GPUs:

```bash
# CUDA 12.1 (Recommended for RTX 40-series / A100 / H100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8 (Compatible with RTX 30-series / older drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step B: Install QPredRNN Dependencies
Install the package in editable mode with all optional dependencies:
```bash
pip install -e ".[all]"
```
*(Or alternatively: `pip install -r requirements.txt` followed by `pip install -e .`)*

---

## 4. Verify GPU & Test Suite

Verify that PyTorch detects your GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run the automated test suite to confirm the classical models, PennyLane quantum simulator, metrics, and dataset loaders function correctly:
```bash
python -m pytest -q
```
**Expected**: All 45 unit and integration tests must pass.

---

## 5. Obtain and Place the KTH Dataset

Due to size limits, preprocessed video sequences are shared separately via Google Drive as `QPredRNN_data.zip`.

### Expected Layout After Extraction:
```
QPredRNN/
â””â”€â”€ data/
    â”œâ”€â”€ manifest.json
    â”œâ”€â”€ toy/
    â””â”€â”€ processed/
        â””â”€â”€ kth64/
            â”œâ”€â”€ splits.json
            â””â”€â”€ sequences/
                â”œâ”€â”€ person01_boxing_d1_take01/
                â””â”€â”€ ... (2,391 annotated sequence takes)
```

### Verification:
Ensure the split definition exists before launching training:
```bash
# Linux
test -f data/processed/kth64/splits.json && echo "Dataset OK"

# Windows PowerShell
Test-Path data\processed\kth64\splits.json
```

---

## 6. Pre-Flight Verification: Short Smoke Tests

Before starting any multi-hour training run, execute a 1-batch smoke test on each architecture to verify GPU memory allocation and forward-backward stability:

```bash
# Smoke test ConvLSTM
python -m qpred.train --config configs/smoke.yaml --device cuda

# Smoke test PredRNN-V2
python -m qpred.train --config configs/kth_predrnn_v2_smoke.yaml --device cuda

# Smoke test QPredRNN (Quantum-Enhanced)
python -m qpred.train --config configs/kth_qpredrnn_smoke.yaml --device cuda
```

---

## 7. Model Training Protocol: Separate Runs & Configs

Each model has its own configuration file and isolated run directory in `runs/`. Keep them separated so histories and checkpoints remain isolated for the dashboard comparison.

### Model 1: ConvLSTM Baseline (Spatial-Temporal LSTM)
- **Architecture**: 2-layer ConvLSTM (32 hidden channels, kernel 3Ã—3).
- **Configuration**: `configs/kth_classical.yaml` (50 epochs) or `configs/kth_classical_pilot.yaml` (pilot).
- **Execution Command**:
  ```bash
  python -m qpred.train --config configs/kth_classical.yaml --run-dir runs/kth_convlstm_50ep_gpu --device cuda
  ```

### Model 2: PredRNN-V2 Baseline (Spatiotemporal Memory & RSS)
- **Architecture**: Dual-memory ST-LSTM with decoupling loss ($\lambda=0.1$) and Reverse Scheduled Sampling ($r_\eta$ context schedule, $\eta$ autoregressive schedule).
- **Configuration**: `configs/kth_predrnn_v2_overnight.yaml` (10 epochs, 500 batches/epoch) or extended to 50 epochs.
- **Execution Command**:
  ```bash
  python -m qpred.train --config configs/kth_predrnn_v2_overnight.yaml --run-dir runs/kth_predrnn_v2_10ep_gpu --device cuda
  ```

### Model 3: QPredRNN (Quantum-Enhanced Predictive RNN)
- **Architecture**: Dual-memory ST-LSTM modulated by a 4-qubit, depth-2 Parameterized Quantum Circuit (VQC) via PennyLane `default.qubit` simulator.
- **Configuration**: `configs/kth_qpredrnn_pilot.yaml` (10 epochs, 500 batches/epoch matching PredRNN-V2) or extended to 50 epochs.
- **Execution Command**:
  ```bash
  python -m qpred.train --config configs/kth_qpredrnn_pilot.yaml --run-dir runs/kth_qpredrnn_10ep_gpu --device cuda
  ```

---

## 8. Preserving Artifacts & Checkpoints

Each training execution writes the following files to its designated `--run-dir`:
1. `checkpoint_best.pt`: Checkpoint with lowest validation loss.
2. `checkpoint_last.pt`: Most recent checkpoint with full optimizer and RNG state (enables `--resume`).
3. `history.json`: Epoch-by-epoch record of `train_loss`, `val_loss`, `decouple_loss`, and sampling rates.
4. `metrics.csv`: Tabular log for plotting.
5. `config.yaml`: Snapshot of the exact configuration used.

### Resuming Checkpoints
If training is interrupted, resume seamlessly without resetting epoch counters:
```bash
python -m qpred.train --config configs/kth_qpredrnn_pilot.yaml --resume runs/kth_qpredrnn_10ep_gpu/checkpoint_last.pt --device cuda
```

---

## 9. Launching the Interactive Research Dashboard

Inspect rollouts, compare degradation curves across future steps $t=1\dots 10$, and view multi-model metrics (MSE, MAE, PSNR, SSIM):

```bash
python scripts/run_dashboard.py
```
Open `http://localhost:8501` in your browser.
