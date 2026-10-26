# QPredRNN: Quantum-Enhanced Spatiotemporal Video Prediction

A research framework for evaluating and comparing classical recurrent video prediction baselines (`ConvLSTM`, `PredRNN-V2`) with a quantum-enhanced recurrent model (`QPredRNN`).

---

## 1. Architecture Overview

### Classical Baselines
- **`ConvLSTM` (`qpred/models/convlstm.py`)**: Multi-layer Convolutional LSTM baseline for spatial-temporal sequence prediction.
- **`PredRNN-V2` (`qpred/models/predrnn_v2.py`)**: Spatiotemporal LSTM cells with dual memory flows (temporal cell memory $C_t^l$ and zigzag spatiotemporal memory $M_t^l$), auxiliary memory decoupling cosine loss, and official dual-schedule Reverse Scheduled Sampling ($r_\text{eta}$ exponential increase and $\text{eta}$ linear decrease).

### Quantum-Enhanced Model: `QPredRNN` (`qpred/models/qpredrnn.py`)
- **Quantum Context Modulator**:
  - Compresses recurrent spatiotemporal memory $M_t \in (B, C, H, W)$ to a compact latent vector via adaptive spatial pooling.
  - Linear encoder bounds features to rotation angles $\theta_i \in [-\pi/2, \pi/2]$ for state preparation.
  - Parameterized Quantum Circuit (PQC) executed via **PennyLane** (`default.qubit`):
    - **Angle Embedding**: $RY(\theta_i)$ rotations preparing the quantum register.
    - **Entangling Layers**: Parameterized $RX$ rotations followed by a closed ring of CNOT gates coupling adjacent qubits across depth $D$.
    - **Measurement**: Expectation values of Pauli-$Z$ observables $\langle Z_i \rangle \in [-1, 1]$ on each qubit wire.
  - Linear decoder and learned sigmoid gate project quantum expectation values back to memory channels, applying gated residual modulation to $M_t$.
- **Simulator Disclaimer**: All quantum circuits are simulated on classical hardware via PennyLane statevector simulation. No physical quantum advantage is claimed.

---

## 2. Interactive Research Studio (React + FastAPI)

Launch the full-stack React + TypeScript Research Studio with FastAPI backend:

```bash
# Production mode (FastAPI serves both API and built React SPA at http://127.0.0.1:8000):
C:\venvs\qpred-gpu\Scripts\python.exe scripts/run_app.py

# Frontend Development Mode (Hot-reload at http://localhost:5173 with API proxy):
cd frontend && npm run dev
```

### Full-Stack Studio Features:
- **Overview Dashboard**: High-level run metrics, hardware monitor, dataset status, and quantum configuration summary.
- **Visual Forecast Explorer**: Interactive frame-by-frame scrubbing, visual diff magma heatmaps, and per-horizon SSIM/MSE curves.
- **Training Monitor**: Real-time live loss curves, reverse scheduled sampling schedules ($r_\text{eta}$, $\text{eta}$), and decoupling loss tracking.
- **Multi-Model Comparison**: Side-by-side benchmark tables comparing ConvLSTM, PredRNN-V2, and QPredRNN against baselines.
- **Evaluation & Benchmarks**: Horizon-by-horizon metric degradation analysis across lead times $t \in [1, 10]$.
- **Report & Data Export**: Instant Markdown/JSON research artifact generation and model inventory.

### Alternative Lightweight Streamlit Dashboard:
```bash
# Launch Streamlit interface:
C:\venvs\qpred-gpu\Scripts\python.exe scripts/run_dashboard.py
```


---

## 3. Training & Checkpoint Resuming

Train classical or quantum models on toy or KTH datasets:

```bash
# 1-batch smoke test for PredRNN-V2
python -m qpred.train --config configs/kth_predrnn_v2_smoke.yaml --data-root data/processed/kth64 --run-dir runs/kth_predrnn_v2_smoke --device cuda

# 1-batch smoke test for QPredRNN
python -m qpred.train --config configs/kth_qpredrnn_smoke.yaml --data-root data/processed/kth64 --run-dir runs/kth_qpredrnn_smoke_gpu --device cuda

# Resume training from existing checkpoint
python -m qpred.train --config configs/kth_predrnn_v2_20ep.yaml --data-root data/processed/kth64 --run-dir runs/kth_predrnn_v2_overnight_gpu --resume runs/kth_predrnn_v2_overnight_gpu/checkpoint_last.pt --device cuda
```

### Checkpoint Resume Guarantees:
- Restores model weights, optimizer state buffers, `global_step`, and `best_val_loss`.
- Continues from `saved_epoch + 1` up to configured total epochs.
- Preserves existing `history.json` and `metrics.csv` rows and appends new epochs.
- Restores RNG states (Python, NumPy, PyTorch CPU/CUDA).
- Validates model/data configuration compatibility and rejects conflicting hyperparameters.

---

## 4. Test Suite

Run full automated tests (54 unit tests):
```bash
python -m pytest -q
```
Covers:
- Dataset sliding windows, splits, and boundary isolation (`test_dataset.py`)
- ConvLSTM baseline operations (`test_model.py`)
- PredRNN-V2 dual memory, memory decoupling loss, and dual-schedule RSS (`test_predrnn_v2.py`)
- Checkpoint resume continuation, optimizer restoration, and config validation (`test_resume.py`)
- QPredRNN quantum VQC gradients, inference leakage isolation, and training (`test_qpredrnn.py`)
- Sequence metrics: MSE, MAE, PSNR, SSIM (`test_metrics.py`)
- Evaluation pipeline: `evaluate_windows`, persistence/mean-context baselines (`test_evaluate.py`)
- Colab TPU results import, verification, and checksums (`test_import_colab_results.py`)
- Training history parsing and run scanning (`test_training.py`)

---

## 5. Collaborative GPU Training & Setup Guides

- **GPU Training Guide**: See [`docs/GPU_TRAINING_GUIDE.md`](docs/GPU_TRAINING_GUIDE.md) for full instructions on training ConvLSTM, PredRNN-V2, and QPredRNN on a dedicated GPU machine.
- **Collaborator Setup**: See [`docs/COLLABORATOR_SETUP.md`](docs/COLLABORATOR_SETUP.md) for step-by-step virtual environment creation and dataset restoration.
- **Environment Specifications**: See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for verified local stack versions and dependency declarations.

---

## 6. Research Audit & Model Verification Deliverables

The repository underwent a rigorous deep-learning systems and research audit. All checkpoints, model architectures, and training histories have been validated:

- **Model Implementation Audit**: [`docs/MODEL_IMPLEMENTATION_AUDIT.md`](docs/MODEL_IMPLEMENTATION_AUDIT.md) — Comprehensive audit of `ConvLSTM`, `PredRNN-V2`, `QPredRNN` (including PennyLane PQC forward pass and gradients), and baseline models.
- **Complete Checkpoint Inventory**: [`docs/CHECKPOINT_INVENTORY.csv`](docs/CHECKPOINT_INVENTORY.csv) — 24 discovered `.pt` checkpoints with SHA-256 fingerprints, file sizes, parameter counts, load status, and forward-pass smoke validation.
- **Experiment Registry**: [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) — Standardized human-friendly naming mapped to raw run IDs, epoch counts, validation losses, and experiment tiers.
- **Evaluation Protocol**: [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) — Held-out test split definition, metric formulations (MSE, MAE, PSNR, SSIM), and fair comparison criteria.
- **Verified Held-Out Benchmarks**: [`docs/VERIFIED_BENCHMARKS.csv`](docs/VERIFIED_BENCHMARKS.csv) — Measured performance on 100 test sequence windows from held-out KTH test persons (17–25).

