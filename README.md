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

## 2. Interactive Research Dashboard

Launch the Streamlit research dashboard to inspect models, visual rollouts, error heatmaps, and metrics:

```bash
# Launch via Python script:
python scripts/run_dashboard.py

# Or directly with Streamlit:
streamlit run qpred/dashboard.py
```

### Dashboard Capabilities:
- **Model & Checkpoint Loader**: Select `ConvLSTM`, `PredRNN-V2`, or `QPredRNN`, inspect parameters, saved epochs, best validation losses, and quantum circuit configuration.
- **Visual Rollout**: Side-by-side observed context frames ($t=1\dots 10$), model predictions ($t=11\dots 20$), ground truth targets, and absolute error heatmaps.
- **Quantitative Metrics**: Real-time MSE, MAE, PSNR (dB), SSIM, inference latency (ms), and per-horizon degradation curves.
- **Multi-Model Comparison**: Compare all three models side-by-side on the exact same sequence.
- **Training Curves**: Plot training and validation loss convergence across multiple experiment runs.
- **Report Export**: Download Markdown summaries and evaluation records.

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

Run full automated tests (43 unit tests):
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
