# QPredRNN: Quantum-Enhanced Spatiotemporal Video Prediction

A research starter repository for comparing a classical recurrent video-prediction baseline with a quantum-enhanced recurrent variant. It is designed for staged development in Antigravity and for running lightweight checks locally before any expensive training.

> **Research honesty:** this starter is a compact, custom research scaffold, not a bit-for-bit reproduction of THUML PredRNN. The model currently included is a small ConvLSTM-style baseline plus an optional PennyLane quantum-context module. If you use the upstream `thuml/predrnn-pytorch` implementation or its checkpoint, preserve its license/attribution and document exactly which code and weights you reuse. Do not describe borrowed components as independently invented.

## What this repository does

- Defines a reproducible video-frame dataset interface and preprocessing/manifest utilities.
- Splits KTH-style data by **subject before creating temporal windows** to avoid leakage.
- Includes a small classical recurrent model and an optional quantum-context variant.
- Provides smoke tests, train/validation loops, checkpoints, metric logging, and a small Streamlit demo.
- Keeps local tests cheap. Full training is intentionally a separate, later step on a GPU machine/cloud runtime.

## Important design choices

- Default input/output: 10 observed frames -> 10 future frames.
- Spatial size default: 64x64 grayscale for an initial experiment. You may use 160x120 after the pipeline is validated.
- Main metrics: MSE, MAE, PSNR, SSIM. LPIPS is not included by default because common implementations use pretrained feature networks.
- The quantum branch is a research hypothesis, not a guarantee of improvement or quantum advantage. A simulator runs on classical hardware.

## Quick start (CPU smoke test)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/make_toy_dataset.py --out data/toy --sequences 24 --frames 20 --size 64
python scripts/check_data.py --root data/toy --split-file data/toy/splits.json
pytest -q
python -m qpred.train --config configs/smoke.yaml --data-root data/toy --run-dir runs/smoke --device cpu
```

The smoke run verifies code paths only. It is **not** a meaningful performance result.

## Project phases

1. Audit upstream and write a provenance note.
2. Create environment and verify dependencies.
3. Build data manifest and subject-level splits.
4. Implement preprocessing and inspect frames/windows.
5. Test dataset loading and leakage checks.
6. Add the classical baseline and unit tests.
7. Add the quantum module behind an explicit configuration switch.
8. Validate gradients and model shapes on tiny synthetic data.
9. Commit and push all code/documentation **before** expensive training.
10. Train on a GPU runtime; evaluate only on the held-out test split once model selection is complete.

See `docs/` for the full workflow and Antigravity prompts.
