# Google Colab TPU Results Handoff, Verification & Evaluation Guide

This guide describes the end-to-end workflow for locating, downloading, safely importing, verifying, and evaluating the **50-epoch benchmark runs** currently training on Google Colab TPU v5e-1.

---

## 1. Overview of Expected Training Artifacts

The sequential Colab TPU training pipeline produces three benchmark experiment runs:

| Experiment Folder | Architecture | Target Training Budget | Key Checkpoint File |
| :--- | :--- | :--- | :--- |
| **`kth_classical_tpu_50ep`** | 2-Layer Classical ConvLSTM Baseline | 50 Epochs | `checkpoint_best.pt` |
| **`kth_predrnn_v2_tpu_50ep`** | Classical PredRNN-V2 + Spatiotemporal LSTM + RSS | 50 Epochs | `checkpoint_best.pt` |
| **`kth_qpredrnn_tpu_50ep`** | 4-Qubit VQC + Spatiotemporal LSTM + Decouple Loss + RSS | 50 Epochs | `checkpoint_best.pt` |

Each folder must contain the following verified files:
1. `checkpoint_best.pt` — Checkpoint weights and metadata from the epoch with the lowest validation MSE.
2. `checkpoint_last.pt` — Final checkpoint at epoch 50.
3. `config.yaml` — Exact training and model configuration.
4. `history.json` — Epoch-by-epoch loss, metrics, and RSS sampling probabilities.
5. `metrics.csv` — Tabular convergence record.
6. `*.log` (e.g., `train.log`) — Complete console stdout and timing log.

---

## 2. Locating & Downloading from Google Drive

### Step 2.1: Locate in Google Drive
In your Google Drive connected to the Colab session:
```text
MyDrive/
└── QPredRNN_Colab/
    └── training_runs/
        ├── kth_classical_tpu_50ep/
        ├── kth_predrnn_v2_tpu_50ep/
        └── kth_qpredrnn_tpu_50ep/
```

### Step 2.2: Download Archive
1. In Google Drive web UI or Google Drive for Desktop, zip or download the `training_runs` folder.
2. Save the download archive to a local staging directory **outside** your Git workspace, for example:
   ```powershell
   # Windows PowerShell staging directory:
   New-Item -ItemType Directory -Path "C:\temp\colab_tpu_staging" -Force
   ```
3. Extract the archive into `C:\temp\colab_tpu_staging\training_runs\`.
4. Ensure the three folder names match exactly:
   - `C:\temp\colab_tpu_staging\training_runs\kth_classical_tpu_50ep`
   - `C:\temp\colab_tpu_staging\training_runs\kth_predrnn_v2_tpu_50ep`
   - `C:\temp\colab_tpu_staging\training_runs\kth_qpredrnn_tpu_50ep`

---

## 3. Safe Import and Verification

The repository provides an automated, non-destructive import utility: [`scripts/import_colab_results.py`](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/scripts/import_colab_results.py).

### Step 3.1: Dry-Run Inspection
Always perform a dry run first to verify checkpoint integrity, target epoch completion, and history consistency **without modifying any local files**:

```powershell
python scripts/import_colab_results.py --source-dir "C:\temp\colab_tpu_staging\training_runs" --dry-run
```

The tool will:
- Check for all 5 required artifacts plus training logs in each folder.
- Attempt CPU deserialization of `checkpoint_best.pt` and `checkpoint_last.pt`.
- Confirm that `checkpoint_last.pt` reached the configured target epoch (50 epochs).
- Cross-check `history.json` against `metrics.csv` and `config.yaml`.
- Compute SHA-256 checksums for all files.
- Refuse to touch any existing local files.

### Step 3.2: Perform Verified Import
Once the dry-run confirms that all runs are complete and uncorrupted, import the runs into the project's local `runs/` directory:

```powershell
python scripts/import_colab_results.py --source-dir "C:\temp\colab_tpu_staging\training_runs"
```

> [!NOTE]
> By default, `import_colab_results.py` refuses to overwrite existing checkpoints. If you need to re-import or update an existing run folder, supply the explicit flag:
> ```powershell
> python scripts/import_colab_results.py --source-dir "C:\temp\colab_tpu_staging\training_runs" --overwrite
> ```

---

## 4. TPU Checkpoint Compatibility & CPU Export Protocol

Checkpoints trained on Google Colab TPU v5e-1 are typically saved via `torch.save(...)` or `xm.save(...)`.

### Windows Deserialization Check
When `import_colab_results.py` inspects a checkpoint, it executes:
```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
```

If the tensors were saved with active XLA device types and your Windows environment does not have `torch_xla`, a deserialization error may occur:
```text
RuntimeError: Device type XLA is not supported
```

### Safe Colab CPU Export Script
If this occurs, do **not** attempt to modify or hack the checkpoint on Windows. Run this one-minute export cell in the original Colab notebook runtime to produce pure CPU-serialized checkpoints without altering the original files:

```python
# Run inside Google Colab (where torch_xla is available):
import torch
from pathlib import Path

runs_dir = Path("/content/drive/MyDrive/QPredRNN_Colab/training_runs")

for run_folder in ["kth_classical_tpu_50ep", "kth_predrnn_v2_tpu_50ep", "kth_qpredrnn_tpu_50ep"]:
    target_dir = runs_dir / run_folder
    for ckpt_name in ["checkpoint_best.pt", "checkpoint_last.pt"]:
        ckpt_path = target_dir / ckpt_name
        if ckpt_path.exists():
            print(f"Exporting {ckpt_path} to CPU...")
            # Load with CPU map location
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Ensure every tensor is detached and CPU-resident
            cpu_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in ckpt["model_state_dict"].items()}
            ckpt_cpu = dict(ckpt)
            ckpt_cpu["model_state_dict"] = cpu_state
            # Save non-destructively as CPU checkpoint
            cpu_export_path = target_dir / ckpt_name
            torch.save(ckpt_cpu, cpu_export_path)
            print(f"  -> Saved clean CPU checkpoint to {cpu_export_path}")

print("CPU export complete.")
```

After re-exporting, re-run `python scripts/import_colab_results.py`.

---

## 5. Running the Final Comparative Evaluation

Once the three 50-epoch TPU runs are imported into `runs/`, run the standardized evaluation script:

```powershell
python scripts/evaluate_tpu_results.py --split test --num-windows 100 --device cuda
```

### What This Command Measures:
1. **Persistence Baseline**: Last-observed frame ($t=10$) repeated for all future steps ($t=11\dots 20$).
2. **Mean Context Baseline**: Temporal average frame across observed context ($t=1\dots 10$).
3. **ConvLSTM (50ep)**: Classical 2-layer baseline.
4. **PredRNN-V2 (50ep)**: Dual-memory classical recurrence with reverse scheduled sampling.
5. **QPredRNN (50ep)**: Variational quantum circuit modulation with spatiotemporal recurrence.

### Output Artifacts:
- Results and per-horizon degradation curves ($t=1\dots 10$) are saved to [`docs/tpu_test_evaluation.json`](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/docs/tpu_test_evaluation.json).
- **Note**: The pilot evaluation report ([`docs/PILOT_EVALUATION_REPORT.md`](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/docs/PILOT_EVALUATION_REPORT.md)) and its JSON are strictly preserved and never overwritten.

---

## 6. Interactive Visual Inspection in the Dashboard

Launch the local research dashboard:
```powershell
python scripts/run_dashboard.py
```

### Navigating the UI for 50-Epoch TPU Models:
1. **Tab 1: Visual Rollout & Metrics**:
   - In the sidebar under **Available Experiments**, select `kth_qpredrnn_tpu_50ep`.
   - Inspect the checkpoint metadata: verify that it displays `Completed 50ep (50 ep, 25,000 steps)`.
   - Select arbitrary KTH test takes (e.g. `person17_walking`, `person21_running`) and inspect autoregressive prediction strips, error heatmaps, and per-horizon degradation curves.
2. **Tab 2: Side-by-Side Model Architecture Comparison**:
   - Select the preset **TPU v5e-1 50-Epoch Benchmark (Final)**.
   - Click **Run Multi-Model Comparative Evaluation**.
   - Compare side-by-side prediction strips across ConvLSTM, PredRNN-V2, and QPredRNN alongside the actual Ground Truth future.
   - Inspect the overlay degradation charts for MSE and SSIM across horizons $t=1\dots 10$.
3. **Tab 3: Training History & Curves**:
   - Select all three TPU runs to plot training loss, validation MSE, and scheduled sampling curves side-by-side.

---

## 7. Reproducibility & Git Safety Rules

- **Do Not Commit Raw Datasets or Checkpoints**: All `.pt` checkpoint files and processed KTH datasets are ignored by [`.gitignore`](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/.gitignore). Keep checkpoints stored locally in `runs/` and backed up on Google Drive or external storage.
- **Commit Only Lightweight Evaluation Artifacts**:
  - `docs/tpu_test_evaluation.json`
  - Any accompanying summary reports or analysis notes.
- **Repository Visibility**: Keep the repository private.
