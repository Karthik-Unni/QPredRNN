# Collaborator Onboarding & Development Guide

**Repository**: `https://github.com/Karthik-Unni/QPredRNN.git`
**Collaboration Branch**: `collab/project-completion`
**Date**: October 26, 2026

Welcome to the **QPredRNN** research project. This guide outlines how to configure your local development environment, restore the preprocessed KTH dataset, launch the research dashboard, and contribute code safely in parallel with ongoing GPU training runs.

---

## 1. Clone the Collaboration Branch

Clone the repository and check out the dedicated collaboration branch:

```powershell
# [TESTED] Clone repository
git clone https://github.com/Karthik-Unni/QPredRNN.git
cd QPredRNN

# [TESTED] Check out the collaboration branch
git checkout collab/project-completion
```

---

## 2. Virtual Environment Setup

Do **not** copy Python environments between machines. Always create an isolated virtual environment:

```powershell
# [TESTED] Create isolated Python 3.11 virtual environment
python -m venv .venv

# [TESTED] Activate virtual environment on Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

---

## 3. Dependency Installation

### A. Install PyTorch (GPU or CPU)
If your machine has a compatible NVIDIA GPU (recommended):
```powershell
# [TESTED with CUDA 11.8 on RTX 30-series]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
For CPU-only development / testing:
```powershell
# [STANDARD] CPU-only PyTorch
pip install torch torchvision
```

### B. Install QPredRNN & Research Dependencies
Install the package in editable mode with all optional dependencies:
```powershell
# [TESTED] Install QPredRNN with quantum and dashboard extras
pip install -e ".[all]"
```
*(Or alternatively: `pip install -r requirements.txt` followed by `pip install -e .`)*

---

## 4. Dataset Placement (`QPredRNN_data.zip`)

Due to GitHub size limits, preprocessed video sequences are shared via Google Drive in `QPredRNN_data.zip`.

### Expected Directory Layout:
```
QPredRNN/
├── data/
│   ├── manifest.json
│   ├── toy/
│   └── processed/
│       └── kth64/
│           ├── splits.json
│           └── sequences/
│               ├── person01_boxing_d1_take01/
│               └── ... (2,391 annotated sequence takes)
```

### Extraction Steps:
1. Download `QPredRNN_data.zip` from the shared Google Drive folder.
2. Place `QPredRNN_data.zip` in your repository root (`QPredRNN/`).
3. Extract the archive directly into the repository root:
   ```powershell
   # [TESTED] Extract data archive on Windows PowerShell
   Expand-Archive -Path .\QPredRNN_data.zip -DestinationPath . -Force
   ```
4. Verify the extraction:
   ```powershell
   # [TESTED] Verify critical dataset paths exist
   Test-Path data\manifest.json
   Test-Path data\processed\kth64\splits.json
   ```
   *(Both commands must return `True`).*
5. Delete the temporary zip file:
   ```powershell
   Remove-Item .\QPredRNN_data.zip
   ```
   *(Note: `.gitignore` automatically excludes `*.zip` and `data/` to protect against accidental commits).*

---

## 5. Environment & Test Suite Verification

Run a quick automated check to confirm that PyTorch, PennyLane, and all dataset splits load correctly:

```powershell
# [TESTED] Run complete regression test suite
python -m pytest -q
```
**Expected Result**: All 45 unit and integration tests should pass in ~15 seconds.

---

## 6. Launching the Research Dashboard

The interactive Streamlit research dashboard allows you to inspect model architectures, load checkpoints, compare predictions side-by-side, view degradation curves across horizons $1\dots 10$, and export experimental reports.

Start the dashboard locally:
```powershell
# [TESTED] Launch research dashboard
python scripts/run_dashboard.py
```
Or directly via Streamlit:
```powershell
# [TESTED]
streamlit run qpred/dashboard.py
```
Access the dashboard at `http://localhost:8501`.

---

## 7. Parallel Development Workflow & Git Best Practices

To ensure smooth collaboration while model training runs locally:

### A. Create a Feature Branch
Always branch off `collab/project-completion`:
```powershell
# [STANDARD] Create your personal feature branch
git checkout collab/project-completion
git pull origin collab/project-completion
git checkout -b feature/your-feature-name
```

### B. Stage Specific Files Only
**Never use `git add .` or `git add -A`.** Stage only the specific source files you modified:
```powershell
# [STANDARD] Stage explicit files
git add qpred/models/your_module.py
git add tests/test_your_feature.py
```

### C. Commit, Push, and Open a PR
```powershell
# [STANDARD] Commit with descriptive message
git commit -m "feat(analysis): add metric visualization helper"

# [STANDARD] Push feature branch to GitHub
git push -u origin feature/your-feature-name
```
Then open a Pull Request targeting `collab/project-completion` on GitHub for review.

---

## 8. Protected Artifacts & Concurrency Rules

To prevent data corruption, race conditions, and merge conflicts during parallel research:

> [!CAUTION]
> **DO NOT OVERWRITE OR COMMIT THE FOLLOWING ARTIFACTS:**
> 1. **Active Checkpoints**: `runs/kth_qpredrnn_pilot_gpu/` is actively managed by ongoing GPU training. Never overwrite, move, or delete files in this folder.
> 2. **Baseline Checkpoints**: `runs/kth_classical_gpu/` and `runs/kth_predrnn_v2_overnight_gpu/` contain benchmark checkpoints. Keep them read-only.
> 3. **Dataset Files**: Never stage or commit anything inside `data/` or any `*.zip` files.
> 4. **Virtual Environment**: Never commit `.venv/`, `venv/`, or `.env`.
> 5. **Git Branches**: Never force-push (`--force`) to `main` or `collab/project-completion`.
