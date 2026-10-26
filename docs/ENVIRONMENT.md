# Environment Specification & Verification

**Host System**: Windows 10/11 (Build 10.0.26300 64-bit AMD64)
**Verification Date**: October 26, 2026
**Primary Accelerator**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 11.8 / Compute Capability 8.6)

This document records the exact runtime environment verified for the **QPredRNN** research codebase and provides instructions for setting up an identical environment on another computer.

---

## 1. Verified System Stack

| Component | Verified Local Version | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11 (x86_64) | Tested on PowerShell & CMD |
| **Python** | `3.11.0` (64-bit) | Compatible with Python `>=3.10, <3.12` |
| **PyTorch** | `2.7.1+cu118` | CUDA acceleration enabled (`torch.cuda.is_available() == True`) |
| **CUDA Runtime** | `11.8` | Supported on RTX 30-series / Ampere architectures |
| **GPU Device** | NVIDIA GeForce RTX 3050 6GB | VRAM footprint: ~97 MB per batch (batch size 1) |
| **PennyLane** | `0.45.1` | Quantum framework for VQC layer |
| **Quantum Backend** | `default.qubit` | CPU statevector simulator via analytical Autograd/PyTorch tape |
| **Streamlit** | `1.56.0` | Interactive research dashboard |
| **Matplotlib** | `3.9.0` | Metrics plotting (`Agg` headless backend on Windows) |
| **scikit-image** | `0.26.0` | SSIM calculation |
| **ImageIO** | `2.38.1` | GIF/video strip exporting |
| **PyYAML** | `6.0.2` | Configuration file parsing |
| **Pillow** | `>=10.0` | Image processing |
| **pytest** | `>=8.0` | Unit & integration test suite |

---

## 2. Windows Setup Guide for Collaborators

Follow these exact steps to configure an isolated Python virtual environment on another Windows machine.

### Step 1: Install Python 3.11
Ensure Python 3.11 (64-bit) is installed from [python.org](https://www.python.org/downloads/) or Windows winget:
```powershell
winget install Python.Python.3.11
```
Verify the installation:
```powershell
python --version
```
*(Ensure output indicates `Python 3.11.x`).*

### Step 2: Create a Dedicated Virtual Environment
In the repository root (`QPredRNN`), create and activate a clean virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*Note: If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.*

### Step 3: Install PyTorch with CUDA Support
Install PyTorch compatible with your GPU. For NVIDIA GPUs matching our CUDA 11.8 environment:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*(For CPU-only machines, use: `pip install torch torchvision`)*

Verify PyTorch and CUDA detection:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

### Step 4: Install QPredRNN Dependencies
Install the package in editable mode with all optional dependencies:
```powershell
pip install -e ".[all]"
```
Or alternatively install via requirements:
```powershell
pip install -r requirements.txt
pip install -e .
```

### Step 5: Verify the Installation
Run the environment verification script:
```powershell
python -c "
import torch, pennylane, streamlit, matplotlib, skimage, imageio, yaml
print('All core dependencies imported successfully!')
print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'PennyLane: {pennylane.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```
Then run the automated test suite:
```powershell
python -m pytest -q
```
*(All 45 unit and regression tests should pass cleanly).*

---

## 3. Dependency Declaration Matrix

Dependencies are centrally defined in [pyproject.toml](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/pyproject.toml) and mirrored in [requirements.txt](file:///c:/Users/rajee/OneDrive/Desktop/Rijin/QPredRNN/requirements.txt):

- **Core Runtime**: `torch>=2.2`, `numpy>=1.24`, `Pillow>=10.0`, `PyYAML>=6.0`, `scikit-image>=0.22`, `tqdm>=4.66`
- **Quantum Integration**: `pennylane>=0.38`
- **Dashboard & Visualization**: `streamlit>=1.35`, `matplotlib>=3.8`, `imageio>=2.34`
- **Testing & Tooling**: `pytest>=8.0`, `ruff>=0.5`
- **Raw Video Preprocessing (Optional)**: `opencv-python-headless>=4.9` *(Only needed if re-extracting frames from raw `.avi` files)*
