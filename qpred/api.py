r"""QPredRNN FastAPI Backend — qpred/api.py

Clean REST API layer over the existing Python/PyTorch research codebase.
Run with:
  C:\venvs\qpred-gpu\Scripts\python.exe -m uvicorn qpred.api:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).parents[1]
RUNS_DIR = ROOT / "runs"
DOCS_DIR = ROOT / "docs"
DATA_PROCESSED = ROOT / "data" / "processed" / "kth64"
DATA_TOY = ROOT / "data" / "toy"

# --------------------------------------------------------------------------- #
# Canonical Model Registry                                                    #
# --------------------------------------------------------------------------- #
CANONICAL_MODELS = [
    {
        "id": "qpredrnn",
        "name": "QPredRNN",
        "display_name": "QPredRNN — Quantum-Enhanced Spatiotemporal Prediction",
        "family": "Hybrid Quantum-Classical Deep Learning",
        "description": "Combines SpatioTemporal LSTM with a 4-qubit Parameterized Quantum Circuit (PQC) bottleneck modulating memory flow M_t.",
        "parameter_count": 57989,
        "quantum_enabled": True,
        "n_qubits": 4,
        "circuit_depth": 2,
        "simulator_backend": "default.qubit",
        "input_frames": 10,
        "output_frames": 10,
        "implementation_file": "qpred/models/qpredrnn.py",
        "default_run_id": "kth_qpredrnn_local_50ep",
    },
    {
        "id": "predrnn_v2",
        "name": "PredRNN-V2",
        "display_name": "PredRNN-V2 — Spatiotemporal Recurrent Video Prediction",
        "family": "Spatiotemporal Recurrent Neural Network",
        "description": "Dual-memory flow architecture (C_t and M_t) with memory decoupling cosine loss and reverse scheduled sampling.",
        "parameter_count": 51777,
        "quantum_enabled": False,
        "input_frames": 10,
        "output_frames": 10,
        "implementation_file": "qpred/models/predrnn_v2.py",
        "default_run_id": "kth_predrnn_v2_overnight_gpu",
    },
    {
        "id": "convlstm",
        "name": "ConvLSTM",
        "display_name": "ConvLSTM — Classical 2D Recurrent Baseline",
        "family": "Convolutional Long Short-Term Memory",
        "description": "Standard 2-layer ConvLSTM architecture for spatial-temporal sequence extrapolation.",
        "parameter_count": 338401,
        "quantum_enabled": False,
        "input_frames": 10,
        "output_frames": 10,
        "implementation_file": "qpred/models/convlstm.py",
        "default_run_id": "kth_classical_pilot_gpu",
    },
    {
        "id": "persistence",
        "name": "Persistence Baseline",
        "display_name": "Persistence Baseline — Static Last-Frame Repetition",
        "family": "Non-Parametric Baseline",
        "description": "Analytical baseline repeating the final observed context frame X_10 across all future horizons.",
        "parameter_count": 0,
        "quantum_enabled": False,
        "input_frames": 10,
        "output_frames": 10,
        "implementation_file": "qpred/evaluate.py",
        "default_run_id": "baseline_persistence",
    },
    {
        "id": "mean_context",
        "name": "Mean Context Baseline",
        "display_name": "Mean Context Baseline — Context Temporal Average",
        "family": "Non-Parametric Baseline",
        "description": "Analytical baseline repeating the temporal average of context frames across all future horizons.",
        "parameter_count": 0,
        "quantum_enabled": False,
        "input_frames": 10,
        "output_frames": 10,
        "implementation_file": "qpred/evaluate.py",
        "default_run_id": "baseline_mean",
    },
]

RUN_METADATA: dict[str, dict[str, Any]] = {
    "kth_qpredrnn_local_50ep": {
        "display_name": "QPredRNN — 50-Epoch Local Run",
        "model_id": "qpredrnn",
        "category": "primary",
        "status": "Completed (Verified)",
        "description": "50-epoch full local training run with 4-qubit VQC context modulation (best validation at epoch 30)",
    },
    "kth_qpredrnn_pilot_gpu": {
        "display_name": "QPredRNN — 10-Epoch Pilot Run",
        "model_id": "qpredrnn",
        "category": "pilot",
        "status": "Completed (Verified)",
        "description": "10-epoch pilot run with 4-qubit VQC context modulation (best validation at epoch 9)",
    },
    "kth_predrnn_v2_overnight_gpu": {
        "display_name": "PredRNN-V2 — 10-Epoch Overnight Run",
        "model_id": "predrnn_v2",
        "category": "primary",
        "status": "Completed (Verified)",
        "description": "10-epoch baseline with memory decoupling & reverse scheduled sampling",
    },
    "kth_classical_pilot_gpu": {
        "display_name": "ConvLSTM — 10-Epoch Pilot Run",
        "model_id": "convlstm",
        "category": "pilot",
        "status": "Completed (Verified)",
        "description": "10-epoch classical multi-layer 2D ConvLSTM baseline",
    },
    "kth_classical_gpu": {
        "display_name": "ConvLSTM — Early 3-Epoch Run",
        "model_id": "convlstm",
        "category": "experimental",
        "status": "Incomplete (3 epochs)",
        "description": "Early 3-epoch preliminary run",
    },
    "kth_qpredrnn_smoke_gpu": {
        "display_name": "QPredRNN — GPU Smoke Test",
        "model_id": "qpredrnn",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "1-epoch smoke test verifying GPU-PennyLane bridge",
    },
    "kth_qpredrnn_smoke_manual": {
        "display_name": "QPredRNN — Manual Smoke Test",
        "model_id": "qpredrnn",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "1-epoch manual parameter check",
    },
    "kth_qpredrnn_profile_gpu": {
        "display_name": "QPredRNN — Profiling Run",
        "model_id": "qpredrnn",
        "category": "profiling",
        "status": "Profiling Artifact",
        "description": "1-epoch CUDA kernel & latency profiling run",
    },
    "kth_predrnn_v2_smoke": {
        "display_name": "PredRNN-V2 — Smoke Test",
        "model_id": "predrnn_v2",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "1-epoch test fixture for PredRNN-V2",
    },
    "kth_smoke": {
        "display_name": "KTH Smoke Test (CPU)",
        "model_id": "convlstm",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "CPU baseline smoke test",
    },
    "kth_smoke_gpu": {
        "display_name": "KTH Smoke Test (GPU)",
        "model_id": "convlstm",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "GPU baseline smoke test",
    },
    "smoke": {
        "display_name": "Toy Synthetic Smoke Test",
        "model_id": "convlstm",
        "category": "smoke_test",
        "status": "Smoke Test (1 epoch)",
        "description": "Synthetic moving-squares smoke test",
    },
}

# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="QPredRNN Research API",
    version="1.0.0",
    description="Backend for the QPredRNN React research dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# GPU lock & Model Cache                                                      #
# --------------------------------------------------------------------------- #
_gpu_lock = threading.Lock()
_model_cache: dict[str, tuple[Any, dict]] = {}
_model_cache_lock = threading.Lock()
MAX_CACHE = 2


def _load_model_cached(ckpt_path: str, device: str) -> tuple[Any, dict]:
    cache_key = f"{ckpt_path}::{device}"
    with _model_cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        if len(_model_cache) >= MAX_CACHE:
            oldest = next(iter(_model_cache))
            del _model_cache[oldest]
    from qpred.evaluate import load_checkpoint_for_eval
    model, ckpt = load_checkpoint_for_eval(ckpt_path, device=device)
    with _model_cache_lock:
        _model_cache[cache_key] = (model, ckpt)
    return model, ckpt


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _tensor_to_b64(t: torch.Tensor) -> str:
    """Convert (1,H,W) float32 tensor [0,1] to base64 PNG."""
    arr = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _tensor_to_b64_magma(t: torch.Tensor) -> str:
    """Convert (1,H,W) float32 tensor to base64 PNG using magma colormap."""
    import matplotlib.pyplot as plt
    arr = t[0].cpu().numpy().clip(0, 1)
    cm = plt.get_cmap("magma")
    colored = (cm(arr)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _scan_runs() -> list[dict]:
    """Discover all run directories with at least one checkpoint and enriched metadata."""
    if not RUNS_DIR.exists():
        return []
    results = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        best = d / "checkpoint_best.pt"
        last = d / "checkpoint_last.pt"
        history = d / "history.json"
        config_f = d / "config.yaml"
        hist_data = None
        if history.exists():
            try:
                hist_data = json.loads(history.read_text(encoding="utf-8"))
            except Exception:
                pass
        best_val = None
        last_ep = None
        total_ep = None
        if hist_data:
            val_entries = [h for h in hist_data if "val_loss" in h]
            if val_entries:
                best_ep_h = min(val_entries, key=lambda h: h["val_loss"])
                best_val = best_ep_h["val_loss"]
            last_ep = hist_data[-1].get("epoch")
            if hist_data[-1].get("eta") == 0:
                total_ep = last_ep

        meta = RUN_METADATA.get(d.name, {})
        display_name = meta.get("display_name", d.name)
        category = meta.get("category", "experimental")
        status = meta.get("status", "Completed" if (best.exists() and hist_data) else "Available")
        model_type = meta.get("model_id")

        if not model_type:
            if config_f.exists():
                try:
                    import yaml
                    cfg = yaml.safe_load(config_f.read_text(encoding="utf-8"))
                    model_type = cfg.get("model", {}).get("name")
                except Exception:
                    pass
            if not model_type:
                name_l = d.name.lower()
                if "qpredrnn" in name_l or "quantum" in name_l:
                    model_type = "qpredrnn"
                elif "predrnn" in name_l:
                    model_type = "predrnn_v2"
                elif "classical" in name_l or "convlstm" in name_l:
                    model_type = "convlstm"
                else:
                    model_type = "unknown"

        results.append({
            "id": d.name,
            "name": d.name,
            "display_name": display_name,
            "model_type": model_type,
            "category": category,
            "status": status,
            "description": meta.get("description", ""),
            "has_best_ckpt": best.exists(),
            "has_last_ckpt": last.exists(),
            "best_val_mse": best_val,
            "last_epoch": last_ep,
            "total_epochs": total_ep,
            "has_history": history.exists(),
            "history_entries": len(hist_data) if hist_data else 0,
            "path": str(d),
        })
    return results


def _run_info(run_id: str) -> dict:
    if run_id in ("baseline_persistence", "baseline_mean"):
        is_pers = "persistence" in run_id
        return {
            "id": run_id,
            "name": run_id,
            "display_name": "Persistence Baseline" if is_pers else "Mean Context Baseline",
            "model_type": "persistence" if is_pers else "mean_context",
            "category": "baseline",
            "status": "Analytical Baseline",
            "has_best_ckpt": False,
            "has_last_ckpt": False,
            "best_val_mse": 0.002260 if is_pers else 0.001903,
            "history_entries": 0,
            "config": {},
        }

    d = RUNS_DIR / run_id
    if not d.exists():
        raise HTTPException(404, f"Run '{run_id}' not found")
    run_list = _scan_runs()
    match = next((r for r in run_list if r["id"] == run_id), None)
    if match is None:
        raise HTTPException(404, f"Run '{run_id}' not scannable")
    config_f = d / "config.yaml"
    cfg = {}
    if config_f.exists():
        try:
            import yaml
            cfg = yaml.safe_load(config_f.read_text(encoding="utf-8"))
        except Exception:
            pass
    match["config"] = cfg
    return match


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    run_id: str
    checkpoint: str = "best"   # "best" | "last"
    dataset: str = "kth"       # "kth" | "toy"
    split: str = "test"
    clip_index: int = 0
    device: str = "cuda"

    @field_validator("checkpoint")
    @classmethod
    def val_ckpt(cls, v: str) -> str:
        if v not in ("best", "last"):
            raise ValueError("checkpoint must be 'best' or 'last'")
        return v

    @field_validator("dataset")
    @classmethod
    def val_ds(cls, v: str) -> str:
        if v not in ("kth", "toy"):
            raise ValueError("dataset must be 'kth' or 'toy'")
        return v

    @field_validator("split")
    @classmethod
    def val_split(cls, v: str) -> str:
        if v not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        return v

    @field_validator("device")
    @classmethod
    def val_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        return v


class CompareItem(BaseModel):
    run_id: str
    checkpoint: str = "best"
    display_name: Optional[str] = None


class CompareRequest(BaseModel):
    models: list[CompareItem]
    dataset: str = "kth"
    split: str = "test"
    clip_index: int = 0
    device: str = "cuda"


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/system")
def system_info() -> dict:
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    vram_total = None
    vram_used = None
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory // (1024 ** 2)
        vram_used = torch.cuda.memory_allocated(0) // (1024 ** 2)
    return {
        "python": __import__("sys").version,
        "pytorch": torch.__version__,
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total,
        "vram_used_mb": vram_used,
        "cached_models": len(_model_cache),
        "kth_available": DATA_PROCESSED.exists(),
        "toy_available": DATA_TOY.exists(),
    }


@app.get("/api/models")
def list_canonical_models() -> list[dict]:
    """Returns the canonical catalog of model architectures."""
    return CANONICAL_MODELS


@app.get("/api/experiments")
def list_canonical_experiments() -> list[dict]:
    """Returns all discovered experiments categorized by run type."""
    return _scan_runs()


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return _scan_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _run_info(run_id)


@app.get("/api/runs/{run_id}/history")
def get_history(run_id: str) -> list[dict]:
    h = RUNS_DIR / run_id / "history.json"
    if not h.exists():
        return []
    try:
        return json.loads(h.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read history: {e}")


@app.get("/api/checkpoints")
def list_checkpoints() -> list[dict]:
    """Returns full 24-checkpoint audited inventory with SHA-256 fingerprints."""
    inv_file = DOCS_DIR / "checkpoint_inventory.json"
    if inv_file.exists():
        try:
            return json.loads(inv_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


@app.get("/api/benchmarks")
def get_benchmarks() -> list[dict]:
    """Returns verified held-out test evaluation benchmark results."""
    bench_file = DOCS_DIR / "verified_evaluation_results.json"
    if bench_file.exists():
        try:
            return json.loads(bench_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    datasets = []
    if DATA_PROCESSED.exists():
        datasets.append({"id": "kth", "name": "KTH-64", "path": str(DATA_PROCESSED), "available": True})
    else:
        datasets.append({"id": "kth", "name": "KTH-64", "available": False,
                          "note": "Run scripts/preprocess_videos.py to generate."})
    if DATA_TOY.exists():
        datasets.append({"id": "toy", "name": "Toy Dataset", "path": str(DATA_TOY), "available": True})
    else:
        datasets.append({"id": "toy", "name": "Toy Dataset", "available": False})
    return datasets


@app.get("/api/datasets/{dataset_id}/splits/{split}/info")
def dataset_split_info(
    dataset_id: str,
    split: str,
    input_frames: int = Query(default=10, ge=1),
    output_frames: int = Query(default=10, ge=1),
    stride: int = Query(default=2, ge=1),
) -> dict:
    if dataset_id == "kth":
        root = DATA_PROCESSED
    elif dataset_id == "toy":
        root = DATA_TOY
    else:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'")

    if not root.exists():
        raise HTTPException(404, f"Dataset '{dataset_id}' not found on disk")
    if split not in ("train", "val", "test"):
        raise HTTPException(400, "split must be train/val/test")

    try:
        from qpred.data.dataset import FrameSequenceDataset
        ds = FrameSequenceDataset(root, split=split, input_frames=input_frames,
                                  output_frames=output_frames, stride=stride)
    except Exception as e:
        raise HTTPException(500, str(e))

    seq_ids = sorted({item[0].name for item in ds.items})
    return {
        "dataset": dataset_id,
        "split": split,
        "num_clips": len(ds),
        "input_frames": input_frames,
        "output_frames": output_frames,
        "stride": stride,
        "num_sequences": len(seq_ids),
        "sequence_ids": seq_ids[:20],
    }


def _run_single_inference(
    run_id: str,
    checkpoint: str,
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    """Helper to run single inference with error containment."""
    # 1. Non-parametric Baselines
    if run_id in ("baseline_persistence", "persistence"):
        pred_seq = seq_x[-1:].repeat(10, 1, 1, 1)
        latency_ms = 0.5
        ckpt_file = "N/A (Analytical)"
        saved_ep = None
        best_vl = 0.002260
    elif run_id in ("baseline_mean", "mean_context"):
        pred_seq = seq_x.mean(dim=0, keepdim=True).repeat(10, 1, 1, 1)
        latency_ms = 0.5
        ckpt_file = "N/A (Analytical)"
        saved_ep = None
        best_vl = 0.001903
    else:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory '{run_id}' not found")
        ckpt_file = f"checkpoint_{checkpoint}.pt"
        ckpt_path = run_dir / ckpt_file
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint '{ckpt_file}' not found in {run_id}")

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model, ckpt_meta = _load_model_cached(str(ckpt_path), device)
        batch_in = seq_x.unsqueeze(0).to(device)
        with _gpu_lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred_out = model(batch_in)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_seq = pred_out[0].cpu()
        saved_ep = ckpt_meta.get("epoch", None)
        best_vl = ckpt_meta.get("best_val_loss", None)

    from qpred.metrics import compute_sequence_metrics
    metrics = compute_sequence_metrics(pred_seq, seq_y)
    diff = torch.abs(seq_y - pred_seq)

    pred_frames = [_tensor_to_b64(pred_seq[i]) for i in range(pred_seq.shape[0])]
    err_frames = [_tensor_to_b64_magma(diff[i]) for i in range(diff.shape[0])]

    return {
        "checkpoint_path": ckpt_file,
        "checkpoint_epoch": saved_ep,
        "checkpoint_best_val_loss": best_vl,
        "device_used": device,
        "latency_ms": round(latency_ms, 2),
        "pred_frames": pred_frames,
        "error_frames": err_frames,
        "metrics": {
            "mse": round(float(metrics["mse"]), 8),
            "mae": round(float(metrics["mae"]), 8),
            "psnr": round(float(metrics["psnr"]), 4),
            "ssim": round(float(metrics["ssim"]), 6),
            "per_frame": {
                "mse": [round(v, 8) for v in metrics["per_frame"]["mse"]],
                "mae": [round(v, 8) for v in metrics["per_frame"]["mae"]],
                "psnr": [round(v, 4) for v in metrics["per_frame"]["psnr"]],
                "ssim": [round(v, 6) for v in metrics["per_frame"]["ssim"]],
            },
        },
        "tensor_info": {
            "pred_shape": list(pred_seq.shape),
            "pred_std": round(float(pred_seq.std()), 6),
            "pred_range": [round(float(pred_seq.min()), 4), round(float(pred_seq.max()), 4)],
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    """Run inference for one temporal clip and return all frames + metrics."""
    root = DATA_PROCESSED if req.dataset == "kth" else DATA_TOY
    if not root.exists():
        raise HTTPException(404, f"Dataset '{req.dataset}' not on disk")

    try:
        from qpred.data.dataset import FrameSequenceDataset
        ds = FrameSequenceDataset(root, split=req.split, input_frames=10,
                                  output_frames=10, stride=2)
    except Exception as e:
        raise HTTPException(500, f"Dataset error: {e}")

    n = len(ds)
    if not (0 <= req.clip_index < n):
        raise HTTPException(400, f"clip_index {req.clip_index} out of range [0, {n-1}]")

    seq_x, seq_y = ds[req.clip_index]  # (10,1,64,64) each
    seq_dir, start_f, _ = ds.items[req.clip_index]
    seq_desc = f"{req.dataset}/{req.split} clip #{req.clip_index} · {seq_dir.name} · frames {start_f}–{start_f+19}"

    try:
        inf_res = _run_single_inference(
            run_id=req.run_id,
            checkpoint=req.checkpoint,
            seq_x=seq_x,
            seq_y=seq_y,
            device=req.device,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise HTTPException(507, "CUDA out of memory — try device=cpu")
        raise HTTPException(500, f"Inference error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    ctx_frames = [_tensor_to_b64(seq_x[i]) for i in range(seq_x.shape[0])]
    gt_frames = [_tensor_to_b64(seq_y[i]) for i in range(seq_y.shape[0])]

    return {
        "run_id": req.run_id,
        "checkpoint": req.checkpoint,
        "dataset": req.dataset,
        "split": req.split,
        "clip_index": req.clip_index,
        "seq_desc": seq_desc,
        "input_frames": ctx_frames,
        "gt_frames": gt_frames,
        **inf_res,
    }


@app.post("/api/compare")
def compare_models(req: CompareRequest) -> dict:
    """Run fair multi-model comparison on the exact same temporal clip."""
    root = DATA_PROCESSED if req.dataset == "kth" else DATA_TOY
    if not root.exists():
        raise HTTPException(404, f"Dataset '{req.dataset}' not on disk")

    try:
        from qpred.data.dataset import FrameSequenceDataset
        ds = FrameSequenceDataset(root, split=req.split, input_frames=10,
                                  output_frames=10, stride=2)
    except Exception as e:
        raise HTTPException(500, f"Dataset error: {e}")

    n = len(ds)
    if not (0 <= req.clip_index < n):
        raise HTTPException(400, f"clip_index {req.clip_index} out of range [0, {n-1}]")

    seq_x, seq_y = ds[req.clip_index]
    seq_dir, start_f, _ = ds.items[req.clip_index]
    seq_desc = f"{req.dataset}/{req.split} clip #{req.clip_index} · {seq_dir.name} · frames {start_f}–{start_f+19}"

    ctx_frames = [_tensor_to_b64(seq_x[i]) for i in range(seq_x.shape[0])]
    gt_frames = [_tensor_to_b64(seq_y[i]) for i in range(seq_y.shape[0])]

    results = []
    for item in req.models:
        meta = RUN_METADATA.get(item.run_id, {})
        disp_name = item.display_name or meta.get("display_name", item.run_id)

        try:
            inf = _run_single_inference(
                run_id=item.run_id,
                checkpoint=item.checkpoint,
                seq_x=seq_x,
                seq_y=seq_y,
                device=req.device,
            )
            results.append({
                "run_id": item.run_id,
                "display_name": disp_name,
                "checkpoint": item.checkpoint,
                "success": True,
                **inf,
            })
        except Exception as e:
            results.append({
                "run_id": item.run_id,
                "display_name": disp_name,
                "checkpoint": item.checkpoint,
                "success": False,
                "error": str(e),
            })

    return {
        "dataset": req.dataset,
        "split": req.split,
        "clip_index": req.clip_index,
        "seq_desc": seq_desc,
        "input_frames": ctx_frames,
        "gt_frames": gt_frames,
        "models": results,
    }


@app.get("/api/metrics")
def available_metrics() -> dict:
    return {
        "metrics": ["mse", "mae", "psnr", "ssim"],
        "per_horizon": True,
        "ssim_available": __import__("qpred.metrics", fromlist=["HAS_SKIMAGE"]).HAS_SKIMAGE,
        "normalization": "pixel values in [0,1] float32",
    }


# --------------------------------------------------------------------------- #
# Static files / SPA catch-all (for built React frontend)                     #
# --------------------------------------------------------------------------- #
DIST_DIR = ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "API endpoint not found")
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
