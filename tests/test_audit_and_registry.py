"""Unit and integration tests for Model Registry, Checkpoint Inventory, and Quantum Execution.

Covers:
- Canonical model registry schema and factory instantiation.
- 24-checkpoint audited inventory integrity.
- Verified test benchmark metrics and per-horizon degradation arrays.
- QPredRNN quantum forward path execution and output finiteness.
- Training history parsing and 50-epoch completion verification.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from qpred.models import build_model, VideoConvLSTM, PredRNNV2, QPredRNN
from qpred.api import CANONICAL_MODELS, RUN_METADATA, _scan_runs

ROOT = Path(__file__).parents[1].resolve()
DOCS_DIR = ROOT / "docs"


def test_canonical_models_registry():
    assert len(CANONICAL_MODELS) == 5
    ids = [m["id"] for m in CANONICAL_MODELS]
    assert "qpredrnn" in ids
    assert "predrnn_v2" in ids
    assert "convlstm" in ids
    assert "persistence" in ids
    assert "mean_context" in ids


def test_build_model_factory():
    # 1. Classical ConvLSTM
    m_conv = build_model({"model": {"name": "classical", "input_frames": 10, "output_frames": 10}})
    assert isinstance(m_conv, VideoConvLSTM)
    out_conv = m_conv(torch.rand(1, 10, 1, 64, 64))
    assert list(out_conv.shape) == [1, 10, 1, 64, 64]
    assert torch.isfinite(out_conv).all()

    # 2. PredRNN-V2
    m_pred = build_model({"model": {"name": "predrnn_v2", "input_frames": 10, "output_frames": 10}})
    assert isinstance(m_pred, PredRNNV2)
    out_pred = m_pred(torch.rand(1, 10, 1, 64, 64))
    assert list(out_pred.shape) == [1, 10, 1, 64, 64]
    assert torch.isfinite(out_pred).all()

    # 3. QPredRNN
    m_qpred = build_model({"model": {"name": "qpredrnn", "input_frames": 10, "output_frames": 10}})
    assert isinstance(m_qpred, QPredRNN)
    assert m_qpred.quantum_enabled is True
    out_qpred = m_qpred(torch.rand(1, 10, 1, 64, 64))
    assert list(out_qpred.shape) == [1, 10, 1, 64, 64]
    assert torch.isfinite(out_qpred).all()


def test_qpredrnn_quantum_path_active():
    """Verify that the quantum modulator actively modifies the recurrent memory state."""
    model = QPredRNN(
        image_channels=1,
        hidden_channels=16,
        input_frames=5,
        output_frames=5,
        num_layers=1,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    model.eval()

    # Test QuantumContextModulator directly
    m_in = torch.randn(1, 16, 32, 32)
    m_out = model.quantum_modulator(m_in)

    # Quantum modulation must alter the memory state
    assert not torch.allclose(m_in, m_out)
    assert m_out.shape == m_in.shape
    assert torch.isfinite(m_out).all()


def test_checkpoint_inventory_artifact():
    inv_path = DOCS_DIR / "checkpoint_inventory.json"
    assert inv_path.exists(), "checkpoint_inventory.json missing"
    data = json.loads(inv_path.read_text(encoding="utf-8"))

    # Exactly 24 checkpoints
    assert len(data) == 24

    # Every checkpoint must load and forward-pass cleanly
    for item in data:
        assert item["load_ok"] is True, f"Failed load: {item['file_path']}"
        assert item["forward_ok"] is True, f"Failed forward: {item['file_path']}"
        assert len(item["sha256"]) == 64, f"Invalid SHA-256 for {item['file_path']}"
        assert item["file_size_bytes"] > 100000, f"Suspiciously small file size: {item['file_path']}"

    # Verify primary 50-epoch checkpoint
    primary_ckpt = next((c for c in data if c["run_id"] == "kth_qpredrnn_local_50ep" and c["is_best"]), None)
    assert primary_ckpt is not None
    assert primary_ckpt["recorded_epoch"] == 30
    assert abs(primary_ckpt["best_val_loss"] - 0.001136286) < 1e-6
    assert primary_ckpt["classification"] == "Valid Final Candidate"


def test_verified_benchmarks_artifact():
    bench_path = DOCS_DIR / "verified_evaluation_results.json"
    assert bench_path.exists(), "verified_evaluation_results.json missing"
    data = json.loads(bench_path.read_text(encoding="utf-8"))

    assert len(data) >= 5
    models_evaluated = [d["model_id"] for d in data]
    assert "qpredrnn_50ep" in models_evaluated
    assert "predrnn_v2_main" in models_evaluated
    assert "convlstm_pilot" in models_evaluated
    assert "persistence" in models_evaluated

    for b in data:
        assert b["status"] == "Verified Complete"
        assert b["evaluated_windows"] == 100
        assert "mse" in b["metrics"]
        assert "ssim" in b["metrics"]
        assert len(b["per_horizon"]["mse"]) == 10
        assert len(b["per_horizon"]["ssim"]) == 10


def test_50ep_run_history_complete():
    hist_path = ROOT / "runs" / "kth_qpredrnn_local_50ep" / "history.json"
    assert hist_path.exists()
    hist = json.loads(hist_path.read_text(encoding="utf-8"))

    # Exactly 50 epochs logged
    assert len(hist) == 50
    assert hist[0]["epoch"] == 1
    assert hist[-1]["epoch"] == 50
    assert hist[-1]["global_step"] == 25000

    # Best validation occurred at epoch 30
    best_entry = min(hist, key=lambda h: h["val_loss"])
    assert best_entry["epoch"] == 30
    assert abs(best_entry["val_loss"] - 0.001136286) < 1e-6
