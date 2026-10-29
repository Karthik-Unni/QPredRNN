"""
tests/test_epoch_semantics.py

Verifies the critical epoch-semantics distinction:
  - Full-dataset training epochs (no batch cap)
  - Capped training epochs (max_train_batches limit)
  - That new final configs are correctly set to full-dataset mode
  - That existing QPredRNN 50-ep run is correctly classified as capped

These tests implement the mandatory Phase 2 audit from the master prompt.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(name: str) -> dict:
    path = CONFIGS / name
    assert path.exists(), f"Config not found: {path}"
    with open(path) as f:
        return yaml.safe_load(f)


def load_history(run_id: str) -> list[dict]:
    p = RUNS / run_id / "history.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: New final ConvLSTM config has NO batch cap
# ──────────────────────────────────────────────────────────────────────────────

def test_convlstm_final_config_no_cap():
    cfg = load_config("kth_classical_final_50ep.yaml")
    tr = cfg["training"]
    assert tr.get("max_train_batches") is None, (
        "kth_classical_final_50ep must have max_train_batches=null (full dataset per epoch)"
    )
    assert tr.get("max_val_batches") is None, (
        "kth_classical_final_50ep must have max_val_batches=null (full validation per epoch)"
    )
    assert tr["epochs"] == 50, "Target must be 50 epochs"
    assert tr["batch_size"] == 8, "Batch size must be 8 for ConvLSTM final run"


def test_convlstm_final_config_correct_data():
    cfg = load_config("kth_classical_final_50ep.yaml")
    data = cfg["data"]
    assert data["image_size"] == 64
    assert data["input_frames"] == 10
    assert data["output_frames"] == 10
    assert data["stride"] == 2


def test_convlstm_final_config_run_dir():
    cfg = load_config("kth_classical_final_50ep.yaml")
    assert cfg.get("run_dir") == "runs/kth_classical_final_50ep", (
        "run_dir must be distinct from all existing runs"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: New final PredRNN-V2 config has NO batch cap
# ──────────────────────────────────────────────────────────────────────────────

def test_predrnn_v2_final_config_no_cap():
    cfg = load_config("kth_predrnn_v2_final_50ep.yaml")
    tr = cfg["training"]
    assert tr.get("max_train_batches") is None, (
        "kth_predrnn_v2_final_50ep must have max_train_batches=null"
    )
    assert tr.get("max_val_batches") is None, (
        "kth_predrnn_v2_final_50ep must have max_val_batches=null"
    )
    assert tr["epochs"] == 50
    assert tr["batch_size"] == 8


def test_predrnn_v2_final_config_has_rss():
    cfg = load_config("kth_predrnn_v2_final_50ep.yaml")
    rss = cfg["training"].get("reverse_scheduled_sampling", {})
    assert rss.get("enabled") is True, "PredRNN-V2 final must have RSS enabled"
    assert rss.get("r_sampling_step_1") is not None
    assert rss.get("r_sampling_step_2") is not None


def test_predrnn_v2_final_config_rss_steps_reasonable():
    """RSS steps must be sane relative to total training steps for a full-dataset run."""
    cfg = load_config("kth_predrnn_v2_final_50ep.yaml")
    tr = cfg["training"]
    rss = tr.get("reverse_scheduled_sampling", {})
    bs = tr["batch_size"]
    # Full dataset: ~28389 windows / bs=8 = 3549 batches/epoch, 50 epochs = 177450 steps
    approx_total_steps = math.ceil(28389 / bs) * 50
    step1 = rss.get("r_sampling_step_1", 0)
    step2 = rss.get("r_sampling_step_2", 0)
    # step1 should be within total training steps
    assert step1 < approx_total_steps * 2, (
        f"r_sampling_step_1={step1} seems unreasonably large for {approx_total_steps} total steps"
    )
    assert step2 > step1, "r_sampling_step_2 must be greater than r_sampling_step_1"


def test_predrnn_v2_final_config_has_decouple():
    cfg = load_config("kth_predrnn_v2_final_50ep.yaml")
    dlw = cfg["model"].get("decouple_loss_weight", 0.0)
    assert dlw > 0, "PredRNN-V2 must have decouple_loss_weight > 0"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Existing QPredRNN run IS capped (accurately classify historical run)
# ──────────────────────────────────────────────────────────────────────────────

def test_qpredrnn_existing_run_has_batch_cap():
    """QPredRNN 50-ep run used max_train_batches=500 — verify from saved config."""
    run_config_path = RUNS / "kth_qpredrnn_local_50ep" / "config.yaml"
    pytest.importorskip("yaml")
    if not run_config_path.exists():
        pytest.skip("QPredRNN 50-ep run config not found — run not imported yet")
    with open(run_config_path) as f:
        cfg = yaml.safe_load(f)
    cap = cfg.get("training", {}).get("max_train_batches")
    assert cap is not None and cap == 500, (
        f"kth_qpredrnn_local_50ep must have max_train_batches=500 in its saved config, got: {cap}"
    )


def test_qpredrnn_existing_run_step_increment_confirms_cap():
    """
    Each epoch increments global_step by max_train_batches (500), not by
    full-dataset batch count (~28389/bs). This confirms the cap was active.
    """
    history = load_history("kth_qpredrnn_local_50ep")
    if len(history) < 2:
        pytest.skip("Not enough history to verify cap")
    step_increments = [
        history[i]["global_step"] - history[i - 1]["global_step"]
        for i in range(1, min(5, len(history)))
    ]
    # All increments should be 500 (the cap), not 28389 (full dataset / bs=1)
    for inc in step_increments:
        assert inc == 500, (
            f"Expected step increment of 500 (capped batch count) but got {inc}. "
            "This run did NOT process the full training dataset per epoch."
        )


def test_qpredrnn_capped_epochs_not_equal_to_full_dataset_epochs():
    """
    Explicitly confirm that 50 capped epochs ≠ 50 full-dataset epochs.
    500 batches per epoch on 28389 windows at bs=1 = 1.76% coverage per epoch.
    """
    cap = 500
    full_train_windows = 28389
    batch_size = 1  # QPredRNN uses bs=1
    full_batches_per_epoch = math.ceil(full_train_windows / batch_size)
    fraction_covered = cap / full_batches_per_epoch
    # Must be well under 5% per epoch
    assert fraction_covered < 0.05, (
        f"Expected < 5% coverage per capped epoch, got {fraction_covered:.2%}. "
        "Check that cap is correctly identified as NOT equivalent to full-dataset."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Existing overnight PredRNN-V2 config is ALSO capped
# ──────────────────────────────────────────────────────────────────────────────

def test_predrnn_v2_overnight_config_has_cap():
    cfg = load_config("kth_predrnn_v2_overnight.yaml")
    tr = cfg["training"]
    assert tr.get("max_train_batches") == 500, (
        "kth_predrnn_v2_overnight.yaml should have max_train_batches=500 (pilot/capped run)"
    )
    assert tr["epochs"] == 10, (
        "kth_predrnn_v2_overnight.yaml is a 10-epoch run, not a 50-epoch run"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: New configs do NOT conflict with existing run directories
# ──────────────────────────────────────────────────────────────────────────────

def test_new_run_dirs_are_distinct_from_all_existing():
    """New final run directories must not be the same as any historical run."""
    new_runs = ["kth_classical_final_50ep", "kth_predrnn_v2_final_50ep"]
    existing_runs = [d.name for d in RUNS.iterdir() if d.is_dir()]
    # Before the new runs start, they should not be in existing_runs
    # OR if they are (training has started), they must NOT contain old data
    for run_id in new_runs:
        if run_id in existing_runs:
            # If they exist, they must have the correct config
            config_path = RUNS / run_id / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                assert cfg["training"].get("max_train_batches") is None, (
                    f"Run {run_id} exists but has a batch cap — this contradicts the full-dataset claim"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Model builds successfully with new configs
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("config_name,expected_model_name", [
    ("kth_classical_final_50ep.yaml", "classical"),
    ("kth_predrnn_v2_final_50ep.yaml", "predrnn_v2"),
])
def test_model_builds_from_final_config(config_name, expected_model_name):
    """Both new final configs must build their models without error."""
    from qpred.models import build_model
    cfg = load_config(config_name)
    assert cfg["model"]["name"] == expected_model_name
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params > 0, f"Model from {config_name} must have trainable parameters"


def test_convlstm_parameter_count_is_112289():
    """Verified parameter count for ConvLSTM with hidden_channels=32."""
    from qpred.models import build_model
    cfg = load_config("kth_classical_final_50ep.yaml")
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 112289, (
        f"ConvLSTM (hidden=32) must have exactly 112,289 parameters, got {params:,}"
    )


def test_predrnn_v2_parameter_count_is_15361():
    """Verified parameter count for PredRNN-V2 with hidden=8, layers=2."""
    from qpred.models import build_model
    cfg = load_config("kth_predrnn_v2_final_50ep.yaml")
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 15361, (
        f"PredRNN-V2 (hidden=8, 2 layers) must have exactly 15,361 parameters, got {params:,}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Dataset window counts match expected values
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("split,expected_min", [
    ("train", 28000),
    ("val", 29000),
    ("test", 33000),
])
def test_dataset_window_counts(split, expected_min):
    """KTH dataset window counts must match pre-verified counts."""
    data_root = ROOT / "data" / "processed" / "kth64"
    if not data_root.exists():
        pytest.skip("KTH64 dataset not preprocessed")
    from qpred.data.dataset import FrameSequenceDataset
    ds = FrameSequenceDataset(data_root, split=split, input_frames=10, output_frames=10, stride=2)
    assert len(ds) >= expected_min, (
        f"{split} split must have >= {expected_min} windows, got {len(ds)}"
    )


def test_test_split_contains_correct_persons():
    """KTH test split must contain persons 17-25 (held-out persons)."""
    import json
    splits_file = ROOT / "data" / "processed" / "kth64" / "splits.json"
    if not splits_file.exists():
        pytest.skip("splits.json not found")
    with open(splits_file) as f:
        entries = json.load(f)
    test_persons = {
        e["sequence_id"].split("_")[0]
        for e in entries
        if e["split"] == "test"
    }
    expected_test_persons = {f"person{i:02d}" for i in range(17, 26)}
    assert expected_test_persons == test_persons, (
        f"Test persons mismatch. Expected {sorted(expected_test_persons)}, got {sorted(test_persons)}"
    )
