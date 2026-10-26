"""Unit tests for scripts/import_colab_results.py."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import pytest
import torch
import yaml

from scripts.import_colab_results import (
    compute_sha256,
    inspect_checkpoint,
    verify_run_consistency,
    import_colab_results,
    EXPECTED_RUN_DIRS,
)


@pytest.fixture
def mock_colab_run(tmp_path: Path):
    """Create a mock completed run directory mimicking a 5-epoch test run."""
    run_dir = tmp_path / "kth_classical_tpu_50ep"
    run_dir.mkdir(parents=True)

    target_epochs = 5
    config = {
        "model": {"name": "convlstm", "hidden_dim": 16},
        "training": {"epochs": target_epochs, "batch_size": 4},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    history = [
        {"epoch": ep, "global_step": ep * 100, "train_loss": 0.05 / ep, "val_loss": 0.04 / ep}
        for ep in range(1, target_epochs + 1)
    ]
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)

    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for rec in history:
            writer.writerow({"epoch": rec["epoch"], "train_loss": rec["train_loss"], "val_loss": rec["val_loss"]})

    with open(run_dir / "train.log", "w", encoding="utf-8") as f:
        f.write("Training started\nEpoch 5 completed\nTraining finished.\n")

    # Create dummy checkpoints
    best_ckpt = {
        "epoch": 5,
        "global_step": 500,
        "val_loss": 0.008,
        "best_val_loss": 0.008,
        "model_state_dict": {"layer.weight": torch.tensor([1.0, 2.0])},
        "config": config,
    }
    last_ckpt = {
        "epoch": 5,
        "global_step": 500,
        "val_loss": 0.008,
        "best_val_loss": 0.008,
        "model_state_dict": {"layer.weight": torch.tensor([1.0, 2.0])},
        "config": config,
    }
    torch.save(best_ckpt, run_dir / "checkpoint_best.pt")
    torch.save(last_ckpt, run_dir / "checkpoint_last.pt")

    return run_dir, tmp_path


def test_compute_sha256(tmp_path: Path):
    test_f = tmp_path / "hello.txt"
    test_f.write_text("QPredRNN test checksum", encoding="utf-8")
    digest = compute_sha256(test_f)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_inspect_checkpoint_valid(mock_colab_run):
    run_dir, _ = mock_colab_run
    meta = inspect_checkpoint(run_dir / "checkpoint_best.pt")
    assert meta["loadable"] is True
    assert meta["epoch"] == 5
    assert meta["global_step"] == 500
    assert meta["val_loss"] == 0.008
    assert meta["tensors_finite"] is True


def test_verify_run_consistency_complete(mock_colab_run):
    run_dir, _ = mock_colab_run
    best_meta = inspect_checkpoint(run_dir / "checkpoint_best.pt")
    last_meta = inspect_checkpoint(run_dir / "checkpoint_last.pt")
    res = verify_run_consistency(run_dir, best_meta, last_meta)

    assert res["is_complete"] is True
    assert res["completed_epochs"] == 5
    assert res["target_epochs"] == 5
    assert len(res["discrepancies"]) == 0


def test_verify_run_consistency_incomplete(mock_colab_run):
    run_dir, _ = mock_colab_run
    best_meta = inspect_checkpoint(run_dir / "checkpoint_best.pt")
    # Simulate an incomplete last checkpoint (e.g. crashed at epoch 3 when target was 5)
    last_meta = {"epoch": 3, "global_step": 300, "val_loss": 0.015}

    res = verify_run_consistency(run_dir, best_meta, last_meta)
    assert res["is_complete"] is False
    assert any("INCOMPLETE RUN" in d for d in res["discrepancies"])


def test_dry_run_import(mock_colab_run, tmp_path: Path):
    _, source_root = mock_colab_run
    dest_root = tmp_path / "dest_runs"

    rep = import_colab_results(source_root, dest_root, dry_run=True, overwrite=False)
    assert rep["dry_run"] is True
    assert rep["summary"]["verified"] >= 1
    # Verify no files were written in dry run
    assert not dest_root.exists() or len(list(dest_root.iterdir())) == 0


def test_import_and_refuse_overwrite_protection(mock_colab_run, tmp_path: Path):
    _, source_root = mock_colab_run
    dest_root = tmp_path / "dest_runs"

    # First import
    rep1 = import_colab_results(source_root, dest_root, dry_run=False, overwrite=False)
    assert rep1["summary"]["imported"] >= 1
    imported_best = dest_root / "kth_classical_tpu_50ep" / "checkpoint_best.pt"
    assert imported_best.exists()

    # Second import without --overwrite should skip
    rep2 = import_colab_results(source_root, dest_root, dry_run=False, overwrite=False)
    assert rep2["summary"]["skipped"] >= 1
    assert rep2["runs"]["kth_classical_tpu_50ep"]["status"] == "SKIPPED_EXISTING"


def test_import_with_config_used_yaml(mock_colab_run, tmp_path: Path):
    run_dir, source_root = mock_colab_run
    # Rename config.yaml to config_used.yaml
    cfg_file = run_dir / "config.yaml"
    cfg_used = run_dir / "config_used.yaml"
    cfg_file.rename(cfg_used)

    dest_root = tmp_path / "dest_runs_config_used"
    rep = import_colab_results(source_root, dest_root, dry_run=False, overwrite=False)
    assert rep["summary"]["imported"] >= 1
    imported_run = dest_root / "kth_classical_tpu_50ep"
    assert (imported_run / "checkpoint_best.pt").exists()
    assert (imported_run / "config.yaml").exists()
    assert (imported_run / "config_used.yaml").exists()

