from __future__ import annotations
import json
import csv
import sys
from pathlib import Path
import pytest
import torch
import yaml

from qpred.train import run_training, main


@pytest.fixture
def base_config(tmp_path):
    """Generate a lightweight base configuration using the toy dataset."""
    root_dir = Path(__file__).parents[1]
    data_root = root_dir / "data" / "toy"
    cfg = {
        "seed": 42,
        "model": {
            "name": "predrnn_v2",
            "image_channels": 1,
            "hidden_channels": 4,
            "num_layers": 2,
            "input_frames": 4,
            "output_frames": 4,
            "kernel_size": 3,
            "decouple_loss_weight": 0.05,
        },
        "data": {
            "image_size": 64,
            "input_frames": 4,
            "output_frames": 4,
            "stride": 1,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "max_train_batches": 2,
            "max_val_batches": 1,
            "reverse_scheduled_sampling": {
                "enabled": True,
                "r_sampling_step_1": 5,
                "r_sampling_step_2": 20,
                "r_exp_alpha": 5.0,
            },
        },
    }
    cfg_file = tmp_path / "train_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg, cfg_file, data_root


def test_resume_epoch_continuation_and_global_step(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_resume_step"

    # 1. Initial 1-epoch run
    result_1 = run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    assert result_1["epochs_completed"] == 1
    last_ckpt_path = run_dir / "checkpoint_last.pt"
    assert last_ckpt_path.exists()

    ckpt_1 = torch.load(last_ckpt_path, weights_only=False)
    assert ckpt_1["epoch"] == 1
    initial_step = ckpt_1["global_step"]
    assert initial_step > 0

    # 2. Extend configuration to 3 epochs
    cfg["training"]["epochs"] = 3
    cfg_extended = tmp_path / "cfg_extended.yaml"
    with open(cfg_extended, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 3. Resume training from checkpoint_last.pt
    result_2 = run_training(
        config_path=cfg_extended,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
        resume_path=last_ckpt_path,
    )
    assert result_2["epochs_completed"] == 3

    # Check history.json preservation and progression
    history_file = run_dir / "history.json"
    assert history_file.exists()
    history = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(history) == 3
    assert [h["epoch"] for h in history] == [1, 2, 3]

    # Global step progression
    assert history[0]["global_step"] == initial_step
    assert history[1]["global_step"] == initial_step * 2
    assert history[2]["global_step"] == initial_step * 3

    # Check metrics.csv preservation
    csv_file = run_dir / "metrics.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # Header + 3 epochs
    assert len(rows) == 4
    assert rows[1][0] == "1"
    assert rows[2][0] == "2"
    assert rows[3][0] == "3"


def test_resume_optimizer_state_and_weights_restoration(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_opt_restore"

    run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    last_ckpt_path = run_dir / "checkpoint_last.pt"
    ckpt_1 = torch.load(last_ckpt_path, weights_only=False)

    assert "optimizer_state_dict" in ckpt_1
    assert "state" in ckpt_1["optimizer_state_dict"]
    assert len(ckpt_1["optimizer_state_dict"]["state"]) > 0

    # Resume into a new run_dir pointing to ckpt_1
    run_dir_resumed = tmp_path / "run_opt_resumed"
    cfg["training"]["epochs"] = 2
    cfg_file_2 = tmp_path / "cfg_2.yaml"
    with open(cfg_file_2, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = run_training(
        config_path=cfg_file_2,
        data_root=data_root,
        run_dir=run_dir_resumed,
        device_name="cpu",
        resume_path=last_ckpt_path,
    )
    assert result["epochs_completed"] == 2
    ckpt_2 = torch.load(run_dir_resumed / "checkpoint_last.pt", weights_only=False)
    assert ckpt_2["epoch"] == 2
    assert ckpt_2["global_step"] > ckpt_1["global_step"]


def test_resume_preserves_best_validation_loss(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_best_loss"

    run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    best_ckpt_path = run_dir / "checkpoint_best.pt"
    ckpt_best = torch.load(best_ckpt_path, weights_only=False)

    # Artificially set best_val_loss to an unbeatable near-zero value in checkpoint
    artificial_best = 1e-8
    ckpt_best["val_loss"] = artificial_best
    ckpt_best["best_val_loss"] = artificial_best
    torch.save(ckpt_best, best_ckpt_path)

    # Resume for epoch 2
    cfg["training"]["epochs"] = 2
    cfg_file_2 = tmp_path / "cfg_2.yaml"
    with open(cfg_file_2, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    run_training(
        config_path=cfg_file_2,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
        resume_path=best_ckpt_path,
    )

    # Checkpoint best must NOT have been overwritten by epoch 2
    reloaded_best = torch.load(best_ckpt_path, weights_only=False)
    assert reloaded_best["epoch"] == 1
    assert abs(reloaded_best["best_val_loss"] - artificial_best) < 1e-12

    # History must show epoch 2 is_best is False
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[1]["is_best"] is False
    assert abs(history[1]["best_val_loss"] - artificial_best) < 1e-12


def test_resume_incompatible_configurations(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_incompat"

    run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    ckpt_path = run_dir / "checkpoint_last.pt"

    # Helper to test invalid config rejection
    def assert_rejected(modifier_fn, expected_match):
        modified_cfg = yaml.safe_load(open(cfg_file, "r", encoding="utf-8"))
        modified_cfg["training"]["epochs"] = 2
        modifier_fn(modified_cfg)
        test_cfg_path = tmp_path / "test_reject.yaml"
        with open(test_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(modified_cfg, f)

        with pytest.raises(ValueError, match=expected_match):
            run_training(
                config_path=test_cfg_path,
                data_root=data_root,
                run_dir=tmp_path / "run_reject",
                device_name="cpu",
                resume_path=ckpt_path,
            )

    # 1. Incompatible model parameter
    assert_rejected(
        lambda c: c["model"].__setitem__("name", "classical"),
        "Incompatible model parameter 'name'",
    )

    # 2. Incompatible batch size
    assert_rejected(
        lambda c: c["training"].__setitem__("batch_size", 99),
        "Incompatible batch size",
    )

    # 3. Incompatible data parameter
    assert_rejected(
        lambda c: c["data"].__setitem__("input_frames", 10),
        "Incompatible data parameter 'input_frames'",
    )

    # 4. Incompatible scheduled sampling parameter
    assert_rejected(
        lambda c: c["training"]["reverse_scheduled_sampling"].__setitem__("r_sampling_step_1", 9999),
        "Incompatible reverse_scheduled_sampling parameter 'r_sampling_step_1'",
    )


def test_resume_older_checkpoint_warning(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_legacy"

    run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    ckpt_path = run_dir / "checkpoint_last.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)

    # Strip rng_states and config to simulate older checkpoint format
    ckpt.pop("rng_states", None)
    ckpt.pop("config", None)
    legacy_ckpt_path = tmp_path / "checkpoint_legacy.pt"
    torch.save(ckpt, legacy_ckpt_path)

    cfg["training"]["epochs"] = 2
    cfg_file_2 = tmp_path / "cfg_2.yaml"
    with open(cfg_file_2, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.warns(UserWarning) as record:
        result = run_training(
            config_path=cfg_file_2,
            data_root=data_root,
            run_dir=tmp_path / "run_resumed_legacy",
            device_name="cpu",
            resume_path=legacy_ckpt_path,
        )

    assert result["epochs_completed"] == 2
    warn_msgs = [str(w.message) for w in record]
    assert any("rng_states" in msg for msg in warn_msgs)
    assert any("config" in msg for msg in warn_msgs)


def test_resume_already_completed_epochs(tmp_path, base_config):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_already_done"

    run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )
    ckpt_path = run_dir / "checkpoint_last.pt"

    # Resuming with epochs = 1 when checkpoint is already at epoch 1
    result = run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
        resume_path=ckpt_path,
    )
    assert result["epochs_completed"] == 1


def test_resume_cli_flag(tmp_path, base_config, monkeypatch):
    cfg, cfg_file, data_root = base_config
    run_dir = tmp_path / "run_cli"

    # Run initial epoch via CLI
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_file),
            "--data-root",
            str(data_root),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
        ],
    )
    main()
    assert (run_dir / "checkpoint_last.pt").exists()

    # Resume via CLI with --resume
    cfg["training"]["epochs"] = 2
    cfg_file_2 = tmp_path / "cfg_2.yaml"
    with open(cfg_file_2, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_file_2),
            "--data-root",
            str(data_root),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--resume",
            str(run_dir / "checkpoint_last.pt"),
        ],
    )
    main()

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
