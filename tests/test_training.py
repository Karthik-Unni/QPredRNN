from pathlib import Path
import json
import torch
from qpred.train import run_training


def test_train_pipeline_smoke_run(tmp_path):
    root_dir = Path(__file__).parents[1]
    cfg_file = root_dir / "configs" / "smoke.yaml"
    data_root = root_dir / "data" / "toy"
    run_dir = tmp_path / "smoke_run"

    result = run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "history.json").exists()

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert history[0]["train_loss"] > 0.0
    assert history[0]["val_loss"] > 0.0

    # Test checkpoint contents
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "val_loss" in ckpt
