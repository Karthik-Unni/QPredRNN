from __future__ import annotations
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from qpred.data.dataset import FrameSequenceDataset
from qpred.evaluate import evaluate_windows


class DummyPredictor(nn.Module):
    """Simple dummy predictor outputting zeros or constant."""
    def __init__(self, output_frames: int = 5):
        super().__init__()
        self.output_frames = output_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t_in, c, h, w = x.shape
        return torch.full((b, self.output_frames, c, h, w), 0.5)


def test_evaluate_windows_toy():
    toy_path = Path("data/toy")
    if not toy_path.exists():
        pytest.skip("data/toy not present")

    ds = FrameSequenceDataset(toy_path, split="val", input_frames=5, output_frames=5, stride=1)
    predictor = DummyPredictor(output_frames=5)

    res = evaluate_windows(predictor, ds, num_windows=5, device="cpu", output_frames=5)

    assert res["num_windows"] == 5
    assert res["is_finite"] is True
    assert "mean_mse" in res
    assert "mean_mae" in res
    assert "mean_psnr" in res
    assert "mean_ssim" in res
    assert len(res["per_horizon"]["mse"]) == 5
    assert len(res["per_horizon"]["mae"]) == 5
    assert len(res["per_horizon"]["psnr"]) == 5
    assert len(res["per_horizon"]["ssim"]) == 5


def test_evaluate_baselines_toy():
    toy_path = Path("data/toy")
    if not toy_path.exists():
        pytest.skip("data/toy not present")

    ds = FrameSequenceDataset(toy_path, split="val", input_frames=5, output_frames=5, stride=1)

    res_pers = evaluate_windows("persistence", ds, num_windows=5, device="cpu", output_frames=5)
    res_mean = evaluate_windows("mean_context", ds, num_windows=5, device="cpu", output_frames=5)

    assert res_pers["num_windows"] == 5
    assert res_pers["is_finite"] is True
    assert res_mean["num_windows"] == 5
    assert res_mean["is_finite"] is True
    assert res_pers["mean_mse"] >= 0.0
    assert res_mean["mean_mse"] >= 0.0
