from __future__ import annotations
import numpy as np
import torch
import pytest

from qpred.metrics import (
    compute_mse,
    compute_mae,
    compute_psnr,
    compute_ssim,
    compute_sequence_metrics,
)


def test_metrics_identical_inputs():
    """Verify MSE/MAE is 0 and PSNR is high for identical tensors."""
    t = torch.rand(10, 1, 64, 64)
    mse = compute_mse(t, t)
    mae = compute_mae(t, t)
    psnr = compute_psnr(t, t)
    ssim = compute_ssim(t, t)

    assert mse == 0.0
    assert mae == 0.0
    assert psnr == 100.0
    assert abs(ssim - 1.0) < 1e-4


def test_metrics_different_inputs():
    """Verify positive error metrics and degraded PSNR/SSIM."""
    t1 = torch.zeros(10, 1, 32, 32)
    t2 = torch.ones(10, 1, 32, 32)

    mse = compute_mse(t1, t2)
    mae = compute_mae(t1, t2)
    psnr = compute_psnr(t1, t2)
    ssim = compute_ssim(t1, t2)

    assert mse == 1.0
    assert mae == 1.0
    assert psnr == 0.0  # 20*log10(1) - 10*log10(1) = 0
    assert ssim < 0.1


def test_sequence_metrics_structure():
    """Verify compute_sequence_metrics returns expected dict and per-frame entries."""
    pred = torch.rand(2, 5, 1, 16, 16)
    target = torch.rand(2, 5, 1, 16, 16)

    res = compute_sequence_metrics(pred, target)
    assert "mse" in res
    assert "mae" in res
    assert "psnr" in res
    assert "ssim" in res
    assert "per_frame" in res

    assert len(res["per_frame"]["mse"]) == 5
    assert len(res["per_frame"]["mae"]) == 5
    assert len(res["per_frame"]["psnr"]) == 5
    assert len(res["per_frame"]["ssim"]) == 5


def test_metrics_per_horizon_independence():
    """Verify that per-frame curves evaluate metrics independently at future steps 1..T."""
    t_steps = 10
    pred = torch.rand(t_steps, 1, 32, 32)
    target = torch.rand(t_steps, 1, 32, 32)

    seq_metrics = compute_sequence_metrics(pred, target)

    # For each individual horizon step t, compute metrics directly on that single frame slice
    for t in range(t_steps):
        p_t = pred[t : t + 1]
        tgt_t = target[t : t + 1]

        expected_mse = compute_mse(p_t, tgt_t)
        expected_mae = compute_mae(p_t, tgt_t)
        expected_psnr = compute_psnr(p_t, tgt_t)

        assert abs(seq_metrics["per_frame"]["mse"][t] - expected_mse) < 1e-6
        assert abs(seq_metrics["per_frame"]["mae"][t] - expected_mae) < 1e-6
        assert abs(seq_metrics["per_frame"]["psnr"][t] - expected_psnr) < 1e-5


def test_constant_baseline_vs_uniform_midgray():
    """Verify diagnostic baseline behavior: persistent background outperforms unadapted 0.50 frame."""
    # Video background with pixel intensity ~0.40 and moving small patch
    target = torch.full((10, 1, 64, 64), 0.40)
    # Context last frame is at 0.40
    last_context_frame = torch.full((1, 1, 64, 64), 0.40)
    persistence_pred = last_context_frame.repeat(10, 1, 1, 1)

    # Unadapted Sigmoid readout (flat 0.50 mid-grey)
    unadapted_pred = torch.full((10, 1, 64, 64), 0.50)

    met_pers = compute_sequence_metrics(persistence_pred, target)
    met_unad = compute_sequence_metrics(unadapted_pred, target)

    # Persistence should have 0 MSE on static background
    assert met_pers["mse"] == 0.0
    # Unadapted mid-gray has (0.50 - 0.40)^2 = 0.01 MSE
    assert abs(met_unad["mse"] - 0.01) < 1e-6
    # Persistence baseline strictly outperforms unadapted model
    assert met_pers["mse"] < met_unad["mse"]
    assert met_pers["psnr"] > met_unad["psnr"]
