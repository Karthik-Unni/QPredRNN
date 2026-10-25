from pathlib import Path
import json
import torch
import torch.nn as nn
from qpred.models.predrnn_v2 import (
    PredRNNV2,
    SpatioTemporalLSTMCell,
    compute_memory_decoupling_loss,
    reserve_schedule_sampling_exp,
    generate_scheduled_sampling_masks,
)
from qpred.models.convlstm import VideoConvLSTM
from qpred.models import build_model
from qpred.train import get_rss_probabilities, run_training


def test_st_lstm_cell_step():
    cell = SpatioTemporalLSTMCell(in_channels=1, hidden_channels=8, kernel_size=3)
    x = torch.randn(2, 1, 16, 16)
    m_prev = torch.randn(2, 8, 16, 16)
    h, c, m, dc, dm = cell(x, state=None, m_prev=m_prev)

    assert h.shape == (2, 8, 16, 16)
    assert c.shape == (2, 8, 16, 16)
    assert m.shape == (2, 8, 16, 16)
    assert dc.shape == (2, 8, 16, 16)
    assert dm.shape == (2, 8, 16, 16)

    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()
    assert torch.isfinite(m).all()
    assert torch.isfinite(dc).all()
    assert torch.isfinite(dm).all()


def test_predrnn_v2_output_shapes():
    model = PredRNNV2(
        image_channels=1,
        hidden_channels=8,
        input_frames=4,
        output_frames=4,
        num_layers=2,
    )
    b, t_in, c, h, w = 2, 4, 1, 32, 32
    x = torch.rand(b, t_in, c, h, w)

    out = model(x)
    assert out.shape == (2, 4, 1, 32, 32)
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0

    # Test future_frames override
    out_override = model(x, future_frames=7)
    assert out_override.shape == (2, 7, 1, 32, 32)


def test_predrnn_v2_finite_predictions_and_auxiliary_loss():
    model = PredRNNV2(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
    )
    x = torch.rand(2, 3, 1, 16, 16)
    y = torch.rand(2, 3, 1, 16, 16)

    preds, decouple_loss = model(x, target=y, return_decouple_loss=True)
    assert preds.shape == (2, 3, 1, 16, 16)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(decouple_loss)
    assert 0.0 <= decouple_loss.item() <= 1.0


def test_predrnn_v2_gradient_flow():
    model = PredRNNV2(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    initial_weight = model.cells[0].conv_x.weight.clone().detach()

    x = torch.rand(2, 3, 1, 16, 16)
    y = torch.rand(2, 3, 1, 16, 16)

    optimizer.zero_grad()
    preds, decouple_loss = model(x, target=y, return_decouple_loss=True)
    mse_loss = criterion(preds, y)
    total_loss = mse_loss + 0.1 * decouple_loss

    total_loss.backward()

    # Check finite gradients on key components
    assert model.cells[0].conv_x.weight.grad is not None
    assert torch.isfinite(model.cells[0].conv_x.weight.grad).all()
    assert model.cells[0].conv_1x1.weight.grad is not None
    assert torch.isfinite(model.cells[0].conv_1x1.weight.grad).all()

    optimizer.step()
    updated_weight = model.cells[0].conv_x.weight.detach()
    assert not torch.equal(initial_weight, updated_weight), "Optimizer step did not update weights!"


def test_predrnn_v2_factory_registration():
    classical_model = build_model({"model": {"name": "classical", "hidden_channels": 8}})
    assert isinstance(classical_model, VideoConvLSTM)

    predrnn_model = build_model({"model": {"name": "predrnn_v2", "hidden_channels": 8, "num_layers": 2}})
    assert isinstance(predrnn_model, PredRNNV2)
    assert predrnn_model.num_layers == 2
    assert predrnn_model.hidden_channels == 8


def test_reverse_scheduled_sampling():
    rss_cfg = {
        "enabled": True,
        "r_sampling_step_1": 100,
        "r_sampling_step_2": 200,
        "r_exp_alpha": 50.0,
    }

    # At step 0: r_eta = 0.5, eta = 0.5
    r_eta, eta = get_rss_probabilities(0, rss_cfg)
    assert r_eta == 0.5
    assert eta == 0.5

    # At step 200: r_eta = 1.0, eta = 0.0
    r_eta_end, eta_end = get_rss_probabilities(200, rss_cfg)
    assert r_eta_end == 1.0
    assert eta_end == 0.0

    # Disabled schedule defaults to pure ground-truth context and pure autoregressive future
    r_eta_dis, eta_dis = get_rss_probabilities(50, {"enabled": False})
    assert r_eta_dis == 1.0
    assert eta_dis == 0.0

    # Model forward with dual scheduled sampling masks
    model = PredRNNV2(image_channels=1, hidden_channels=8, input_frames=3, output_frames=3, num_layers=2)
    model.train()
    x = torch.rand(2, 3, 1, 16, 16)
    target = torch.rand(2, 3, 1, 16, 16)
    mask_c, mask_f = generate_scheduled_sampling_masks(
        batch_size=2, input_frames=3, output_frames=3, r_eta=0.5, eta=0.5
    )
    out = model(x, target=target, mask_context=mask_c, mask_future=mask_f)
    assert out.shape == (2, 3, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_rss_schedule_progression():
    step_1 = 1000
    step_2 = 3000
    alpha = 500.0

    # 1. Before step_1: both stay at 0.5
    r_eta_0, eta_0 = reserve_schedule_sampling_exp(0, step_1, step_2, alpha)
    r_eta_999, eta_999 = reserve_schedule_sampling_exp(999, step_1, step_2, alpha)
    assert r_eta_0 == 0.5 and eta_0 == 0.5
    assert r_eta_999 == 0.5 and eta_999 == 0.5

    # 2. Progression between step_1 and step_2
    steps = [1000, 1500, 2000, 2500, 3000]
    r_etas = []
    etas = []
    for s in steps:
        r, e = reserve_schedule_sampling_exp(s, step_1, step_2, alpha)
        r_etas.append(r)
        etas.append(e)

    # r_eta must monotonically INCREASE towards 1.0
    for i in range(len(r_etas) - 1):
        assert r_etas[i] <= r_etas[i + 1], f"r_eta not monotonically increasing: {r_etas}"
    assert r_etas[0] == 0.5
    assert r_etas[-1] == 1.0

    # eta must monotonically DECREASE towards 0.0
    for i in range(len(etas) - 1):
        assert etas[i] >= etas[i + 1], f"eta not monotonically decreasing: {etas}"
    assert etas[0] == 0.5
    assert etas[-1] == 0.0

    # 3. Beyond step_2: boundaries stay clamped
    r_eta_post, eta_post = reserve_schedule_sampling_exp(5000, step_1, step_2, alpha)
    assert r_eta_post == 1.0
    assert eta_post == 0.0


def test_rss_mask_generation():
    b, t_in, t_out = 4, 5, 6

    # Extreme 1: r_eta=1.0 (all 1s for context), eta=0.0 (all 0s for future)
    m_ctx, m_fut = generate_scheduled_sampling_masks(b, t_in, t_out, r_eta=1.0, eta=0.0)
    assert m_ctx.shape == (b, t_in - 1)
    assert m_fut.shape == (b, t_out)
    assert (m_ctx == 1.0).all()
    assert (m_fut == 0.0).all()

    # Extreme 2: r_eta=0.0 (all 0s for context), eta=1.0 (all 1s for future)
    m_ctx_inv, m_fut_inv = generate_scheduled_sampling_masks(b, t_in, t_out, r_eta=0.0, eta=1.0)
    assert (m_ctx_inv == 0.0).all()
    assert (m_fut_inv == 1.0).all()

    # Intermediate values are binary 0.0 or 1.0
    m_ctx_mid, m_fut_mid = generate_scheduled_sampling_masks(b, t_in, t_out, r_eta=0.5, eta=0.5)
    assert ((m_ctx_mid == 0.0) | (m_ctx_mid == 1.0)).all()
    assert ((m_fut_mid == 0.0) | (m_fut_mid == 1.0)).all()

    # Edge case: single input frame t_in=1
    m_ctx_single, m_fut_single = generate_scheduled_sampling_masks(b, 1, t_out, r_eta=0.5, eta=0.5)
    assert m_ctx_single.shape == (b, 0)
    assert m_fut_single.shape == (b, t_out)


def test_rss_masks_applied_to_correct_sections():
    torch.manual_seed(42)
    model = PredRNNV2(image_channels=1, hidden_channels=8, input_frames=3, output_frames=3, num_layers=2)
    model.train()

    b, t_in, t_out, c, h, w = 2, 3, 3, 1, 16, 16
    x = torch.rand(b, t_in, c, h, w)
    target1 = torch.rand(b, t_out, c, h, w)
    target2 = target1 + 5.0  # distinct target

    # Test Future Mask section isolation:
    # When mask_future is all 0.0, target is decoupled during future rollout
    mask_c = torch.ones(b, t_in - 1)
    mask_f_zeros = torch.zeros(b, t_out)
    out1 = model(x, target=target1, mask_context=mask_c, mask_future=mask_f_zeros)
    out2 = model(x, target=target2, mask_context=mask_c, mask_future=mask_f_zeros)
    assert torch.allclose(out1, out2), "Future mask=0 did not decouple output from target ground-truth!"

    # Conversely, when mask_future is all 1.0, future steps receive target, so outputs MUST differ
    mask_f_ones = torch.ones(b, t_out)
    out_tf1 = model(x, target=target1, mask_context=mask_c, mask_future=mask_f_ones)
    out_tf2 = model(x, target=target2, mask_context=mask_c, mask_future=mask_f_ones)
    assert not torch.allclose(out_tf1, out_tf2), "Future mask=1 failed to feed target ground truth to future steps!"

    # Test Context Mask section isolation:
    # When mask_context is all 1.0 vs all 0.0 (holding mask_future constant at 0), context transitions differ
    out_ctx_gt = model(x, target=target1, mask_context=torch.ones(b, t_in - 1), mask_future=mask_f_zeros)
    out_ctx_pred = model(x, target=target1, mask_context=torch.zeros(b, t_in - 1), mask_future=mask_f_zeros)
    assert not torch.allclose(out_ctx_gt, out_ctx_pred), "Context mask did not alter context feed transitions!"


def test_inference_zero_ground_truth_leakage():
    model = PredRNNV2(image_channels=1, hidden_channels=8, input_frames=3, output_frames=3, num_layers=2)
    model.eval()

    b, t_in, t_out, c, h, w = 2, 3, 3, 1, 16, 16
    x = torch.rand(b, t_in, c, h, w)
    target = torch.rand(b, t_out, c, h, w)
    mask_c = torch.ones(b, t_in - 1)
    mask_f = torch.ones(b, t_out)

    with torch.no_grad():
        clean_out = model(x)
        target_leak_candidate = model(x, target=target, mask_context=mask_c, mask_future=mask_f)

    assert torch.allclose(clean_out, target_leak_candidate), (
        "Ground truth leaked during inference/evaluation! model.eval() must remain purely autoregressive."
    )


def test_predrnn_v2_checkpoint_compatibility(tmp_path):
    model1 = PredRNNV2(image_channels=1, hidden_channels=8, input_frames=3, output_frames=3, num_layers=2)
    ckpt_file = tmp_path / "predrnn_v2.pt"
    torch.save(model1.state_dict(), ckpt_file)

    model2 = PredRNNV2(image_channels=1, hidden_channels=8, input_frames=3, output_frames=3, num_layers=2)
    model2.load_state_dict(torch.load(ckpt_file, weights_only=False))

    x = torch.rand(1, 3, 1, 16, 16)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2), "Reloaded PredRNN-V2 checkpoint produced mismatched predictions!"


def test_predrnn_v2_smoke_training_pipeline(tmp_path):
    root_dir = Path(__file__).parents[1]
    cfg_file = root_dir / "configs" / "kth_predrnn_v2_smoke.yaml"
    data_root = root_dir / "data" / "toy"
    run_dir = tmp_path / "predrnn_v2_smoke"

    result = run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "history.json").exists()

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "decouple_loss" in history[0]
    assert "r_eta" in history[0]
    assert "eta" in history[0]
    assert history[0]["train_loss"] > 0.0
    assert history[0]["val_loss"] > 0.0
    assert history[0]["decouple_loss"] >= 0.0
