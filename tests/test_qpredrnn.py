from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from qpred.models import (
    build_model,
    QPredRNN,
    QuantumContextModulator,
)
from qpred.models.qpredrnn import create_quantum_vqc_layer
from qpred.train import run_training


def test_qpredrnn_factory_build():
    """Verify build_model instantiates QPredRNN for both qpredrnn and quantum_context names."""
    cfg_1 = {
        "model": {
            "name": "qpredrnn",
            "hidden_channels": 8,
            "num_layers": 2,
            "input_frames": 4,
            "output_frames": 4,
            "quantum": {"enabled": True, "n_qubits": 4, "circuit_depth": 2},
        }
    }
    model_1 = build_model(cfg_1)
    assert isinstance(model_1, QPredRNN)
    assert model_1.is_quantum is True
    assert model_1.n_qubits == 4
    assert model_1.circuit_depth == 2

    cfg_2 = {
        "model": {
            "name": "quantum_context",
            "hidden_channels": 8,
            "num_layers": 2,
            "input_frames": 4,
            "output_frames": 4,
            "quantum": {"enabled": True, "n_qubits": 4, "circuit_depth": 1},
        }
    }
    model_2 = build_model(cfg_2)
    assert isinstance(model_2, QPredRNN)
    assert model_2.circuit_depth == 1


def test_quantum_vqc_gradients():
    """Verify quantum VQC layer receives analytical gradients from loss."""
    n_qubits = 4
    depth = 2
    layer = create_quantum_vqc_layer(n_qubits=n_qubits, circuit_depth=depth)

    # Input tensor
    x = torch.randn(2, n_qubits, requires_grad=True)
    out = layer(x)

    assert out.shape == (2, n_qubits)
    assert torch.isfinite(out).all()
    # Expectation values of PauliZ are bounded in [-1, 1]
    assert -1.0 <= out.min().item() and out.max().item() <= 1.0

    loss = out.sum()
    loss.backward()

    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()
    assert (layer.weights.grad != 0).any()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_qpredrnn_forward_shapes_and_finite():
    """Verify forward pass tensor shapes and output range [0, 1]."""
    model = QPredRNN(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    b, t_in, c, h, w = 2, 3, 1, 16, 16
    x = torch.rand(b, t_in, c, h, w)

    out = model(x)
    assert out.shape == (2, 3, 1, 16, 16)
    assert torch.isfinite(out).all()
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0


def test_qpredrnn_end_to_end_gradient_flow():
    """Verify backpropagation through quantum modulator and classical ST-LSTM cells."""
    model = QPredRNN(
        image_channels=1,
        hidden_channels=8,
        input_frames=2,
        output_frames=2,
        num_layers=2,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.rand(2, 2, 1, 16, 16)
    y = torch.rand(2, 2, 1, 16, 16)

    initial_q_weight = model.quantum_modulator.vqc.weights.clone().detach()

    optimizer.zero_grad()
    preds, decouple_loss = model(x, target=y, return_decouple_loss=True)
    loss = criterion(preds, y) + 0.1 * decouple_loss
    loss.backward()

    # Verify gradients exist on quantum VQC weights
    q_grad = model.quantum_modulator.vqc.weights.grad
    assert q_grad is not None
    assert torch.isfinite(q_grad).all()
    assert (q_grad != 0).any(), "Quantum circuit weights received zero gradients!"

    # Verify gradients on classical convolutional cells
    assert model.cells[0].conv_x.weight.grad is not None
    assert torch.isfinite(model.cells[0].conv_x.weight.grad).all()

    # Verify step updates weights
    optimizer.step()
    updated_q_weight = model.quantum_modulator.vqc.weights.detach()
    assert not torch.equal(initial_q_weight, updated_q_weight), "Optimizer step failed to update quantum weights!"


def test_qpredrnn_inference_pure_autoregressive():
    """Verify eval mode does not leak ground truth targets."""
    model = QPredRNN(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    model.eval()

    b, t_in, t_out, c, h, w = 2, 3, 3, 1, 16, 16
    x = torch.rand(b, t_in, c, h, w)
    target1 = torch.rand(b, t_out, c, h, w)
    target2 = target1 + 10.0

    with torch.no_grad():
        out_clean = model(x)
        out_with_target1 = model(x, target=target1)
        out_with_target2 = model(x, target=target2)

    assert torch.allclose(out_clean, out_with_target1), "Ground truth target leaked during inference!"
    assert torch.allclose(out_with_target1, out_with_target2), "Different targets caused output divergence in eval mode!"


def test_qpredrnn_checkpoint_compatibility(tmp_path):
    """Verify saving and re-loading QPredRNN state dict."""
    model1 = QPredRNN(
        image_channels=1,
        hidden_channels=8,
        input_frames=2,
        output_frames=2,
        num_layers=2,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    ckpt_file = tmp_path / "qpredrnn_ckpt.pt"
    torch.save(model1.state_dict(), ckpt_file)

    model2 = QPredRNN(
        image_channels=1,
        hidden_channels=8,
        input_frames=2,
        output_frames=2,
        num_layers=2,
        quantum_cfg={"enabled": True, "n_qubits": 4, "circuit_depth": 1},
    )
    model2.load_state_dict(torch.load(ckpt_file, weights_only=False))

    x = torch.rand(1, 2, 1, 16, 16)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2), "Reloaded QPredRNN checkpoint produced mismatched predictions!"


def test_qpredrnn_smoke_training_run(tmp_path):
    """Run a 1-batch training pipeline test with QPredRNN on toy dataset."""
    root_dir = Path(__file__).parents[1]
    cfg_file = root_dir / "configs" / "kth_qpredrnn_smoke.yaml"
    data_root = root_dir / "data" / "toy"
    run_dir = tmp_path / "qpredrnn_smoke"

    result = run_training(
        config_path=cfg_file,
        data_root=data_root,
        run_dir=run_dir,
        device_name="cpu",
    )

    assert result["epochs_completed"] == 1
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "history.json").exists()

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["train_loss"] > 0.0
    assert history[0]["val_loss"] > 0.0
