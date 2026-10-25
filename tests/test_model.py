import torch
import torch.nn as nn
from qpred.models.convlstm import VideoConvLSTM, ConvLSTMCell
from qpred.models import build_model


def test_convlstm_cell_step():
    cell = ConvLSTMCell(in_channels=1, hidden_channels=8, kernel_size=3)
    x = torch.randn(2, 1, 16, 16)
    h, c = cell(x)
    assert h.shape == (2, 8, 16, 16)
    assert c.shape == (2, 8, 16, 16)
    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()


def test_model_output_shapes():
    model = VideoConvLSTM(
        image_channels=1,
        hidden_channels=16,
        input_frames=5,
        output_frames=5,
        num_layers=2,
    )
    b, t_in, c, h, w = 2, 5, 1, 32, 32
    x = torch.rand(b, t_in, c, h, w)
    out = model(x)
    assert out.shape == (2, 5, 1, 32, 32)
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0

    # Test future_frames override
    out_10 = model(x, future_frames=10)
    assert out_10.shape == (2, 10, 1, 32, 32)


def test_model_finite_predictions_and_loss():
    model = VideoConvLSTM(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=1,
    )
    x = torch.rand(2, 3, 1, 16, 16)
    y = torch.rand(2, 3, 1, 16, 16)
    pred = model(x)
    assert torch.isfinite(pred).all()

    criterion = nn.MSELoss()
    loss = criterion(pred, y)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_backpropagation_and_optimizer_updates():
    model = VideoConvLSTM(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    initial_weights = model.cells[0].conv.weight.clone().detach()

    x = torch.rand(2, 3, 1, 16, 16)
    y = torch.rand(2, 3, 1, 16, 16)

    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    updated_weights = model.cells[0].conv.weight.detach()
    assert not torch.equal(initial_weights, updated_weights), "Weights did not update after optimizer step!"


def test_checkpoint_save_and_reload(tmp_path):
    model1 = VideoConvLSTM(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
    )
    ckpt_file = tmp_path / "model.pt"
    torch.save(model1.state_dict(), ckpt_file)

    model2 = VideoConvLSTM(
        image_channels=1,
        hidden_channels=8,
        input_frames=3,
        output_frames=3,
        num_layers=2,
    )
    model2.load_state_dict(torch.load(ckpt_file, weights_only=False))

    x = torch.rand(1, 3, 1, 16, 16)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2), "Reloaded model produced different predictions!"
