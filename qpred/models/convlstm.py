from __future__ import annotations
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """A standard 2D Convolutional LSTM cell.

    Operates over 4D tensors of shape (B, C, H, W).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution weights with Xavier uniform and biases to zero."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W).
            state: Tuple of previous (h, c) each of shape (B, C_hidden, H, W).
                   If None, initialized to zeros.

        Returns:
            Tuple of updated (h, c) each of shape (B, C_hidden, H, W).
        """
        b, _, h, w = x.shape
        if state is None:
            zeros = torch.zeros(
                b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype
            )
            h_prev, c_prev = zeros, zeros
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class VideoConvLSTM(nn.Module):
    """Compact stacked ConvLSTM baseline for autoregressive video prediction.

    Pipeline baseline model; this is not the official THUML PredRNN architecture.
    Accepts (B, T_in, C, H, W) and generates (B, T_out, C, H, W).
    """

    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 16,
        input_frames: int = 5,
        output_frames: int = 5,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_layers = num_layers

        cells: list[ConvLSTMCell] = []
        for layer_idx in range(num_layers):
            c_in = image_channels if layer_idx == 0 else hidden_channels
            cells.append(ConvLSTMCell(c_in, hidden_channels, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, image_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        future_frames: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively predict future video frames.

        Args:
            x: Observed context frames of shape (B, T_in, C, H, W).
            future_frames: Number of future frames to predict (defaults to self.output_frames).

        Returns:
            Predicted frames tensor of shape (B, T_out, C, H, W) in [0.0, 1.0].
        """
        b, t_in, c, h, w = x.shape
        n_out = future_frames if future_frames is not None else self.output_frames

        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers

        # 1. Encode context frames through the ConvLSTM stack
        last_pred = None
        for t in range(t_in):
            cur_in = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = cell(cur_in, states[layer_idx])
                states[layer_idx] = (h_cur, c_cur)
                cur_in = h_cur
            last_pred = self.readout(states[-1][0])

        # 2. Autoregressively roll out future frame predictions
        preds: list[torch.Tensor] = []
        cur_input = last_pred if last_pred is not None else x[:, -1]
        for _ in range(n_out):
            cur_in = cur_input
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = cell(cur_in, states[layer_idx])
                states[layer_idx] = (h_cur, c_cur)
                cur_in = h_cur
            pred_frame = self.readout(states[-1][0])
            preds.append(pred_frame)
            cur_input = pred_frame

        return torch.stack(preds, dim=1)
