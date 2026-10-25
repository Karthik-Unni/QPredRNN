from __future__ import annotations
import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn


def compute_memory_decoupling_loss(
    delta_c_list: List[torch.Tensor],
    delta_m_list: List[torch.Tensor],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the memory-decoupling auxiliary loss between temporal (C) and spatiotemporal (M) memory.

    L_decouple = sum_{t, l} |<Delta C_t^l, Delta M_t^l>| / (||Delta C_t^l||_2 * ||Delta M_t^l||_2 + eps)
    averaged across batch and sequence steps.
    """
    if not delta_c_list:
        return torch.tensor(0.0)

    total_decouple = torch.tensor(0.0, device=delta_c_list[0].device)
    count = 0

    for dc, dm in zip(delta_c_list, delta_m_list):
        # Flatten spatial and channel dimensions per batch item: (B, C * H * W)
        dc_flat = dc.reshape(dc.size(0), -1)
        dm_flat = dm.reshape(dm.size(0), -1)

        dot = torch.sum(dc_flat * dm_flat, dim=1)  # (B,)
        norm_c = torch.norm(dc_flat, p=2, dim=1)   # (B,)
        norm_m = torch.norm(dm_flat, p=2, dim=1)   # (B,)

        cos_sim = torch.abs(dot) / (norm_c * norm_m + eps)  # (B,) bounded in [0, 1]
        total_decouple = total_decouple + cos_sim.mean()
        count += 1

    return total_decouple / max(1, count)


def reserve_schedule_sampling_exp(
    itr: int,
    r_sampling_step_1: int = 25000,
    r_sampling_step_2: int = 50000,
    r_exp_alpha: float = 5000.0,
) -> Tuple[float, float]:
    """Official Reverse Scheduled Sampling (RSS) schedules from THUML PredRNN-V2.

    Returns:
        r_eta: context/encoder sampling probability (INCREASES: 0.5 -> 1.0)
        eta: future/decoder sampling probability (DECREASES: 0.5 -> 0.0)
    """
    # Context / Encoder schedule: starts at 0.5 and exponentially increases to 1.0
    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - r_sampling_step_1) / max(1e-5, r_exp_alpha))
    else:
        r_eta = 1.0

    # Future / Decoder schedule: starts at 0.5 and linearly decreases to 0.0
    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / max(1, r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    return float(r_eta), float(eta)


def generate_scheduled_sampling_masks(
    batch_size: int,
    input_frames: int,
    output_frames: int,
    r_eta: float,
    eta: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate boolean / float sampling masks for context and future transitions.

    Args:
        batch_size: Batch size B.
        input_frames: Number of context frames T_in.
        output_frames: Number of future frames T_out.
        r_eta: Probability of using ground-truth in context phase transitions (length T_in - 1).
        eta: Probability of using ground-truth in future phase transitions (length T_out).
        device: Device on which to allocate masks.

    Returns:
        mask_context: shape (B, input_frames - 1), 1.0 for ground truth, 0.0 for prediction.
        mask_future: shape (B, output_frames), 1.0 for ground truth, 0.0 for prediction.
    """
    if input_frames > 1:
        r_rand = torch.rand(batch_size, input_frames - 1, device=device)
        mask_context = (r_rand < r_eta).float()
    else:
        mask_context = torch.ones(batch_size, 0, device=device)

    rand = torch.rand(batch_size, output_frames, device=device)
    mask_future = (rand < eta).float()

    return mask_context, mask_future


class SpatioTemporalLSTMCell(nn.Module):
    """SpatioTemporal LSTM cell (ST-LSTM) as formulated in PredRNN / PredRNN-V2.

    Maintains dual memory flows:
      - Temporal memory C_t^l: flows horizontally across time within the same layer.
      - Spatiotemporal memory M_t^l: flows vertically across layers and horizontally across time (zigzag flow).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Convolutions for input x: produces 7 chunks
        # 3 for temporal gates (i, f, g), 3 for spatiotemporal gates (i', f', g'), 1 for output gate (o)
        self.conv_x = nn.Conv2d(
            in_channels,
            7 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Convolutions for hidden state h: produces 4 chunks
        # 3 for temporal gates (i, f, g), 1 for output gate (o)
        self.conv_h = nn.Conv2d(
            hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Convolutions for spatiotemporal memory m: produces 3 chunks
        # 3 for spatiotemporal gates (i', f', g')
        self.conv_m = nn.Conv2d(
            hidden_channels,
            3 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Output gate couplings to updated memory states
        self.conv_c_o = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv_m_o = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # 1x1 convolution for coupling temporal and spatiotemporal memory into hidden state
        self.conv_1x1 = nn.Conv2d(
            2 * hidden_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Xavier uniform and biases to zero."""
        for m in (self.conv_x, self.conv_h, self.conv_m, self.conv_c_o, self.conv_m_o, self.conv_1x1):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step and layer.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).
            state: Previous temporal states (h_prev, c_prev) of shape (B, hidden_channels, H, W).
            m_prev: Previous spatiotemporal memory of shape (B, hidden_channels, H, W).

        Returns:
            Tuple of:
              h_cur: (B, hidden_channels, H, W)
              c_cur: (B, hidden_channels, H, W)
              m_cur: (B, hidden_channels, H, W)
              delta_c: temporal memory increment (B, hidden_channels, H, W)
              delta_m: spatiotemporal memory increment (B, hidden_channels, H, W)
        """
        b, _, h, w = x.shape
        if state is None:
            zeros = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            h_prev, c_prev = zeros, zeros
        else:
            h_prev, c_prev = state

        # Compute conv projections
        x_gates = self.conv_x(x)
        h_gates = self.conv_h(h_prev)
        m_gates = self.conv_m(m_prev)

        # Unpack temporal gate inputs
        xi, xf, xg, xi_m, xf_m, xg_m, xo = torch.split(x_gates, self.hidden_channels, dim=1)
        hi, hf, hg, ho = torch.split(h_gates, self.hidden_channels, dim=1)
        mi, mf, mg = torch.split(m_gates, self.hidden_channels, dim=1)

        # 1. Temporal memory update (horizontal flow)
        i_t = torch.sigmoid(xi + hi)
        f_t = torch.sigmoid(xf + hf)
        g_t = torch.tanh(xg + hg)
        delta_c = i_t * g_t
        c_cur = f_t * c_prev + delta_c

        # 2. Spatiotemporal memory update (zigzag flow)
        i_prime = torch.sigmoid(xi_m + mi)
        f_prime = torch.sigmoid(xf_m + mf)
        g_prime = torch.tanh(xg_m + mg)
        delta_m = i_prime * g_prime
        m_cur = f_prime * m_prev + delta_m

        # 3. Output gate & hidden state coupling
        o_t = torch.sigmoid(xo + ho + self.conv_c_o(c_cur) + self.conv_m_o(m_cur))
        h_cur = o_t * torch.tanh(self.conv_1x1(torch.cat([c_cur, m_cur], dim=1)))

        return h_cur, c_cur, m_cur, delta_c, delta_m


class PredRNNV2(nn.Module):
    """PredRNN-V2 architecture with Spatiotemporal LSTM cells and Memory-Decoupling.

    Architecture features:
      - Multi-layer ST-LSTM cells with zigzag spatiotemporal memory flow (M) across layers and time.
      - Horizontal temporal memory (C) within each layer.
      - Auxiliary memory-decoupling loss.
      - Reverse scheduled sampling support during training across both encoder and decoder stages.
    """

    has_decouple_loss: bool = True

    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 32,
        input_frames: int = 10,
        output_frames: int = 10,
        num_layers: int = 2,
        kernel_size: int = 3,
        decouple_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_layers = num_layers
        self.decouple_loss_weight = decouple_loss_weight

        cells = []
        for i in range(num_layers):
            c_in = image_channels if i == 0 else hidden_channels
            cells.append(SpatioTemporalLSTMCell(c_in, hidden_channels, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        # Output projection head: hidden_channels -> image_channels
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, image_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        future_frames: Optional[int] = None,
        mask_context: Optional[torch.Tensor] = None,
        mask_future: Optional[torch.Tensor] = None,
        return_decouple_loss: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive forward rollout with dual-phase scheduled sampling & decouple loss.

        Args:
            x: Input context frames of shape (B, T_in, C, H, W).
            target: Optional ground truth future frames of shape (B, T_out, C, H, W) for training.
            future_frames: Number of future frames to predict (defaults to self.output_frames).
            mask_context: Optional binary sampling mask for context phase of shape (B, T_in - 1).
                          1.0 = use ground truth, 0.0 = use model prediction.
            mask_future: Optional binary sampling mask for future phase of shape (B, T_out).
                         1.0 = use ground truth, 0.0 = use model prediction.
            return_decouple_loss: If True, computes and returns memory decoupling auxiliary loss.

        Returns:
            If return_decouple_loss is False:
                preds of shape (B, T_out, C, H, W)
            If return_decouple_loss is True:
                tuple of (preds, decouple_loss)
        """
        b, t_in, c, h, w = x.shape
        n_out = future_frames if future_frames is not None else self.output_frames

        # States for each layer: (h_l, c_l)
        h_states: List[Optional[torch.Tensor]] = [None] * self.num_layers
        c_states: List[Optional[torch.Tensor]] = [None] * self.num_layers

        # Global spatiotemporal memory m flows across layers and time
        m_state = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)

        delta_c_list: List[torch.Tensor] = []
        delta_m_list: List[torch.Tensor] = []

        # -------------------------------------------------------------
        # 1. Context / Encoder Phase (t = 0 to t_in - 1)
        # -------------------------------------------------------------
        last_pred: Optional[torch.Tensor] = None
        for t in range(t_in):
            if t == 0:
                cur_in = x[:, 0]
            else:
                # Apply reverse scheduled sampling mask to context frames if active during training
                if self.training and mask_context is not None and last_pred is not None:
                    # mask_context[:, t - 1]: (B,) -> (B, 1, 1, 1)
                    m_val = mask_context[:, t - 1].view(b, 1, 1, 1)
                    cur_in = m_val * x[:, t] + (1.0 - m_val) * last_pred
                else:
                    cur_in = x[:, t]

            for layer_idx, cell in enumerate(self.cells):
                layer_state = None if h_states[layer_idx] is None else (h_states[layer_idx], c_states[layer_idx])
                h_cur, c_cur, m_cur, dc, dm = cell(cur_in, layer_state, m_state)
                h_states[layer_idx] = h_cur
                c_states[layer_idx] = c_cur
                m_state = m_cur
                cur_in = h_cur

                if return_decouple_loss:
                    delta_c_list.append(dc)
                    delta_m_list.append(dm)

            last_pred = self.readout(h_states[-1])

        # -------------------------------------------------------------
        # 2. Future / Decoder Phase (step = 0 to n_out - 1)
        # -------------------------------------------------------------
        preds: List[torch.Tensor] = []

        for step in range(n_out):
            # Decide input for future prediction step
            if self.training and mask_future is not None and target is not None:
                # Ground truth candidate: x[:, -1] for step 0, or target[:, step - 1] for step > 0
                gt_candidate = target[:, step - 1] if step > 0 else x[:, -1]
                m_val = mask_future[:, step].view(b, 1, 1, 1)
                cur_in = m_val * gt_candidate + (1.0 - m_val) * last_pred
            else:
                # Validation / inference / pure autoregressive rollout
                cur_in = last_pred

            for layer_idx, cell in enumerate(self.cells):
                layer_state = (h_states[layer_idx], c_states[layer_idx])
                h_cur, c_cur, m_cur, dc, dm = cell(cur_in, layer_state, m_state)
                h_states[layer_idx] = h_cur
                c_states[layer_idx] = c_cur
                m_state = m_cur
                cur_in = h_cur

                if return_decouple_loss:
                    delta_c_list.append(dc)
                    delta_m_list.append(dm)

            pred_frame = self.readout(h_states[-1])
            preds.append(pred_frame)
            last_pred = pred_frame

        stacked_preds = torch.stack(preds, dim=1)

        if return_decouple_loss:
            decouple_loss = compute_memory_decoupling_loss(delta_c_list, delta_m_list)
            return stacked_preds, decouple_loss

        return stacked_preds
