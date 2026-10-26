from __future__ import annotations
import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    qml = None
    HAS_PENNYLANE = False

from .predrnn_v2 import SpatioTemporalLSTMCell


def create_quantum_vqc_layer(
    n_qubits: int = 4,
    circuit_depth: int = 2,
    backend: str = "default.qubit",
) -> nn.Module:
    """Instantiate a differentiable Parameterized Quantum Circuit (PQC) layer using PennyLane.

    Architecture:
        - AngleEmbedding with Y-axis rotations: theta_i in [-pi/2, pi/2].
        - BasicEntanglerLayers: parameterized RX rotations followed by a closed ring of CNOT gates.
        - PauliZ expectation value measurements on each qubit wire: <Z_i> in [-1, 1].
    """
    if not HAS_PENNYLANE:
        raise ImportError(
            "PennyLane is required for QPredRNN quantum components. "
            "Install it via `pip install pennylane`."
        )

    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (circuit_depth, n_qubits)}
    layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    return layer


class QuantumContextModulator(nn.Module):
    """Hybrid Quantum-Classical Bottleneck for Spatiotemporal Memory Modulation.

    Compresses spatial features into a compact N-qubit register, applies variational quantum
    entanglement, and maps quantum expectation values back to modulate memory channels.
    """

    def __init__(
        self,
        channels: int,
        n_qubits: int = 4,
        circuit_depth: int = 2,
        enabled: bool = True,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.channels = channels
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.enabled = enabled
        self.backend = backend

        if self.enabled:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.encoder = nn.Linear(channels, n_qubits)
            self.vqc = create_quantum_vqc_layer(n_qubits, circuit_depth, backend)
            self.decoder = nn.Linear(n_qubits, channels)
            self.gate = nn.Sequential(
                nn.Linear(n_qubits, channels),
                nn.Sigmoid(),
            )

    def forward(self, m_state: torch.Tensor) -> torch.Tensor:
        """Modulate spatiotemporal memory m_state (B, C, H, W)."""
        if not self.enabled:
            return m_state

        orig_device = m_state.device
        b, c, h, w = m_state.shape

        # 1. Pool spatial dimensions: (B, C, 1, 1) -> (B, C)
        z = self.pool(m_state).view(b, c)

        # 2. Encode to quantum angles: (B, n_qubits) scaled to [-pi/2, pi/2]
        angles = torch.tanh(self.encoder(z)) * (math.pi / 2.0)

        # 3. Parameterized quantum circuit evaluation (handling device compatibility)
        if orig_device.type == "cuda":
            angles_cpu = angles.cpu()
            q_out_cpu = self.vqc(angles_cpu)
            q_out = q_out_cpu.to(orig_device)
        else:
            q_out = self.vqc(angles)

        # 4. Decode to memory space and compute residual update
        delta_m = self.decoder(q_out).view(b, c, 1, 1)
        gate_m = self.gate(q_out).view(b, c, 1, 1)

        # Gated quantum residual modulation
        return m_state + gate_m * delta_m


class QPredRNN(nn.Module):
    """Quantum-Enhanced Predictive Recurrent Neural Network (QPredRNN).

    Combines SpatioTemporal LSTM cells (dual temporal and spatiotemporal memory flows)
    with a parameterized quantum circuit bottleneck that modulates global spatiotemporal memory.
    """

    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 32,
        input_frames: int = 10,
        output_frames: int = 10,
        num_layers: int = 2,
        kernel_size: int = 3,
        decouple_loss_weight: float = 0.1,
        quantum_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.decouple_loss_weight = decouple_loss_weight
        self.has_decouple_loss = True
        self.is_quantum = True

        q_cfg = quantum_cfg or {}
        self.quantum_enabled = q_cfg.get("enabled", True)
        self.n_qubits = int(q_cfg.get("n_qubits", 4))
        self.circuit_depth = int(q_cfg.get("circuit_depth", 2))
        self.quantum_backend = str(q_cfg.get("backend", "default.qubit"))

        # SpatioTemporal LSTM Cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = image_channels if i == 0 else hidden_channels
            self.cells.append(
                SpatioTemporalLSTMCell(
                    in_channels=in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                )
            )

        # Quantum Context Modulator
        self.quantum_modulator = QuantumContextModulator(
            channels=hidden_channels,
            n_qubits=self.n_qubits,
            circuit_depth=self.circuit_depth,
            enabled=self.quantum_enabled,
            backend=self.quantum_backend,
        )

        # Readout Conv to regenerate video frames
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
        """Forward pass with quantum-modulated recurrent propagation.

        Inference mode (model.eval() or target=None):
            Zero ground-truth leakage: purely autoregressive rollout from previous predictions.
        Training mode with RSS:
            Applies mask_context to context steps and mask_future to future rollout.
        """
        b, t_in, c, h, w = x.shape
        n_out = future_frames if future_frames is not None else (target.size(1) if target is not None else self.output_frames)

        h_states: List[Optional[torch.Tensor]] = [None] * self.num_layers
        c_states: List[Optional[torch.Tensor]] = [None] * self.num_layers
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
                if self.training and mask_context is not None and last_pred is not None:
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

            # Quantum Modulation of SpatioTemporal Memory
            m_state = self.quantum_modulator(m_state)
            last_pred = self.readout(h_states[-1])

        # -------------------------------------------------------------
        # 2. Future / Decoder Phase (step = 0 to n_out - 1)
        # -------------------------------------------------------------
        preds: List[torch.Tensor] = []

        for step in range(n_out):
            if self.training and mask_future is not None and target is not None:
                gt_candidate = target[:, step - 1] if step > 0 else x[:, -1]
                m_val = mask_future[:, step].view(b, 1, 1, 1)
                cur_in = m_val * gt_candidate + (1.0 - m_val) * last_pred
            else:
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

            # Quantum Modulation of SpatioTemporal Memory
            m_state = self.quantum_modulator(m_state)
            pred = self.readout(h_states[-1])
            preds.append(pred)
            last_pred = pred

        pred_tensor = torch.stack(preds, dim=1)

        if return_decouple_loss:
            from .predrnn_v2 import compute_memory_decoupling_loss
            decouple_loss = compute_memory_decoupling_loss(delta_c_list, delta_m_list)
            return pred_tensor, decouple_loss

        return pred_tensor


QuantumContextModel = QPredRNN
