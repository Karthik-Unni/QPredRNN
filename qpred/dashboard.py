"""QPredRNN Research Dashboard — Premium Edition.

Interactive evaluation, visualisation & experiment analysis for classical and
quantum-enhanced spatiotemporal video prediction.

Run with:
    streamlit run qpred/dashboard.py
"""
from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch

from qpred.data.dataset import FrameSequenceDataset
from qpred.metrics import compute_sequence_metrics, HAS_SKIMAGE
from qpred.models import build_model, QPredRNN, PredRNNV2, VideoConvLSTM

st.set_page_config(
    page_title="QPredRNN · Research Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/RijinRajeevan/QPredRNN",
        "Report a bug": "https://github.com/RijinRajeevan/QPredRNN/issues",
        "About": "QPredRNN — Quantum-Enhanced Spatiotemporal Video Prediction Research Dashboard",
    },
)

# ─── Premium Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #08090f; }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* Animated Gradient Hero */
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.qpred-hero {
  background: linear-gradient(135deg, #0f0c29, #302b63, #24243e, #0a0a2e, #1a0533, #0f0c29);
  background-size: 400% 400%;
  animation: gradientShift 12s ease infinite;
  border-radius: 16px;
  padding: 2.5rem 2.5rem 2rem 2.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(139, 92, 246, 0.25);
  box-shadow: 0 0 60px rgba(139, 92, 246, 0.12), 0 4px 32px rgba(0,0,0,0.6);
  position: relative; overflow: hidden;
}
.qpred-hero::before {
  content: ''; position: absolute; top: -40%; left: -20%;
  width: 60%; height: 200%;
  background: radial-gradient(ellipse, rgba(139,92,246,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title {
  font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em;
  background: linear-gradient(135deg, #e2d9f3 0%, #a78bfa 40%, #60a5fa 70%, #c4b5fd 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin: 0 0 0.35rem 0; line-height: 1.1;
}
.hero-sub { font-size: 1.05rem; color: #9ca3af; margin: 0 0 1.2rem 0; line-height: 1.5; }
.hero-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hbadge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.25rem 0.75rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}
.hb-quantum  { background: rgba(167,139,250,0.15); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.35); }
.hb-pennylane{ background: rgba(96,165,250,0.12);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.3); }
.hb-pytorch  { background: rgba(239,68,68,0.12);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.hb-kth      { background: rgba(16,185,129,0.12);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

/* Glassmorphism Cards */
.glass-card {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; padding: 1.25rem 1.5rem;
  backdrop-filter: blur(12px); transition: border-color 0.2s, box-shadow 0.2s;
}
.glass-card:hover { border-color: rgba(167,139,250,0.25); box-shadow: 0 0 20px rgba(139,92,246,0.08); }

/* Metric Tiles */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
.metric-tile {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px; padding: 1rem 0.75rem; text-align: center; transition: all 0.2s;
}
.metric-tile:hover { border-color: rgba(167,139,250,0.3); transform: translateY(-2px); }
.metric-val { font-size: 1.45rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #e2e8f0; line-height: 1.1; }
.metric-lbl { font-size: 0.7rem; color: #64748b; margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-delta { font-size: 0.74rem; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace; }
.delta-good { color: #34d399; } .delta-bad { color: #f87171; } .delta-neu { color: #94a3b8; }

/* Status Pills */
.status-pill {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.76rem; font-weight: 600;
}
.pill-quantum   { background: rgba(167,139,250,0.15); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.3); }
.pill-classical { background: rgba(96,165,250,0.12);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.25); }
.pill-training  { background: rgba(251,191,36,0.12);  color: #fcd34d; border: 1px solid rgba(251,191,36,0.3); }
.pill-complete  { background: rgba(52,211,153,0.12);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }

/* Training Progress Bar */
.train-progress-wrap { margin: 0.75rem 0; }
.train-progress-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.4rem; }
.train-progress-track { height: 6px; background: rgba(255,255,255,0.07); border-radius: 999px; overflow: hidden; }
.train-progress-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #7c3aed, #4f46e5, #2563eb);
  box-shadow: 0 0 10px rgba(124,58,237,0.5); transition: width 0.4s ease;
}

/* Quantum Circuit Card */
.qc-card {
  background: linear-gradient(135deg, rgba(109,40,217,0.12), rgba(37,99,235,0.08));
  border: 1px solid rgba(139,92,246,0.25); border-radius: 10px; padding: 1rem 1.25rem; margin-top: 0.75rem;
}
.qc-title { font-size: 0.82rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.6rem; }
.qc-row { display: flex; justify-content: space-between; padding: 0.2rem 0; font-size: 0.82rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
.qc-row:last-child { border-bottom: none; }
.qc-key { color: #94a3b8; } .qc-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* Section Headers */
.section-title { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin: 1.5rem 0 0.75rem 0; }
.section-divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.25rem 0; }

/* Comparison Table */
.comp-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.comp-table th { padding: 0.6rem 0.8rem; text-align: left; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #64748b; border-bottom: 1px solid rgba(255,255,255,0.08); }
.comp-table td { padding: 0.6rem 0.8rem; color: #cbd5e1; border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.comp-table tr:hover td { background: rgba(255,255,255,0.02); }
.comp-table td.model-name { color: #e2e8f0; font-weight: 600; font-family: 'Inter', sans-serif; font-size: 0.85rem; }
.td-good { color: #34d399 !important; font-weight: 700; } .td-mid { color: #fbbf24 !important; } .td-bad { color: #f87171 !important; }

/* Streamlit overrides */
div[data-testid="stTabs"] button { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.88rem !important; }
div[data-testid="stSidebar"] { background: #0d0f1a !important; }
.stSelectbox label, .stRadio label, .stSlider label, .stTextInput label { font-size: 0.82rem !important; font-weight: 600 !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.05em; }
button[kind="primary"] {
  background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
  border: none !important; font-weight: 700 !important;
  box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important; transition: all 0.2s !important;
}
button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important; }
button[kind="secondary"] { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.12) !important; color: #e2e8f0 !important; font-weight: 600 !important; }
code { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82em !important; background: rgba(255,255,255,0.06) !important; border-radius: 4px; padding: 0.1em 0.35em; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.35); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_checkpoint_model(ckpt_path: str, device_str: str) -> tuple[torch.nn.Module, dict]:
    """Load and cache model from checkpoint path."""
    device = torch.device(device_str)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        # Fallback config guessing if not present
        config = {
            "model": {"name": "predrnn_v2", "hidden_channels": 32, "num_layers": 2},
            "data": {"input_frames": 10, "output_frames": 10},
        }

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def scan_available_runs(runs_root: str = "runs") -> list[dict]:
    """Scan runs directory for checkpoints, metrics, and configs."""
    runs_dir = Path(runs_root)
    if not runs_dir.exists():
        return []

    available = []
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir():
            best_ckpt = p / "checkpoint_best.pt"
            last_ckpt = p / "checkpoint_last.pt"
            history_file = p / "history.json"
            config_file = p / "config.yaml"

            has_ckpt = best_ckpt.exists() or last_ckpt.exists()
            if has_ckpt or history_file.exists():
                available.append({
                    "name": p.name,
                    "path": p,
                    "best_ckpt": best_ckpt if best_ckpt.exists() else None,
                    "last_ckpt": last_ckpt if last_ckpt.exists() else None,
                    "history_file": history_file if history_file.exists() else None,
                    "config_file": config_file if config_file.exists() else None,
                })
    return available


def tensor_to_pil_strip(
    tensor: torch.Tensor,
    title: str = "",
    start_t: int = 1,
    cmap_name: str | None = None,
    border_color: tuple[int, int, int] | None = None,
    scale: int = 1,
) -> Image.Image:
    """Convert tensor (T, C, H, W) to horizontal labeled filmstrip image.

    Args:
        tensor: Video tensor of shape (T, C, H, W).
        title: Optional header text drawn above frames.
        start_t: Starting timestep label (default 1).
        cmap_name: Matplotlib colormap name (e.g. 'magma') for heatmaps.
        border_color: Optional RGB tuple for a 2-px border around the strip.
        scale: Integer upscale factor for small frames (default 1 = no scaling).
    """
    if tensor.numel() == 0:
        return Image.new("RGB", (200, 80), (20, 24, 33))

    tensor = tensor.detach().cpu()
    # Handle (C,H,W) single frame
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    t_steps, _c, h, w = tensor.shape
    h_s, w_s = h * scale, w * scale
    pad = 5
    header_h = 28
    label_h = 18
    total_w = t_steps * (w_s + pad) + pad
    total_h = h_s + header_h + label_h + pad * 2

    bg = (14, 16, 24)
    canvas = Image.new("RGB", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    # Optional border
    if border_color:
        draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=border_color, width=2)

    # Header title
    if title:
        draw.text((pad + 3, 6), title, fill=(180, 195, 225))

    tensor_np = tensor.numpy()

    for i in range(t_steps):
        frame_raw = np.clip(tensor_np[i, 0], 0.0, 1.0)
        if cmap_name:
            cm = plt.get_cmap(cmap_name)
            colored = (cm(frame_raw)[:, :, :3] * 255.0).astype(np.uint8)
            frame_im = Image.fromarray(colored, mode="RGB")
        else:
            frame_gray = (frame_raw * 255.0).astype(np.uint8)
            frame_im = Image.fromarray(frame_gray, mode="L").convert("RGB")

        if scale > 1:
            frame_im = frame_im.resize((w_s, h_s), Image.NEAREST)

        x_off = pad + i * (w_s + pad)
        y_off = header_h + pad
        canvas.paste(frame_im, (x_off, y_off))
        # Frame timestep label below image
        draw.text((x_off + 2, y_off + h_s + 2), f"t={start_t + i}", fill=(255, 215, 50))

    return canvas


def generate_synthetic_sample(
    num_frames: int = 20,
    size: int = 64,
    square_size: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a clean synthetic moving-square sequence for on-the-fly testing."""
    frames = []
    x, y = 8.0, 8.0
    vx, vy = 2.4, 3.1

    for _ in range(num_frames):
        img = np.zeros((size, size), dtype=np.float32)
        ix, iy = int(round(x)), int(round(y))
        img[iy : iy + square_size, ix : ix + square_size] = 1.0
        frames.append(img)

        x += vx
        y += vy
        if x <= 0 or x + square_size >= size:
            vx = -vx
            x = np.clip(x, 0, size - square_size)
        if y <= 0 or y + square_size >= size:
            vy = -vy
            y = np.clip(y, 0, size - square_size)

    seq = torch.from_numpy(np.stack(frames)[:, None, :, :]).float()  # (T, 1, H, W)
    half = num_frames // 2
    return seq[:half], seq[half:]


# ─── Helper utilities ────────────────────────────────────────────────────────

def read_live_history(run: dict) -> list[dict] | None:
    """Read history.json for a run; return None on parse failure."""
    if run.get("history_file") is None:
        return None
    try:
        return json.loads(run["history_file"].read_text(encoding="utf-8"))
    except Exception:
        return None


def render_progress_bar(current: int, total: int, label: str = "Training Progress") -> None:
    """Render an animated styled HTML progress bar."""
    pct = min(100.0, current / max(total, 1) * 100)
    fill_grad = "#059669,#10b981" if pct >= 100 else "#7c3aed,#4f46e5,#2563eb"
    st.markdown(f"""
    <div class="train-progress-wrap">
      <div class="train-progress-label">
        <span>{label}</span>
        <span style="color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-size:0.82rem">{current}/{total} ({pct:.1f}%)</span>
      </div>
      <div class="train-progress-track">
        <div class="train-progress-fill" style="width:{pct}%;background:linear-gradient(90deg,{fill_grad})"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_tiles(metrics: dict, baseline_mse: float | None = None) -> None:
    """Render premium glassmorphism metric tiles."""
    mse_delta = ""
    if baseline_mse is not None and metrics["mse"] > 0:
        rel = (metrics["mse"] - baseline_mse) / baseline_mse * 100
        cls = "delta-good" if rel < 0 else "delta-bad"
        mse_delta = f'<div class="metric-delta {cls}">{"+" if rel > 0 else ""}{rel:.1f}% vs persist.</div>'
    ssim_txt = f"{metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A"
    tiles = [("MSE", f"{metrics['mse']:.6f}", mse_delta),
             ("MAE", f"{metrics['mae']:.6f}", ""),
             ("PSNR", f"{metrics['psnr']:.2f} dB", ""),
             ("SSIM", ssim_txt, "")]
    html = '<div class="metric-grid">'
    for lbl, val, delta in tiles:
        html += f'<div class="metric-tile"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div>{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def styled_fig(nrows: int = 1, ncols: int = 2, figsize: tuple = (11, 3.8)) -> tuple:
    """Create a pre-styled dark matplotlib figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="#0d0f1a")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.flatten():
        ax.set_facecolor("#0d0f1a")
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.grid(color="#1e293b", linestyle="--", linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
    fig.tight_layout(pad=2.0)
    return fig, axes.flatten()


# ==============================================================================
# SIDEBAR: Checkpoint & Device Selection
# ==============================================================================
# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="qpred-hero">
  <div class="hero-title">⚛ QPredRNN Dashboard</div>
  <div class="hero-sub">
    Interactive research platform for classical and quantum-enhanced spatiotemporal video prediction.
    Compare ConvLSTM, PredRNN-V2, and QPredRNN with live metrics, error heatmaps, and per-horizon degradation curves.
  </div>
  <div class="hero-badges">
    <span class="hbadge hb-quantum">⚛ Quantum-Enhanced</span>
    <span class="hbadge hb-pennylane">PennyLane VQC</span>
    <span class="hbadge hb-pytorch">PyTorch</span>
    <span class="hbadge hb-kth">KTH Dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚛️ Control Panel")

    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    chosen_device = st.selectbox("Inference Device", device_options, index=0)
    if torch.cuda.is_available():
        st.caption(f"🟢 GPU: `{torch.cuda.get_device_name(0)}`")
    else:
        st.caption("🔵 CPU inference mode")

    st.markdown("---")
    st.markdown("#### 1. Select Model & Checkpoint")

    available_runs = scan_available_runs()
    run_names = [r["name"] for r in available_runs]

    selection_mode = st.radio("Checkpoint Source", ["Discovered Runs", "Manual File Path"])

    selected_ckpt_path = None
    if selection_mode == "Discovered Runs" and run_names:
        selected_run_name = st.selectbox("Available Experiments", run_names, index=0)
        selected_run = next(r for r in available_runs if r["name"] == selected_run_name)

        ckpt_type = st.radio("Checkpoint Type", ["best", "last"], horizontal=True)
        target_ckpt = selected_run["best_ckpt"] if ckpt_type == "best" else selected_run["last_ckpt"]
        if target_ckpt and target_ckpt.exists():
            selected_ckpt_path = str(target_ckpt)
        else:
            st.warning(f"No {ckpt_type} checkpoint found in {selected_run_name}")
    else:
        manual_path = st.text_input("Checkpoint Path (.pt)", value="runs/kth_predrnn_v2_overnight_gpu/checkpoint_best.pt")
        if Path(manual_path).exists():
            selected_ckpt_path = manual_path
        else:
            st.error("Specified checkpoint file does not exist.")

    # Model Status Card
    loaded_model = None
    loaded_checkpoint = None

    if selected_ckpt_path:
        try:
            loaded_model, loaded_checkpoint = load_checkpoint_model(selected_ckpt_path, chosen_device)
            st.success("Checkpoint Loaded Successfully")

            config = loaded_checkpoint.get("config", {})
            model_cfg = config.get("model", {})
            model_name = model_cfg.get("name", "Unknown")

            is_q = getattr(loaded_model, "is_quantum", False)
            badge_class = "badge-quantum" if is_q else "badge-classical"
            badge_text = "QUANTUM-ENHANCED (PennyLane VQC)" if is_q else "CLASSICAL BASELINE"

            st.markdown(f"<span class='status-badge {badge_class}'>{badge_text}</span>", unsafe_allow_html=True)

            total_params = sum(p.numel() for p in loaded_model.parameters())
            saved_ep = loaded_checkpoint.get("epoch", "N/A")
            saved_step = loaded_checkpoint.get("global_step", "N/A")
            best_vl = float(loaded_checkpoint.get("best_val_loss", loaded_checkpoint.get("val_loss", 0.0)))
            st.markdown(f"""
            - **Model Type**: `{model_name}`
            - **Parameters**: `{total_params:,}`
            - **Saved Epoch**: `{saved_ep}`
            - **Optimization Steps**: `{saved_step}`
            - **Best Val Loss**: `{best_vl:.6f}`
            """)

            if isinstance(saved_step, int) and saved_step <= 10:
                st.warning(
                    f"⚠️ **Undertrained Checkpoint ({saved_step} step{'s' if saved_step != 1 else ''})**\n\n"
                    "This checkpoint is from an early smoke test. The Sigmoid readout weights have not adapted "
                    "to dataset background statistics (~0.50 mid-grey). Output does not reflect learned dynamics."
                )

            if is_q:
                st.info(
                    f"🔬 **Quantum Circuit Configuration**:\n"
                    f"- Register: `{loaded_model.n_qubits} qubits`\n"
                    f"- Variational Depth: `{loaded_model.circuit_depth} layers`\n"
                    f"- Backend: `{loaded_model.quantum_backend}`\n\n"
                    f"*Note: Differentiable simulation on classical hardware. No physical quantum advantage claimed.*"
                )
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
    else:
        st.info("No checkpoint selected. Untrained/fresh models can still be previewed in evaluation tab.")


# ==============================================================================
# MAIN TABS
# ==============================================================================
tab_pred, tab_compare, tab_training, tab_metrics, tab_export = st.tabs([
    "🔮 Forecast Explorer",
    "⚖️ Model Comparison",
    "📈 Training Curves",
    "📊 Evaluation Metrics",
    "📄 Report & Export",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FORECAST EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    # ── Dataset / sequence selection ─────────────────────────────────────────
    st.markdown('<div class="section-title">🎬 Video Sequence Selection</div>', unsafe_allow_html=True)
    c_src, c_ctl = st.columns([4, 1])
    with c_src:
        data_source = st.radio(
            "Source",
            ["KTH Processed Dataset", "Synthetic Moving-Square", "Toy Dataset"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with c_ctl:
        scale_factor = st.selectbox("Frame Scale", [1, 2, 3, 4], index=1, help="Upscale small 64×64 frames for readability.")

    input_frames_cfg = 10
    output_frames_cfg = 10
    if loaded_model is not None:
        input_frames_cfg = getattr(loaded_model, "input_frames", 10)
        output_frames_cfg = getattr(loaded_model, "output_frames", 10)

    seq_x: torch.Tensor | None = None
    seq_y: torch.Tensor | None = None
    seq_desc = ""

    if data_source == "KTH Processed Dataset":
        kth_path = Path("data/processed/kth64")
        if kth_path.exists():
            col_sp, col_sl = st.columns([1, 3])
            with col_sp:
                split_choice = st.selectbox("Split", ["test", "val", "train"], index=0,
                    help="Use 'test' for the subject-disjoint held-out KTH test set.")
            try:
                ds = FrameSequenceDataset(
                    kth_path, split=split_choice,
                    input_frames=input_frames_cfg, output_frames=output_frames_cfg, stride=2,
                )
                with col_sl:
                    sample_idx = st.slider(
                        f"Clip Index (0–{len(ds)-1})", 0, max(0, len(ds) - 1), 0,
                        help="Each index is a 20-frame sliding window (stride=2) from a KTH sequence take.",
                    )
                seq_x, seq_y = ds[sample_idx]
                seq_dir, start_f, _ = ds.items[sample_idx]
                total_takes = len({item[0] for item in ds.items})
                seq_desc = f"KTH/{split_choice} clip #{sample_idx} · {seq_dir.name} · frames {start_f}–{start_f + input_frames_cfg + output_frames_cfg - 1}"
                st.caption(f"ℹ️ `{split_choice}` split: **{total_takes}** takes → **{len(ds):,}** temporal clips (stride=2)")
            except Exception as e:
                st.error(f"❌ KTH dataset load failed: {e}")
                with st.expander("Traceback", expanded=False):
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ `data/processed/kth64` not found. Select **Synthetic Moving-Square** or **Toy Dataset**."
                       "\n\nRun `python scripts/preprocess_kth.py` to generate the dataset.")

    elif data_source == "Toy Dataset":
        toy_path = Path("data/toy")
        if toy_path.exists():
            col_sp2, col_sl2 = st.columns([1, 3])
            with col_sp2:
                split_choice = st.selectbox("Split", ["train", "val", "test"], index=1)
            try:
                ds = FrameSequenceDataset(
                    toy_path, split=split_choice,
                    input_frames=input_frames_cfg, output_frames=output_frames_cfg, stride=1,
                )
                with col_sl2:
                    sample_idx = st.slider(f"Clip Index (0–{len(ds)-1})", 0, max(0, len(ds) - 1), 0)
                seq_x, seq_y = ds[sample_idx]
                seq_dir, start_f, _ = ds.items[sample_idx]
                seq_desc = f"Toy/{split_choice} clip #{sample_idx} · {seq_dir.name} · frame {start_f}"
                st.caption(f"ℹ️ `{split_choice}` split: **{len(ds)}** temporal clips")
            except Exception as e:
                st.error(f"❌ Toy dataset load failed: {e}")
        else:
            st.warning("⚠️ `data/toy` not found.")

    # Fall back to synthetic only if no real data was loaded
    if seq_x is None:
        if data_source == "KTH Processed Dataset":
            # Real dataset was requested but unavailable — do NOT silently swap to synthetic
            st.error("Cannot show KTH frames — dataset not available. Select 'Synthetic Moving-Square' to proceed.")
        else:
            seq_x, seq_y = generate_synthetic_sample(num_frames=input_frames_cfg + output_frames_cfg)
            seq_desc = "Synthetic Moving Square"

    if seq_x is not None:
        # ── Context filmstrip ────────────────────────────────────────────────
        st.markdown('<div class="section-title">👁 Observed Context Frames</div>', unsafe_allow_html=True)
        strip_in = tensor_to_pil_strip(
            seq_x, title=f"CONTEXT  ·  {seq_desc}",
            start_t=1, border_color=(60, 40, 140), scale=scale_factor,
        )
        st.image(strip_in, use_container_width=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Run prediction button ─────────────────────────────────────────────
        btn_col, info_col = st.columns([2, 3])
        with btn_col:
            btn_predict = st.button("🚀 Run Prediction Rollout", type="primary", use_container_width=True,
                                    help="Runs autoregressive inference through the selected model for all output frames.")
        with info_col:
            if loaded_model is None:
                st.warning("⚠️ No checkpoint loaded — prediction will use a random-init demo model.")
            else:
                ck_ep = loaded_checkpoint.get("epoch", "?") if loaded_checkpoint else "?"
                ck_best = loaded_checkpoint.get("best_val_loss", None) if loaded_checkpoint else None
                best_str = f" · best val MSE `{ck_best:.6f}`" if isinstance(ck_best, float) else ""
                st.info(f"✅ Loaded checkpoint epoch **{ck_ep}**{best_str}")

        if btn_predict:
            if loaded_model is None:
                st.warning("Using random-init demo model (no checkpoint selected).")
                config_dummy = {"model": {"name": "predrnn_v2", "hidden_channels": 8, "num_layers": 2,
                                          "input_frames": input_frames_cfg, "output_frames": output_frames_cfg}}
                model_eval = build_model(config_dummy).to(chosen_device)
            else:
                model_eval = loaded_model

            model_eval.eval()
            batch_in = seq_x.unsqueeze(0).to(chosen_device)

            pred_seq = None
            pred_error_msg = None
            try:
                t_start = time.perf_counter()
                with torch.no_grad():
                    pred_out = model_eval(batch_in)
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                pred_seq = pred_out[0].cpu()
            except Exception as e:
                pred_error_msg = str(e)
                pred_tb = traceback.format_exc()

            if pred_seq is None:
                st.error(f"❌ Prediction failed: {pred_error_msg}")
                with st.expander("🔍 Full Traceback", expanded=True):
                    st.code(pred_tb, language="python")
                st.info("Remaining dashboard sections (training curves, comparison, export) are unaffected.")
            else:
                # ── Metrics ─────────────────────────────────────────────────
                metrics = compute_sequence_metrics(pred_seq, seq_y)
                last_frame_baseline = seq_x[-1:].repeat(output_frames_cfg, 1, 1, 1)
                persistence_metrics = compute_sequence_metrics(last_frame_baseline, seq_y)
                mean_frame_baseline = seq_x.mean(dim=0, keepdim=True).repeat(output_frames_cfg, 1, 1, 1)
                mean_metrics = compute_sequence_metrics(mean_frame_baseline, seq_y)

                st.markdown('<div class="section-title">🎯 Rollout Performance</div>', unsafe_allow_html=True)
                render_metric_tiles(metrics, baseline_mse=persistence_metrics["mse"])
                st.caption(f"⚡ Inference latency: **{latency_ms:.1f} ms** · Output shape: `{tuple(pred_seq.shape)}`")

                # ── 3-Row filmstrip: Context | GT | Prediction ───────────────
                st.markdown('<div class="section-title">🎞 Full 30-Frame Sequence View</div>', unsafe_allow_html=True)
                strip_ctx = tensor_to_pil_strip(
                    seq_x, title="OBSERVED CONTEXT", start_t=1,
                    border_color=(60, 40, 140), scale=scale_factor,
                )
                strip_gt = tensor_to_pil_strip(
                    seq_y, title="GROUND TRUTH FUTURE",
                    start_t=input_frames_cfg + 1,
                    border_color=(30, 100, 70), scale=scale_factor,
                )
                strip_pred = tensor_to_pil_strip(
                    pred_seq, title="MODEL PREDICTION",
                    start_t=input_frames_cfg + 1,
                    border_color=(100, 40, 160), scale=scale_factor,
                )
                diff_seq = torch.abs(seq_y - pred_seq)
                strip_err = tensor_to_pil_strip(
                    diff_seq, title="|GT − PRED|  ERROR MAP (magma)",
                    start_t=input_frames_cfg + 1,
                    cmap_name="magma", scale=scale_factor,
                )
                for lbl, strip in [
                    ("Observed Context (input)", strip_ctx),
                    ("Ground Truth Future", strip_gt),
                    ("Model Prediction", strip_pred),
                    ("Absolute Error Heatmap", strip_err),
                ]:
                    st.image(strip, caption=lbl, use_container_width=True)

                # ── Frame-by-frame horizon inspector ────────────────────────
                st.markdown('<div class="section-title">🔭 Frame-by-Frame Horizon Inspector</div>', unsafe_allow_html=True)
                horizon_idx = st.slider(
                    "Prediction Horizon (future step)",
                    1, output_frames_cfg, 1,
                    help="Select which future step to inspect in detail. t=1 is the next frame after context.",
                )
                fi = horizon_idx - 1  # 0-indexed
                hcols = st.columns(4)
                # GT frame
                gt_frame_np = (seq_y[fi, 0].cpu().numpy() * 255).astype(np.uint8)
                gt_frame_img = Image.fromarray(gt_frame_np, mode="L").resize(
                    (gt_frame_np.shape[1] * 4, gt_frame_np.shape[0] * 4), Image.NEAREST
                )
                # Pred frame
                pr_frame_np = np.clip(pred_seq[fi, 0].cpu().numpy(), 0, 1)
                pr_frame_np_u8 = (pr_frame_np * 255).astype(np.uint8)
                pr_frame_img = Image.fromarray(pr_frame_np_u8, mode="L").resize(
                    (pr_frame_np_u8.shape[1] * 4, pr_frame_np_u8.shape[0] * 4), Image.NEAREST
                )
                # Error frame (magma)
                err_frame_np = np.clip(diff_seq[fi, 0].cpu().numpy(), 0, 1)
                cm_m = plt.get_cmap("magma")
                err_colored = (cm_m(err_frame_np)[:, :, :3] * 255).astype(np.uint8)
                err_frame_img = Image.fromarray(err_colored).resize(
                    (err_colored.shape[1] * 4, err_colored.shape[0] * 4), Image.NEAREST
                )
                # Last context frame
                ctx_last_np = (seq_x[-1, 0].cpu().numpy() * 255).astype(np.uint8)
                ctx_last_img = Image.fromarray(ctx_last_np, mode="L").resize(
                    (ctx_last_np.shape[1] * 4, ctx_last_np.shape[0] * 4), Image.NEAREST
                )
                with hcols[0]:
                    st.image(ctx_last_img, caption="Last Context Frame (t=10)", use_container_width=True)
                with hcols[1]:
                    st.image(gt_frame_img, caption=f"Ground Truth t={input_frames_cfg + horizon_idx}", use_container_width=True)
                with hcols[2]:
                    st.image(pr_frame_img, caption=f"Prediction t={input_frames_cfg + horizon_idx}", use_container_width=True)
                with hcols[3]:
                    st.image(err_frame_img, caption=f"Error Map t={input_frames_cfg + horizon_idx}", use_container_width=True)

                # Per-frame metrics for selected horizon
                pf_mse = metrics["per_frame"]["mse"][fi]
                pf_mae = metrics["per_frame"]["mae"][fi]
                pf_psnr = metrics["per_frame"]["psnr"][fi]
                pf_ssim = metrics["per_frame"]["ssim"][fi] if HAS_SKIMAGE else None
                hm1, hm2, hm3, hm4 = st.columns(4)
                hm1.metric("Frame MSE", f"{pf_mse:.6f}")
                hm2.metric("Frame MAE", f"{pf_mae:.6f}")
                hm3.metric("Frame PSNR", f"{pf_psnr:.2f} dB")
                hm4.metric("Frame SSIM", f"{pf_ssim:.4f}" if pf_ssim is not None else "N/A")

                # ── Tensor diagnostics ──────────────────────────────────────
                with st.expander("🔬 Tensor Diagnostics & Baseline Table", expanded=False):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.markdown(f"**Context x**\n- Shape: `{tuple(seq_x.shape)}`"
                                    f"\n- Range: `[{seq_x.min():.4f}, {seq_x.max():.4f}]`"
                                    f"\n- Std: `{seq_x.std():.4f}`")
                    with dc2:
                        st.markdown(f"**Target y**\n- Shape: `{tuple(seq_y.shape)}`"
                                    f"\n- Range: `[{seq_y.min():.4f}, {seq_y.max():.4f}]`"
                                    f"\n- Std: `{seq_y.std():.4f}`")
                    with dc3:
                        p_std = float(pred_seq.std())
                        flat_warn = " ⚠️ near-flat" if p_std < 0.01 else ""
                        st.markdown(f"**Prediction ŷ**{flat_warn}\n- Shape: `{tuple(pred_seq.shape)}`"
                                    f"\n- Range: `[{pred_seq.min():.4f}, {pred_seq.max():.4f}]`"
                                    f"\n- Std: `{p_std:.4f}`")
                    model_disp = getattr(model_eval, "model_name", "Model")
                    st.table([
                        {"Method": f"{model_disp}",
                         "MSE": f"{metrics['mse']:.6f}", "MAE": f"{metrics['mae']:.6f}",
                         "PSNR": f"{metrics['psnr']:.2f}",
                         "SSIM": f"{metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A"},
                        {"Method": "Persistence baseline",
                         "MSE": f"{persistence_metrics['mse']:.6f}", "MAE": f"{persistence_metrics['mae']:.6f}",
                         "PSNR": f"{persistence_metrics['psnr']:.2f}",
                         "SSIM": f"{persistence_metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A"},
                        {"Method": "Mean-context baseline",
                         "MSE": f"{mean_metrics['mse']:.6f}", "MAE": f"{mean_metrics['mae']:.6f}",
                         "PSNR": f"{mean_metrics['psnr']:.2f}",
                         "SSIM": f"{mean_metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A"},
                    ])

                # ── Degradation curves ───────────────────────────────────────
                st.markdown('<div class="section-title">📉 Per-Frame Degradation Curves</div>', unsafe_allow_html=True)
                fig_tab1, axes_tab1 = styled_fig(1, 2, figsize=(11, 3.8))
                horizons = list(range(1, output_frames_cfg + 1))
                # Highlight the selected horizon
                axes_tab1[0].plot(horizons, metrics["per_frame"]["mse"], marker="o", color="#f87171",
                                  linewidth=2, label="MSE", markersize=5)
                axes_tab1[0].plot(horizons, metrics["per_frame"]["mae"], marker="s", color="#fbbf24",
                                  linewidth=2, label="MAE", markersize=5)
                axes_tab1[0].axvline(x=horizon_idx, color="#a78bfa", linestyle="--", linewidth=1.5,
                                     label=f"Selected (t={input_frames_cfg + horizon_idx})")
                axes_tab1[0].set_title("Error vs Horizon", color="#e2e8f0", fontsize=10, fontweight="bold")
                axes_tab1[0].set_xlabel("Future Step", color="#64748b", fontsize=9)
                axes_tab1[0].legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
                axes_tab1[1].plot(horizons, metrics["per_frame"]["psnr"], marker="o", color="#60a5fa",
                                  linewidth=2, label="PSNR (dB)", markersize=5)
                if HAS_SKIMAGE:
                    axes_tab1[1].plot(horizons, metrics["per_frame"]["ssim"], marker="^", color="#34d399",
                                      linewidth=2, label="SSIM", markersize=5)
                axes_tab1[1].axvline(x=horizon_idx, color="#a78bfa", linestyle="--", linewidth=1.5)
                axes_tab1[1].set_title("Quality vs Horizon", color="#e2e8f0", fontsize=10, fontweight="bold")
                axes_tab1[1].set_xlabel("Future Step", color="#64748b", fontsize=9)
                axes_tab1[1].legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
                st.pyplot(fig_tab1)
                plt.close(fig_tab1)

                # Persist to session state for export tab
                st.session_state["last_metrics"] = metrics
                st.session_state["last_ckpt"] = selected_ckpt_path
                st.session_state["last_latency_ms"] = latency_ms
                st.session_state["last_seq_desc"] = seq_desc


# ------------------------------------------------------------------------------
# TAB 2: Multi-Model Comparison
# ------------------------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="section-title">⚖️ Side-by-Side Model Comparison</div>', unsafe_allow_html=True)
    st.caption("Compare ConvLSTM, PredRNN-V2, and QPredRNN on the same input sequence.")

    suite_choice = st.radio(
        "Benchmark Suite",
        ["TPU v5e-1 · 50-Epoch Final", "Local GPU · Pilot Runs", "Custom Paths"],
        horizontal=True,
    )

    if suite_choice == "TPU v5e-1 · 50-Epoch Final":
        default_m1 = "runs/kth_classical_tpu_50ep/checkpoint_best.pt"
        default_m2 = "runs/kth_predrnn_v2_tpu_50ep/checkpoint_best.pt"
        default_m3 = "runs/kth_qpredrnn_tpu_50ep/checkpoint_best.pt"
        st.info("🚀 **TPU v5e-1 50-Epoch Benchmark** — Import with `python scripts/import_colab_results.py --source-dir <dir>`.")
    elif suite_choice == "Local GPU · Pilot Runs":
        default_m1 = "runs/kth_classical_gpu/checkpoint_best.pt"
        default_m2 = "runs/kth_predrnn_v2_overnight_gpu/checkpoint_best.pt"
        default_m3 = "runs/kth_qpredrnn_pilot_gpu/checkpoint_best.pt"
        st.warning("⚠️ **Unmatched pilot budgets.** ConvLSTM=1ep, PredRNN-V2=10ep, QPredRNN=10ep. Results are not directly comparable.")
    else:
        default_m1 = "runs/kth_classical_tpu_50ep/checkpoint_best.pt"
        default_m2 = "runs/kth_predrnn_v2_tpu_50ep/checkpoint_best.pt"
        default_m3 = "runs/kth_qpredrnn_tpu_50ep/checkpoint_best.pt"

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("**🔵 ConvLSTM**")
        ckpt_m1 = st.text_input("ConvLSTM Path", value=default_m1, key="ckpt_m1")
    with col_m2:
        st.markdown("**🟢 PredRNN-V2**")
        ckpt_m2 = st.text_input("PredRNN-V2 Path", value=default_m2, key="ckpt_m2")
    with col_m3:
        st.markdown("**🟣 QPredRNN**")
        ckpt_m3 = st.text_input("QPredRNN Path", value=default_m3, key="ckpt_m3")

    if st.button("▶ Run Comparative Evaluation", type="secondary"):
        models_to_test = [
            ("ConvLSTM", ckpt_m1, "#3b82f6"),
            ("PredRNN-V2", ckpt_m2, "#10b981"),
            ("QPredRNN", ckpt_m3, "#a78bfa"),
        ]

        batch_in = seq_x.unsqueeze(0).to(chosen_device)
        comp_results = []
        prog = st.progress(0, text="Loading models…")

        for _ci, (name, ckpt_p, color) in enumerate(models_to_test):
            prog.progress((_ci + 1) / len(models_to_test), text=f"Evaluating {name}…")
            if Path(ckpt_p).exists():
                try:
                    m, ck = load_checkpoint_model(ckpt_p, chosen_device)
                    cfg = ck.get("config", {})
                    saved_ep = ck.get("epoch", "N/A")
                    saved_st = ck.get("global_step", "N/A")
                    target_ep = cfg.get("training", {}).get("epochs")
                    if isinstance(saved_st, int) and saved_st <= 10:
                        train_status = f"⚠️ Smoke ({saved_st}st)"
                    elif isinstance(saved_ep, int) and target_ep and saved_ep < target_ep:
                        train_status = f"⚠️ Partial ({saved_ep}/{target_ep}ep)"
                    elif isinstance(saved_ep, int) and saved_ep >= 50:
                        train_status = "✅ Full 50ep"
                    elif isinstance(saved_ep, int) and saved_ep >= 10:
                        train_status = f"🔵 Pilot ({saved_ep}ep)"
                    else:
                        train_status = f"Epoch {saved_ep}"
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        preds = m(batch_in)[0].cpu()
                    lat = (time.perf_counter() - t0) * 1000.0
                    met = compute_sequence_metrics(preds, seq_y)
                    comp_results.append({
                        "Model": name, "Color": color, "Status": train_status,
                        "Epoch": saved_ep, "Steps": saved_st, "Path": ckpt_p,
                        "MSE": met["mse"], "MAE": met["mae"],
                        "PSNR": met["psnr"], "SSIM": met["ssim"],
                        "Latency": lat, "Preds": preds, "per_frame": met["per_frame"],
                    })
                except Exception as e:
                    comp_results.append({"Model": name, "Color": color, "Error": str(e), "MSE": None})
            else:
                is_tpu = "tpu_50ep" in ckpt_p
                comp_results.append({
                    "Model": name, "Color": color,
                    "Status": "🕐 Awaiting TPU Import" if is_tpu else "❌ Not Found",
                    "MSE": None, "Path": ckpt_p,
                })
        prog.empty()

        st.markdown('<div class="section-title">📊 Quantitative Results</div>', unsafe_allow_html=True)
        valid_mses = [r["MSE"] for r in comp_results if r.get("MSE") is not None]
        best_mse = min(valid_mses) if valid_mses else None
        tbl = '<table class="comp-table"><tr><th>Model</th><th>Status</th><th>Epoch</th><th>MSE ↓</th><th>MAE ↓</th><th>PSNR ↑</th><th>SSIM ↑</th><th>Latency</th></tr>'
        for r in comp_results:
            if r.get("MSE") is not None:
                mse_cls = "td-good" if r["MSE"] == best_mse else ""
                ssim_v = f"{r['SSIM']:.4f}" if isinstance(r.get("SSIM"), float) else str(r.get("SSIM", "N/A"))
                tbl += f'<tr><td class="model-name">{r["Model"]}</td><td>{r.get("Status","")}</td><td>{r.get("Epoch","N/A")}</td><td class="{mse_cls}">{r["MSE"]:.6f}</td><td>{r["MAE"]:.6f}</td><td>{r["PSNR"]:.2f}</td><td>{ssim_v}</td><td>{r["Latency"]:.1f} ms</td></tr>'
            else:
                err = r.get("Error", r.get("Status", "N/A"))
                tbl += f'<tr><td class="model-name">{r["Model"]}</td><td colspan="7" style="color:#64748b">{err}</td></tr>'
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎞 Visual Output Comparison</div>', unsafe_allow_html=True)
        strip_gt_comp = tensor_to_pil_strip(seq_y, title="GROUND TRUTH (Reference)")
        st.image(strip_gt_comp, caption="Ground Truth Future", use_container_width=True)
        for r in comp_results:
            if "Preds" in r:
                strip = tensor_to_pil_strip(r["Preds"], title=f"{r['Model']} — {r.get('Status','')}")
                st.image(strip, caption=r["Model"], use_container_width=True)

        valid_models = [r for r in comp_results if "per_frame" in r]
        if valid_models:
            st.markdown('<div class="section-title">📉 Horizon Degradation</div>', unsafe_allow_html=True)
            fig_comp, axes_comp = styled_fig(1, 2, figsize=(11, 3.8))
            for r in valid_models:
                c = r["Color"]
                horizons = list(range(1, len(r["per_frame"]["mse"]) + 1))
                axes_comp[0].plot(horizons, r["per_frame"]["mse"], marker="o", label=r["Model"], color=c, linewidth=2, markersize=4)
                if HAS_SKIMAGE and "ssim" in r["per_frame"]:
                    axes_comp[1].plot(horizons, r["per_frame"]["ssim"], marker="^", label=r["Model"], color=c, linewidth=2, markersize=4)
            axes_comp[0].set_title("MSE vs Horizon (↓ Better)", color="#e2e8f0", fontsize=10, fontweight="bold")
            axes_comp[0].set_xlabel("Horizon Step", color="#64748b", fontsize=9)
            axes_comp[0].legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
            axes_comp[1].set_title("SSIM vs Horizon (↑ Better)", color="#e2e8f0", fontsize=10, fontweight="bold")
            axes_comp[1].set_xlabel("Horizon Step", color="#64748b", fontsize=9)
            axes_comp[1].legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
            st.pyplot(fig_comp)
            plt.close(fig_comp)


# ------------------------------------------------------------------------------
# TAB 3: Training History & Curves
# ------------------------------------------------------------------------------
with tab_training:
    st.markdown('<div class="section-title">📈 Training Convergence Curves</div>', unsafe_allow_html=True)

    available_runs_t3 = scan_available_runs()
    runs_with_history = [r for r in available_runs_t3 if r["history_file"] is not None]

    if not runs_with_history:
        st.info("No runs with history.json discovered in runs/ directory.")
    else:
        run_names_hist = [r["name"] for r in runs_with_history]
        selected_compare_runs = st.multiselect(
            "Select Runs to Plot",
            run_names_hist,
            default=run_names_hist[:min(4, len(run_names_hist))],
        )

        if selected_compare_runs:
            _col1, _col2, _col3 = st.columns(3)
            show_train = _col1.checkbox("Show training loss", value=True)
            show_val = _col2.checkbox("Show validation loss", value=True)
            log_scale = _col3.checkbox("Log Y-axis", value=False)

            fig_loss, ax_loss = plt.subplots(figsize=(11, 4.5), facecolor="#0d0f1a")
            ax_loss.set_facecolor("#0d0f1a")
            ax_loss.tick_params(colors="#64748b", labelsize=8)
            ax_loss.grid(color="#1e293b", linestyle="--", linewidth=0.8, alpha=0.7)
            for spine in ax_loss.spines.values():
                spine.set_color("#1e293b")

            palette = ["#3b82f6", "#10b981", "#a78bfa", "#f59e0b", "#ef4444", "#22d3ee"]

            for i, r_name in enumerate(selected_compare_runs):
                run_obj = next(r for r in runs_with_history if r["name"] == r_name)
                hist_data = read_live_history(run_obj)
                if hist_data is None:
                    st.warning(f"Failed to read history for `{r_name}`")
                    continue
                ep_list = [h["epoch"] for h in hist_data]
                train_loss = [h["train_loss"] for h in hist_data]
                val_loss = [h["val_loss"] for h in hist_data]
                c = palette[i % len(palette)]
                if show_val:
                    ax_loss.plot(ep_list, val_loss, marker="o", color=c, linewidth=2, markersize=4, label=f"{r_name} (val)", zorder=3)
                if show_train:
                    ax_loss.plot(ep_list, train_loss, linestyle="--", marker="s", color=c, linewidth=1.5, markersize=3, alpha=0.6, label=f"{r_name} (train)")
                # Mark best val epoch
                best_h = min(hist_data, key=lambda h: h["val_loss"])
                ax_loss.axvline(x=best_h["epoch"], color=c, linestyle=":", alpha=0.4, linewidth=1)
                ax_loss.annotate(f"best\n{best_h['val_loss']:.5f}",
                    xy=(best_h["epoch"], best_h["val_loss"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=c, fontfamily="monospace")

            if log_scale:
                ax_loss.set_yscale("log")
            ax_loss.set_title("Training & Validation MSE Convergence", color="#e2e8f0", fontsize=11, fontweight="bold")
            ax_loss.set_xlabel("Epoch", color="#64748b", fontsize=9)
            ax_loss.set_ylabel("MSE Loss" + (" (log)" if log_scale else ""), color="#64748b", fontsize=9)
            ax_loss.legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=8, loc="upper right")
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            # Run summary table
            st.markdown('<div class="section-title">📋 Run Summary</div>', unsafe_allow_html=True)
            summary_rows = []
            for r_name in selected_compare_runs:
                run_obj = next(r for r in runs_with_history if r["name"] == r_name)
                hist_data = read_live_history(run_obj)
                if not hist_data:
                    continue
                best_h = min(hist_data, key=lambda h: h["val_loss"])
                last_h = hist_data[-1]
                summary_rows.append({
                    "Run": r_name,
                    "Epochs Logged": len(hist_data),
                    "Best Val MSE": f"{best_h['val_loss']:.6f}",
                    "Best Epoch": best_h["epoch"],
                    "Last Train MSE": f"{last_h['train_loss']:.6f}",
                    "Last Val MSE": f"{last_h['val_loss']:.6f}",
                })
            if summary_rows:
                st.table(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown('<div class="section-title">📊 Checkpoint & Run Evaluation Metrics</div>', unsafe_allow_html=True)

    # Show last computed rollout metrics if available
    last_metrics = st.session_state.get("last_metrics")
    if last_metrics:
        last_ckpt = st.session_state.get("last_ckpt", "N/A")
        last_desc = st.session_state.get("last_seq_desc", "N/A")
        st.success(f"**Last rollout:** `{last_ckpt}` on `{last_desc}`")
        render_metric_tiles(last_metrics)

        # Per-horizon bar chart
        pf = last_metrics.get("per_frame", {})
        if pf.get("mse"):
            st.markdown('<div class="section-title">Per-Horizon Metric Bar Chart</div>', unsafe_allow_html=True)
            fig_bar, ax_bar = plt.subplots(figsize=(10, 3.5), facecolor="#0d0f1a")
            ax_bar.set_facecolor("#0d0f1a")
            ax_bar.tick_params(colors="#64748b", labelsize=8)
            ax_bar.grid(color="#1e293b", linestyle="--", linewidth=0.8, alpha=0.7)
            for spine in ax_bar.spines.values():
                spine.set_color("#1e293b")
            xp = range(1, len(pf["mse"]) + 1)
            ax_bar.bar([x - 0.2 for x in xp], pf["mse"], width=0.35, color="#f87171", alpha=0.85, label="MSE")
            ax_bar.bar([x + 0.2 for x in xp], pf["mae"], width=0.35, color="#fbbf24", alpha=0.85, label="MAE")
            ax_bar.set_xlabel("Future Step (Horizon)", color="#64748b", fontsize=9)
            ax_bar.set_title("Per-Frame MSE & MAE", color="#e2e8f0", fontsize=10, fontweight="bold")
            ax_bar.legend(facecolor="#0d0f1a", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # Raw per-frame table
            with st.expander("Raw Per-Frame Metric Table", expanded=False):
                rows = []
                for i, (m, a, p) in enumerate(zip(pf["mse"], pf["mae"], pf["psnr"])):
                    row = {"Horizon t": i + 1, "MSE": f"{m:.6f}", "MAE": f"{a:.6f}", "PSNR": f"{p:.2f} dB"}
                    if HAS_SKIMAGE and pf.get("ssim"):
                        row["SSIM"] = f"{pf['ssim'][i]:.4f}"
                    rows.append(row)
                st.table(rows)
    else:
        st.info("Run a **Forecast Explorer** rollout first — results will appear here.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">All Runs — Training Summary</div>', unsafe_allow_html=True)
    _all_runs_m = scan_available_runs()
    if _all_runs_m:
        mrows = []
        for _r in _all_runs_m:
            _h = read_live_history(_r) if _r["history_file"] else None
            best_val, best_ep, last_ep, steps = "—", "—", "—", "—"
            status = "⚠️ No history"
            if _h:
                _bh = min(_h, key=lambda h: h["val_loss"])
                best_val = f"{_bh['val_loss']:.6f}"
                best_ep = str(_bh["epoch"])
                last_ep = str(_h[-1]["epoch"])
                steps = f"{_h[-1].get('global_step', '?'):,}" if isinstance(_h[-1].get("global_step"), int) else "?"
                status = "✅ Complete (50ep)" if _h[-1]["epoch"] >= 50 else f"🔵 {_h[-1]['epoch']}/50 ep"
            mrows.append({
                "Run": _r["name"],
                "Status": status,
                "Best Val MSE": best_val,
                "Best Epoch": best_ep,
                "Last Epoch": last_ep,
                "Total Steps": steps,
                "Best Ckpt": "✓" if _r["best_ckpt"] else "✗",
                "Last Ckpt": "✓" if _r["last_ckpt"] else "✗",
            })
        st.table(mrows)
    else:
        st.info("No runs found in `runs/`.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — REPORT & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-title">📄 Experiment Report Generator</div>', unsafe_allow_html=True)
    st.caption("Generate reproducible Markdown summaries with checkpoint metadata, metrics, and configuration snapshots.")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        report_title = st.text_input("Report Title", value="QPredRNN — Video Prediction Evaluation Report")
        notes = st.text_area("Research Notes", placeholder="Add observations, hypotheses, or conclusions…",
                             value="Pilot evaluation comparing classical and quantum-enhanced spatiotemporal predictors.", height=100)
    with col_r2:
        _rdev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.markdown(f"""
        <div class="glass-card" style="margin-top:1.6rem">
          <div style="font-size:0.7rem;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem">Metadata</div>
          <div class="qc-row"><span class="qc-key">Timestamp</span><span class="qc-val">{time.strftime('%Y-%m-%d %H:%M')}</span></div>
          <div class="qc-row"><span class="qc-key">Platform</span><span class="qc-val">{_rdev}</span></div>
          <div class="qc-row"><span class="qc-key">PyTorch</span><span class="qc-val">{torch.__version__}</span></div>
          <div class="qc-row"><span class="qc-key">Device</span><span class="qc-val">{chosen_device}</span></div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("📝 Generate Markdown Report", type="primary"):
        ckpt_label = st.session_state.get("last_ckpt") or selected_ckpt_path or "None"
        metrics_dict = st.session_state.get("last_metrics", {})
        latency = st.session_state.get("last_latency_ms", "N/A")
        model_type = "Quantum-Enhanced (PennyLane VQC)" if (loaded_model and getattr(loaded_model, "is_quantum", False)) else "Classical Baseline"
        qc_section = ""
        if loaded_model and getattr(loaded_model, "is_quantum", False):
            qc_section = f"""
## 3. Quantum Circuit Configuration
| Property | Value |
| :--- | :--- |
| Qubits | {loaded_model.n_qubits} |
| VQC Depth | {loaded_model.circuit_depth} layers |
| Backend | `{loaded_model.quantum_backend}` |
| Embedding | AngleEmbedding (RY rotations) |
| Entangler | BasicEntanglerLayers + CNOT ring |
| Measurement | Pauli-Z expectation ⟨Z_i⟩ |

> **Disclaimer**: All quantum circuits are simulated on classical hardware via PennyLane `default.qubit`. No physical quantum advantage claimed.
"""
        _params_str = f"{sum(p.numel() for p in loaded_model.parameters()):,}" if loaded_model else "N/A"
        report_md = f"""# {report_title}

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Platform**: {_rdev}
**Device**: `{chosen_device}`
**PyTorch**: `{torch.__version__}`

---

## 1. Checkpoint & Model
| Field | Value |
| :--- | :--- |
| Checkpoint | `{ckpt_label}` |
| Model Type | {model_type} |
| Saved Epoch | {loaded_checkpoint.get('epoch', 'N/A') if loaded_checkpoint else 'N/A'} |
| Global Steps | {loaded_checkpoint.get('global_step', 'N/A') if loaded_checkpoint else 'N/A'} |
| Parameters | {_params_str} |
| Best Val Loss | {loaded_checkpoint.get('best_val_loss', 'N/A') if loaded_checkpoint else 'N/A'} |

## 2. Quantitative Evaluation
| Metric | Value |
| :--- | :--- |
| MSE | {metrics_dict.get('mse', 'N/A')} |
| MAE | {metrics_dict.get('mae', 'N/A')} |
| PSNR (dB) | {metrics_dict.get('psnr', 'N/A')} |
| SSIM | {metrics_dict.get('ssim', 'N/A')} |
| Inference Latency | {f"{latency:.1f} ms" if isinstance(latency, float) else latency} |
{qc_section}
## 4. Observations & Notes

{notes}

---
*Generated by QPredRNN Research Dashboard*
"""
        st.text_area("Generated Report (Markdown)", value=report_md, height=320)
        st.download_button(
            label="📥 Download as Markdown",
            data=report_md,
            file_name=f"qpredrnn_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗄 All Discovered Runs</div>', unsafe_allow_html=True)
    _all_runs = scan_available_runs()
    if _all_runs:
        _run_table = []
        for _r in _all_runs:
            _h = read_live_history(_r) if _r["history_file"] else None
            _best_val = "-"
            _last_ep = "-"
            if _h:
                _best_h = min(_h, key=lambda h: h["val_loss"])
                _best_val = f"{_best_h['val_loss']:.6f} (ep {_best_h['epoch']})"
                _last_ep = str(_h[-1]["epoch"])
            _run_table.append({
                "Run": _r["name"],
                "Best Ckpt": "✓" if _r["best_ckpt"] else "✗",
                "Last Ckpt": "✓" if _r["last_ckpt"] else "✗",
                "History": "✓" if _r["history_file"] else "✗",
                "Last Epoch": _last_ep,
                "Best Val MSE": _best_val,
            })
        st.table(_run_table)
    else:
        st.info("No runs found in `runs/`.")
