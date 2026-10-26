"""QPredRNN Research Dashboard: Interactive Evaluation, Visualization & Experiment Analysis.

Run with:
    streamlit run qpred/dashboard.py
"""
from __future__ import annotations
import io
import json
import time
from pathlib import Path
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
    page_title="QPredRNN Research Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, modern scientific UI
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #4F8BF9; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #8892b0; margin-bottom: 1.5rem; }
    .status-badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-quantum { background: rgba(138, 43, 226, 0.2); color: #c084fc; border: 1px solid rgba(138, 43, 226, 0.4); }
    .badge-classical { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .metric-card { background: #1e222d; border-radius: 8px; padding: 1rem; border: 1px solid #2d3748; }
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
) -> Image.Image:
    """Convert tensor (T, C, H, W) to horizontal labeled image strip."""
    t_steps, c, h, w = tensor.shape
    pad = 4
    header_h = 24
    total_w = t_steps * (w + pad) + pad
    total_h = h + header_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (20, 24, 33))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad + 2, 4), title, fill=(200, 215, 240))

    tensor_np = tensor.detach().cpu().numpy()

    for i in range(t_steps):
        frame_raw = np.clip(tensor_np[i, 0], 0.0, 1.0)
        if cmap_name:
            cm = plt.get_cmap(cmap_name)
            colored = (cm(frame_raw)[:, :, :3] * 255.0).astype(np.uint8)
            frame_im = Image.fromarray(colored, mode="RGB")
        else:
            frame_gray = (frame_raw * 255.0).astype(np.uint8)
            frame_im = Image.fromarray(frame_gray, mode="L").convert("RGB")

        x_off = pad + i * (w + pad)
        y_off = header_h + pad
        canvas.paste(frame_im, (x_off, y_off))
        draw.text((x_off + 2, y_off + 2), f"t={start_t + i}", fill=(255, 230, 80))

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


# ==============================================================================
# SIDEBAR: Checkpoint & Device Selection
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚛️ QPredRNN Control Panel")

    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    chosen_device = st.selectbox("Inference Device", device_options, index=0)

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
# MAIN APPLICATION INTERFACE
# ==============================================================================
st.markdown("<div class='main-header'>QPredRNN: Spatiotemporal Video Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Interactive research dashboard for evaluating classical and quantum-enhanced recurrent video predictors.</div>",
    unsafe_allow_html=True,
)

tab_pred, tab_compare, tab_training, tab_export = st.tabs([
    "🔮 Visual Rollout & Metrics",
    "⚖️ Multi-Model Comparison",
    "📈 Training History & Curves",
    "📄 Report & Data Export",
])

# ------------------------------------------------------------------------------
# TAB 1: Visual Rollout & Quantitative Metrics
# ------------------------------------------------------------------------------
with tab_pred:
    st.markdown("### 1. Select Input Video Sequence")

    col_src, col_ctl = st.columns([2, 1])

    with col_src:
        data_source = st.radio(
            "Input Sequence Source",
            ["KTH Processed Dataset", "Synthetic Moving-Square Generator", "Toy Dataset"],
            horizontal=True,
        )

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
            split_choice = st.selectbox("Dataset Split", ["test", "val", "train"], index=0)
            try:
                ds = FrameSequenceDataset(
                    kth_path,
                    split=split_choice,
                    input_frames=input_frames_cfg,
                    output_frames=output_frames_cfg,
                    stride=2,
                )
                sample_idx = st.slider(
                    "Temporal Window (Clip) Index",
                    0,
                    len(ds) - 1,
                    0,
                    help="Indexes 20-frame sliding temporal windows (clips) extracted with stride=2 from annotated sequence takes.",
                )
                seq_x, seq_y = ds[sample_idx]
                seq_dir, start_f, _ = ds.items[sample_idx]
                total_takes = len({item[0] for item in ds.items})
                total_span = input_frames_cfg + output_frames_cfg
                seq_desc = f"KTH {split_choice} clip #{sample_idx} [{seq_dir.name}, frame {start_f}..{start_f + total_span - 1}]"
                st.caption(
                    f"ℹ️ **KTH Split Structure**: `{split_choice}` contains **{total_takes}** annotated sequence takes producing **{len(ds):,}** temporal sliding clips (stride=2)."
                )
            except Exception as e:
                st.warning(f"Could not load KTH dataset: {e}")
        else:
            st.warning("data/processed/kth64 does not exist. Use Synthetic Generator or Toy Dataset.")

    elif data_source == "Toy Dataset":
        toy_path = Path("data/toy")
        if toy_path.exists():
            split_choice = st.selectbox("Split", ["train", "val", "test"], index=1)
            ds = FrameSequenceDataset(
                toy_path,
                split=split_choice,
                input_frames=input_frames_cfg,
                output_frames=output_frames_cfg,
                stride=1,
            )
            sample_idx = st.slider("Temporal Window (Clip) Index", 0, len(ds) - 1, 0)
            seq_x, seq_y = ds[sample_idx]
            seq_dir, start_f, _ = ds.items[sample_idx]
            seq_desc = f"Toy {split_choice} clip #{sample_idx} [{seq_dir.name}, frame {start_f}]"
            st.caption(f"ℹ️ **Toy Split Structure**: `{split_choice}` has **{len(ds)}** temporal clips.")
        else:
            st.warning("data/toy does not exist. Use Synthetic Generator.")

    if seq_x is None:
        st.info("Generating dynamic synthetic sequence (10 observed frames -> 10 future frames)...")
        seq_x, seq_y = generate_synthetic_sample(num_frames=input_frames_cfg + output_frames_cfg, size=64)
        seq_desc = "Synthetic Moving Square"

    st.markdown("---")

    # Display Input / Observed Strip
    st.markdown("#### Observed Context Frames")
    strip_in = tensor_to_pil_strip(seq_x, title=f"INPUT / CONTEXT ({seq_desc})", start_t=1)
    st.image(strip_in, use_container_width=True)

    # Prediction Action
    btn_predict = st.button("🚀 Run Video Prediction Rollout", type="primary")

    if btn_predict:
        if loaded_model is None:
            st.warning("No checkpoint loaded! Instantiating an initialized model for demonstration...")
            config_dummy = {
                "model": {"name": "predrnn_v2", "hidden_channels": 8, "num_layers": 2, "input_frames": input_frames_cfg, "output_frames": output_frames_cfg},
            }
            model_eval = build_model(config_dummy).to(chosen_device)
        else:
            model_eval = loaded_model

        model_eval.eval()

        batch_in = seq_x.unsqueeze(0).to(chosen_device)  # (1, T_in, C, H, W)

        t_start = time.perf_counter()
        with torch.no_grad():
            pred_out = model_eval(batch_in)  # (1, T_out, C, H, W)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        pred_seq = pred_out[0].cpu()  # (T_out, C, H, W)

        # Quantitative Metrics Computation
        metrics = compute_sequence_metrics(pred_seq, seq_y)

        # Baseline benchmark metrics
        last_frame_baseline = seq_x[-1:].repeat(output_frames_cfg, 1, 1, 1)
        persistence_metrics = compute_sequence_metrics(last_frame_baseline, seq_y)
        mean_frame_baseline = seq_x.mean(dim=0, keepdim=True).repeat(output_frames_cfg, 1, 1, 1)
        mean_metrics = compute_sequence_metrics(mean_frame_baseline, seq_y)

        st.markdown("#### 🎯 Rollout Performance Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MSE Loss", f"{metrics['mse']:.6f}")
        m2.metric("MAE", f"{metrics['mae']:.6f}")
        m3.metric("PSNR (dB)", f"{metrics['psnr']:.2f}")
        m4.metric("SSIM", f"{metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A")
        m5.metric("Latency", f"{latency_ms:.1f} ms")

        with st.expander("🔬 Statistical Distribution & Baseline Diagnostics", expanded=True):
            st.markdown("##### 1. Tensor Value Distribution & Range Audit")
            col_st1, col_st2, col_st3 = st.columns(3)
            with col_st1:
                st.markdown(
                    f"**Context Input $x$**:\n"
                    f"- Shape: `{tuple(seq_x.shape)}`\n"
                    f"- Range: `[{seq_x.min():.4f}, {seq_x.max():.4f}]`\n"
                    f"- Mean ± Std: `{seq_x.mean():.4f} ± {seq_x.std():.4f}`\n"
                    f"- Finite Values: `{torch.isfinite(seq_x).all().item()}`"
                )
            with col_st2:
                st.markdown(
                    f"**Target Future $y$**:\n"
                    f"- Shape: `{tuple(seq_y.shape)}`\n"
                    f"- Range: `[{seq_y.min():.4f}, {seq_y.max():.4f}]`\n"
                    f"- Mean ± Std: `{seq_y.mean():.4f} ± {seq_y.std():.4f}`\n"
                    f"- Finite Values: `{torch.isfinite(seq_y).all().item()}`"
                )
            with col_st3:
                p_std = float(pred_seq.std())
                flat_alert = " ⚠️ *(Near-flat uniform gray)*" if p_std < 0.01 else ""
                st.markdown(
                    f"**Prediction $\\hat{{y}}$**:\n"
                    f"- Shape: `{tuple(pred_seq.shape)}`\n"
                    f"- Range: `[{pred_seq.min():.4f}, {pred_seq.max():.4f}]`\n"
                    f"- Mean ± Std: `{pred_seq.mean():.4f} ± {p_std:.4f}`{flat_alert}\n"
                    f"- Finite Values: `{torch.isfinite(pred_seq).all().item()}`"
                )

            st.markdown("##### 2. Benchmark Against Non-Parametric Baselines")
            st.caption(
                "A model producing unadapted ~0.50 mid-grey frames suffers high error because it cannot preserve scene background intensity."
            )
            model_disp_name = getattr(model_eval, "model_name", "Selected Model")
            b_rows = [
                {
                    "Method": f"{model_disp_name} (Prediction)",
                    "MSE": f"{metrics['mse']:.6f}",
                    "MAE": f"{metrics['mae']:.6f}",
                    "PSNR (dB)": f"{metrics['psnr']:.2f}",
                    "SSIM": f"{metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A",
                },
                {
                    "Method": "Persistence Baseline (Repeat Last Context Frame)",
                    "MSE": f"{persistence_metrics['mse']:.6f}",
                    "MAE": f"{persistence_metrics['mae']:.6f}",
                    "PSNR (dB)": f"{persistence_metrics['psnr']:.2f}",
                    "SSIM": f"{persistence_metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A",
                },
                {
                    "Method": "Mean Context Baseline (Repeat Average Context Frame)",
                    "MSE": f"{mean_metrics['mse']:.6f}",
                    "MAE": f"{mean_metrics['mae']:.6f}",
                    "PSNR (dB)": f"{mean_metrics['psnr']:.2f}",
                    "SSIM": f"{mean_metrics['ssim']:.4f}" if HAS_SKIMAGE else "N/A",
                },
            ]
            st.table(b_rows)

        st.markdown("---")
        st.markdown("#### Visual Comparison: Predicted vs Ground Truth Future")

        c_pred, c_gt = st.columns(2)
        with c_pred:
            strip_pred = tensor_to_pil_strip(
                pred_seq,
                title=f"MODEL PREDICTION ({output_frames_cfg} frames)",
                start_t=input_frames_cfg + 1,
            )
            st.image(strip_pred, caption="Autoregressive Predictions", use_container_width=True)

        with c_gt:
            strip_gt = tensor_to_pil_strip(
                seq_y,
                title=f"GROUND TRUTH TARGET ({output_frames_cfg} frames)",
                start_t=input_frames_cfg + 1,
            )
            st.image(strip_gt, caption="Actual Future Ground Truth", use_container_width=True)

        # Difference Heatmap
        st.markdown("#### Absolute Error Heatmap (|Ground Truth - Prediction|)")
        diff_seq = torch.abs(seq_y - pred_seq)
        strip_diff = tensor_to_pil_strip(
            diff_seq,
            title="ABSOLUTE ERROR HEATMAP (Dark = 0, Bright = High Error)",
            start_t=input_frames_cfg + 1,
            cmap_name="magma",
        )
        st.image(strip_diff, caption="Per-pixel Error Distribution", use_container_width=True)

        # Per-horizon metric curves
        st.markdown("#### Per-Frame Degradation Curves Across Prediction Horizon")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#1e222d")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e222d")
            ax.tick_params(colors="white")
            ax.grid(color="#334155", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#475569")

        horizons = list(range(1, output_frames_cfg + 1))
        ax1.plot(horizons, metrics["per_frame"]["mse"], marker="o", color="#ef4444", label="MSE")
        ax1.plot(horizons, metrics["per_frame"]["mae"], marker="s", color="#f59e0b", label="MAE")
        ax1.set_title("Error vs Horizon", color="white")
        ax1.set_xlabel("Future Step", color="white")
        ax1.legend(facecolor="#1e222d", edgecolor="#334155", labelcolor="white")

        ax2.plot(horizons, metrics["per_frame"]["psnr"], marker="o", color="#3b82f6", label="PSNR (dB)")
        if HAS_SKIMAGE:
            ax2.plot(horizons, metrics["per_frame"]["ssim"], marker="^", color="#10b981", label="SSIM")
        ax2.set_title("Quality vs Horizon", color="white")
        ax2.set_xlabel("Future Step", color="white")
        ax2.legend(facecolor="#1e222d", edgecolor="#334155", labelcolor="white")

        st.pyplot(fig)


# ------------------------------------------------------------------------------
# TAB 2: Multi-Model Comparison
# ------------------------------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Model Architecture Comparison")
    st.markdown("Compare predictions and metrics across ConvLSTM, PredRNN-V2, and QPredRNN.")

    st.warning(
        "⚠️ **METHODOLOGICAL AUDIT: NON-COMPARABLE TRAINING PROTOCOLS**\n\n"
        "The current default checkpoints in `runs/` were saved under fundamentally different training budgets:\n"
        "- **ConvLSTM Baseline**: Epoch 1 (pilot exploration)\n"
        "- **PredRNN-V2**: Epoch 10 (~5,000 optimization steps, overnight run)\n"
        "- **QPredRNN**: Epoch 1 smoke test (exactly 1 batch / 1 optimization step)\n\n"
        "**Direct performance comparison between these checkpoints is not scientifically valid.** "
        "The high error in QPredRNN is caused by undertraining (unadapted Sigmoid readout weights producing uniform ~0.50 mid-grey frames), "
        "not an architectural flaw in quantum recurrence. All models must be evaluated under matched step/epoch budgets before drawing research conclusions."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("**Model 1: ConvLSTM Baseline**")
        ckpt_m1 = st.text_input("ConvLSTM Checkpoint", value="runs/kth_classical_gpu/checkpoint_best.pt")
    with col_m2:
        st.markdown("**Model 2: PredRNN-V2**")
        ckpt_m2 = st.text_input("PredRNN-V2 Checkpoint", value="runs/kth_predrnn_v2_overnight_gpu/checkpoint_best.pt")
    with col_m3:
        st.markdown("**Model 3: QPredRNN**")
        ckpt_m3 = st.text_input("QPredRNN Checkpoint", value="runs/kth_qpredrnn_smoke_gpu/checkpoint_best.pt")

    if st.button("Run Multi-Model Comparative Evaluation", type="secondary"):
        models_to_test = [
            ("ConvLSTM", ckpt_m1),
            ("PredRNN-V2", ckpt_m2),
            ("QPredRNN", ckpt_m3),
        ]

        batch_in = seq_x.unsqueeze(0).to(chosen_device)
        comp_results = []

        for name, ckpt_p in models_to_test:
            if Path(ckpt_p).exists():
                try:
                    m, ck = load_checkpoint_model(ckpt_p, chosen_device)
                    cfg_mod = ck.get("config", {}).get("model", {})
                    mod_arch = cfg_mod.get("name", name)
                    saved_ep = ck.get("epoch", "N/A")
                    saved_st = ck.get("global_step", "N/A")

                    if isinstance(saved_st, int) and saved_st <= 10:
                        train_status = f"⚠️ Smoke Test ({saved_st} step{'s' if saved_st != 1 else ''})"
                    elif isinstance(saved_ep, int) and saved_ep == 1:
                        train_status = "Pilot Run (1 epoch)"
                    elif isinstance(saved_ep, int) and saved_ep >= 10:
                        train_status = f"Trained ({saved_ep} ep, {saved_st} steps)"
                    else:
                        train_status = f"Checkpoint (ep {saved_ep})"

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        preds = m(batch_in)[0].cpu()
                    lat = (time.perf_counter() - t0) * 1000.0
                    met = compute_sequence_metrics(preds, seq_y)
                    comp_results.append({
                        "Model": name,
                        "Architecture": mod_arch,
                        "Training Status": train_status,
                        "Epoch": saved_ep,
                        "Global Steps": saved_st,
                        "Checkpoint Path": ckpt_p,
                        "MSE": met["mse"],
                        "MAE": met["mae"],
                        "PSNR (dB)": met["psnr"],
                        "SSIM": met["ssim"] if HAS_SKIMAGE else "N/A",
                        "Latency (ms)": lat,
                        "Predictions": preds,
                    })
                except Exception as e:
                    comp_results.append({
                        "Model": name,
                        "Architecture": "Error",
                        "Training Status": f"Load Error: {e}",
                        "Epoch": "N/A",
                        "Global Steps": "N/A",
                        "Checkpoint Path": ckpt_p,
                        "MSE": None,
                    })
            else:
                comp_results.append({
                    "Model": name,
                    "Architecture": "N/A",
                    "Training Status": "Checkpoint Missing",
                    "Epoch": "N/A",
                    "Global Steps": "N/A",
                    "Checkpoint Path": ckpt_p,
                    "MSE": None,
                })

        st.markdown("#### Quantitative Comparison Table")
        table_rows = []
        for r in comp_results:
            row = {
                "Model": r["Model"],
                "Architecture": r.get("Architecture", "N/A"),
                "Training Status": r.get("Training Status", "N/A"),
                "Epoch": r.get("Epoch", "N/A"),
                "Global Steps": r.get("Global Steps", "N/A"),
            }
            if r["MSE"] is not None:
                row["MSE"] = f"{r['MSE']:.6f}"
                row["MAE"] = f"{r['MAE']:.6f}"
                row["PSNR"] = f"{r['PSNR (dB)']:.2f}"
                row["SSIM"] = f"{r['SSIM']:.4f}" if isinstance(r['SSIM'], float) else r['SSIM']
                row["Latency (ms)"] = f"{r['Latency (ms)']:.1f}"
            row["Checkpoint Path"] = r.get("Checkpoint Path", "")
            table_rows.append(row)

        st.table(table_rows)

        st.markdown("#### Visual Output Comparison")
        # Show Ground Truth future strip first as reference
        strip_gt_comp = tensor_to_pil_strip(seq_y, title="GROUND TRUTH FUTURE (Reference Target)")
        st.image(strip_gt_comp, caption="Actual Ground Truth Future Frames", use_container_width=True)

        for r in comp_results:
            if "Predictions" in r:
                strip = tensor_to_pil_strip(r["Predictions"], title=f"{r['Model']} [{r.get('Training Status', '')}] Output Strip")
                st.image(strip, caption=f"{r['Model']} ({r.get('Architecture', '')}) Rollout", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: Training History & Curves
# ------------------------------------------------------------------------------
with tab_training:
    st.markdown("### 📈 Training Curves & Convergence Comparison")

    available_runs = scan_available_runs()
    runs_with_history = [r for r in available_runs if r["history_file"] is not None]

    if not runs_with_history:
        st.info("No runs with history.json discovered in runs/ directory.")
    else:
        selected_compare_runs = st.multiselect(
            "Select Runs to Plot",
            [r["name"] for r in runs_with_history],
            default=[r["name"] for r in runs_with_history][:3],
        )

        if selected_compare_runs:
            fig_loss, ax_loss = plt.subplots(figsize=(9, 4), facecolor="#1e222d")
            ax_loss.set_facecolor("#1e222d")
            ax_loss.tick_params(colors="white")
            ax_loss.grid(color="#334155", linestyle="--", alpha=0.5)
            for spine in ax_loss.spines.values():
                spine.set_color("#475569")

            colors = ["#3b82f6", "#10b981", "#c084fc", "#f59e0b", "#ef4444"]

            for i, r_name in enumerate(selected_compare_runs):
                run_obj = next(r for r in runs_with_history if r["name"] == r_name)
                try:
                    hist_data = json.loads(run_obj["history_file"].read_text(encoding="utf-8"))
                    ep_list = [h["epoch"] for h in hist_data]
                    train_loss = [h["train_loss"] for h in hist_data]
                    val_loss = [h["val_loss"] for h in hist_data]

                    c = colors[i % len(colors)]
                    ax_loss.plot(ep_list, val_loss, marker="o", color=c, label=f"{r_name} (Val)")
                    ax_loss.plot(ep_list, train_loss, linestyle="--", marker="s", color=c, alpha=0.6, label=f"{r_name} (Train)")
                except Exception as e:
                    st.warning(f"Failed to parse history for {r_name}: {e}")

            ax_loss.set_title("Training and Validation MSE Convergence", color="white")
            ax_loss.set_xlabel("Epoch", color="white")
            ax_loss.set_ylabel("Loss", color="white")
            ax_loss.legend(facecolor="#1e222d", edgecolor="#334155", labelcolor="white")
            st.pyplot(fig_loss)


# ------------------------------------------------------------------------------
# TAB 4: Report Export
# ------------------------------------------------------------------------------
with tab_export:
    st.markdown("### 📄 Experiment Report Generation & Export")
    st.markdown("Export reproducible experiment summaries, configuration snapshots, and metric tables.")

    report_title = st.text_input("Report Title", value="QPredRNN Video Prediction Evaluation Report")
    notes = st.text_area("Research Notes / Observations", value="Pilot evaluation comparing classical and quantum-enhanced spatiotemporal video prediction.")

    if st.button("Generate Markdown Report"):
        ckpt_label = selected_ckpt_path or "No checkpoint selected"
        metrics_dict = {}
        if "metrics" in locals() and metrics:
            metrics_dict = metrics

        report_md = f"""# {report_title}
**Generated**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Platform**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
**Device**: `{chosen_device}`

## 1. Selected Checkpoint & Model
- **Checkpoint**: `{ckpt_label}`
- **Quantum Integration**: {'Enabled (PennyLane VQC)' if loaded_model and getattr(loaded_model, 'is_quantum', False) else 'Classical Baseline'}
- **Simulation Disclaimer**: All quantum circuits are simulated via PennyLane `default.qubit` statevector simulator on classical hardware.

## 2. Quantitative Evaluation
| Metric | Value |
| :--- | :--- |
| MSE | {metrics_dict.get('mse', 'N/A')} |
| MAE | {metrics_dict.get('mae', 'N/A')} |
| PSNR (dB) | {metrics_dict.get('psnr', 'N/A')} |
| SSIM | {metrics_dict.get('ssim', 'N/A')} |

## 3. Observations & Notes
{notes}
"""
        st.text_area("Generated Markdown", value=report_md, height=260)
        st.download_button(
            label="📥 Download Report as Markdown (.md)",
            data=report_md,
            file_name="qpredrnn_experiment_report.md",
            mime="text/markdown",
        )
