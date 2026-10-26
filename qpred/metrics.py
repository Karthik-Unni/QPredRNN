from __future__ import annotations
import math
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    ssim_fn = None
    HAS_SKIMAGE = False


def compute_mse(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Compute Mean Squared Error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.mean(np.abs(pred - target)))


def compute_psnr(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(pred, target)
    if mse < eps:
        return 100.0  # Perfect match ceiling
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def compute_ssim(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) averaged across frames."""
    if not HAS_SKIMAGE:
        return 0.0

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Squeeze batch dimension if present: expected shape (T, C, H, W) or (T, H, W)
    if pred.ndim == 5:  # (B, T, C, H, W)
        pred = pred[0]
        target = target[0]

    if pred.ndim == 4:  # (T, C, H, W)
        # For grayscale with C=1, squeeze channel
        if pred.shape[1] == 1:
            pred = pred[:, 0]
            target = target[:, 0]
        else:
            # Multi-channel: transpose to (T, H, W, C)
            pred = np.transpose(pred, (0, 2, 3, 1))
            target = np.transpose(target, (0, 2, 3, 1))

    t_steps = pred.shape[0]
    ssim_vals = []
    for t in range(t_steps):
        p_frame = np.clip(pred[t], 0.0, data_range)
        t_frame = np.clip(target[t], 0.0, data_range)
        val = ssim_fn(t_frame, p_frame, data_range=data_range)
        ssim_vals.append(float(val))

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


def compute_sequence_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> dict:
    """Compute comprehensive per-frame and overall sequence metrics."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if pred.ndim == 5:
        pred = pred[0]
        target = target[0]

    t_steps = pred.shape[0]
    frame_mse = []
    frame_mae = []
    frame_psnr = []
    frame_ssim = []

    for t in range(t_steps):
        p = pred[t]
        tgt = target[t]
        mse_t = compute_mse(p, tgt)
        mae_t = compute_mae(p, tgt)
        psnr_t = compute_psnr(p, tgt, data_range=data_range)

        frame_mse.append(mse_t)
        frame_mae.append(mae_t)
        frame_psnr.append(psnr_t)

        if HAS_SKIMAGE:
            p_img = p[0] if p.ndim == 3 and p.shape[0] == 1 else p
            tgt_img = tgt[0] if tgt.ndim == 3 and tgt.shape[0] == 1 else tgt
            ssim_t = float(ssim_fn(tgt_img, p_img, data_range=data_range))
            frame_ssim.append(ssim_t)
        else:
            frame_ssim.append(0.0)

    return {
        "mse": float(np.mean(frame_mse)),
        "mae": float(np.mean(frame_mae)),
        "psnr": float(np.mean(frame_psnr)),
        "ssim": float(np.mean(frame_ssim)),
        "per_frame": {
            "mse": frame_mse,
            "mae": frame_mae,
            "psnr": frame_psnr,
            "ssim": frame_ssim,
        },
    }
