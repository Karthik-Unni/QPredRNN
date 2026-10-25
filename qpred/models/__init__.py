from __future__ import annotations
import torch.nn as nn
from .convlstm import VideoConvLSTM
from .predrnn_v2 import (
    PredRNNV2,
    SpatioTemporalLSTMCell,
    compute_memory_decoupling_loss,
    reserve_schedule_sampling_exp,
    generate_scheduled_sampling_masks,
)


def build_model(config: dict) -> nn.Module:
    """Build a video prediction model based on configuration."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "classical")

    if name == "classical":
        return VideoConvLSTM(
            image_channels=model_cfg.get("image_channels", 1),
            hidden_channels=model_cfg.get("hidden_channels", 16),
            input_frames=model_cfg.get("input_frames", 5),
            output_frames=model_cfg.get("output_frames", 5),
            num_layers=model_cfg.get("num_layers", 2),
            kernel_size=model_cfg.get("kernel_size", 3),
        )
    elif name == "predrnn_v2":
        return PredRNNV2(
            image_channels=model_cfg.get("image_channels", 1),
            hidden_channels=model_cfg.get("hidden_channels", 32),
            input_frames=model_cfg.get("input_frames", 10),
            output_frames=model_cfg.get("output_frames", 10),
            num_layers=model_cfg.get("num_layers", 2),
            kernel_size=model_cfg.get("kernel_size", 3),
            decouple_loss_weight=model_cfg.get("decouple_loss_weight", 0.1),
        )

    raise ValueError(f"Unknown or unsupported model name: {name}")


__all__ = [
    "VideoConvLSTM",
    "PredRNNV2",
    "SpatioTemporalLSTMCell",
    "compute_memory_decoupling_loss",
    "reserve_schedule_sampling_exp",
    "generate_scheduled_sampling_masks",
    "build_model",
]
