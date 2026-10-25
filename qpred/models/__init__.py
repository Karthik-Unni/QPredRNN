from __future__ import annotations
import torch.nn as nn
from .convlstm import VideoConvLSTM

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
    raise ValueError(f"Unknown or unsupported model name: {name}")

__all__ = ["VideoConvLSTM", "build_model"]
