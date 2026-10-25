from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

def read_frame_directory(directory: str | Path, size: int = 64) -> np.ndarray:
    """Read sorted image frames from a directory into T,H,W float32 in [0,1]."""
    directory = Path(directory)
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"No image frames found in {directory}")
    frames = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            frames.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)

def write_frame_directory(frames: np.ndarray, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        array = np.clip(frame * 255.0, 0, 255).astype("uint8")
        Image.fromarray(array, mode="L").save(directory / f"{i:05d}.png")
