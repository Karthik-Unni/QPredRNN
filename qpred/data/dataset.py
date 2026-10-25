from __future__ import annotations
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FrameSequenceDataset(Dataset):
    """Dataset for pre-extracted frame directories with split metadata.

    Expected layout:
      root/
        sequences/<sequence_id>/00000.png ...
        sequences/<sequence_id>/metadata.json  # optional
        splits.json  # list of {sequence_id, split}
    
    Guarantees:
      - Strictly disjoint splitting (verified via splits.json)
      - Temporally consistent augmentation (entire clip is flipped identically)
      - Predictable shape: (T, 1, H, W) normalized to [0.0, 1.0] float32
    """
    def __init__(self, root: str | Path, split: str, input_frames: int = 10,
                 output_frames: int = 10, stride: int = 1, image_size: int = 64,
                 augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.image_size = image_size
        self.augment = augment

        split_path = self.root / "splits.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_path}. Create an explicit split file first.")

        rows = json.loads(split_path.read_text(encoding="utf-8"))
        self.items = []
        required = input_frames + output_frames

        for row in rows:
            if row.get("split") != split:
                continue
            seq_dir = self.root / "sequences" / row["sequence_id"]
            if not seq_dir.exists():
                continue
            frame_paths = sorted(seq_dir.glob("*.png"))
            if len(frame_paths) < required:
                continue
            for start in range(0, len(frame_paths) - required + 1, stride):
                self.items.append((seq_dir, start, frame_paths))

        if not self.items:
            raise ValueError(f"No windows for split={split}; check data root, splits, and sequence length.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_dir, start, paths = self.items[index]
        clip = []
        total = self.input_frames + self.output_frames
        
        # Decide augmentation consistently for the ENTIRE clip (never frame-by-frame)
        do_hflip = self.augment and (random.random() < 0.5)

        for path in paths[start:start + total]:
            with Image.open(path) as im:
                im = im.convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                if do_hflip:
                    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                clip.append(arr)

        tensor = torch.from_numpy(np.stack(clip)[:, None, :, :])  # (T, 1, H, W)
        x = tensor[:self.input_frames]
        y = tensor[self.input_frames:]
        return x, y
