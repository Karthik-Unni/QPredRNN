"""Visual QA Script: generate horizontal frame strip / contact sheet for manual inspection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from qpred.data.dataset import FrameSequenceDataset

def create_strip(frames: torch.Tensor, title: str, start_t: int = 1) -> Image.Image:
    """Create a labeled horizontal strip from a (T, 1, H, W) tensor."""
    t, c, h, w = frames.shape
    pad = 4
    header_h = 24
    total_w = t * (w + pad) + pad
    total_h = h + header_h + pad * 2
    
    strip = Image.new("RGB", (total_w, total_h), (20, 24, 33))
    draw = ImageDraw.Draw(strip)
    draw.text((pad, 4), title, fill=(180, 200, 240))
    
    for i in range(t):
        arr = (frames[i, 0].clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        frame_im = Image.fromarray(arr, mode="L").convert("RGB")
        x_offset = pad + i * (w + pad)
        y_offset = header_h + pad
        strip.paste(frame_im, (x_offset, y_offset))
        draw.text((x_offset + 2, y_offset + 2), f"t={start_t + i}", fill=(255, 230, 80))
        
    return strip

def main():
    parser = argparse.ArgumentParser(description="Create visual QA frame strips from dataset.")
    parser.add_argument("--root", default="data/toy", help="Path to processed dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", default="data/qa_sample_strip.png", help="Path to save output contact sheet")
    parser.add_argument("--input-frames", type=int, default=10)
    parser.add_argument("--output-frames", type=int, default=10)
    parser.add_argument("--index", type=int, default=0, help="Sample sequence window index")
    args = parser.parse_args()

    ds = FrameSequenceDataset(
        args.root,
        split=args.split,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        stride=1
    )
    
    if args.index >= len(ds):
        raise IndexError(f"Sample index {args.index} out of range (dataset has {len(ds)} windows for {args.split}).")
        
    x, y = ds[args.index]
    print(f"Sample {args.index} from {args.split} split:")
    print(f"  Input x shape : {tuple(x.shape)}, min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")
    print(f"  Target y shape: {tuple(y.shape)}, min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    strip_x = create_strip(x, f"INPUT FRAMES ({args.split} clip #{args.index})", start_t=1)
    strip_y = create_strip(y, f"TARGET FUTURE FRAMES ({args.split} clip #{args.index})", start_t=args.input_frames + 1)
    
    total_w = max(strip_x.width, strip_y.width)
    total_h = strip_x.height + strip_y.height + 8
    canvas = Image.new("RGB", (total_w, total_h), (12, 14, 20))
    canvas.paste(strip_x, (0, 0))
    canvas.paste(strip_y, (0, strip_x.height + 8))
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved visual QA strip to: {out_path}")

if __name__ == "__main__":
    main()
