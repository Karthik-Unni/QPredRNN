import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy")
    p.add_argument("--sequences", type=int, default=24)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args()
    root = Path(a.out)
    rows = []
    for i in range(a.sequences):
        seq_id = f"toy_{i:04d}"
        split = "train" if i < int(a.sequences * .65) else ("val" if i < int(a.sequences * .82) else "test")
        seq_dir = root / "sequences" / seq_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        size = a.size
        x0 = 5 + (i * 7) % max(8, size - 20)
        y0 = 5 + (i * 11) % max(8, size - 20)
        vx = 1 if i % 2 == 0 else -1
        vy = 1 if i % 3 == 0 else -1
        for t in range(a.frames):
            x = (x0 + vx * t) % (size - 14)
            y = (y0 + vy * t) % (size - 14)
            im = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(im)
            draw.rectangle((x, y, x + 12, y + 12), fill=255)
            im.save(seq_dir / f"{t:05d}.png")
        rows.append({"sequence_id": seq_id, "split": split})
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} toy sequences to {root}")
if __name__ == "__main__":
    main()
