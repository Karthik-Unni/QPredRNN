import argparse, json, re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Decode videos into grayscale frame sequences.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="data/processed/kth64")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    sequence_rows = []
    for idx, row in enumerate(rows):
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Video {row.get('relative_path')} has no reviewed split. Rebuild/review manifest.")
        src = Path(row["path"])
        if not src.exists():
            raise FileNotFoundError(src)
        seq_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem)
        # Prefix index avoids collisions if two folders contain identically named clips.
        seq_id = f"{idx:05d}_{seq_id}"
        seq_dir = out_root / "sequences" / seq_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not decode video: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_count % args.sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_AREA)
                Image.fromarray(resized).save(seq_dir / f"{saved:05d}.png")
                saved += 1
            frame_count += 1
        cap.release()
        if saved < args.min_frames:
            import shutil
            shutil.rmtree(seq_dir, ignore_errors=True)
            print(f"SKIP {src}: only {saved} sampled frames")
            continue
        meta = {
            "source_path": str(src), "subject_id": row.get("subject_id"),
            "split": row["split"], "fps": fps, "source_frames": frame_count,
            "saved_frames": saved, "sample_every": args.sample_every,
            "image_size": args.size, "normalization": "uint8 PNG; loader maps to [0,1]"
        }
        (seq_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        sequence_rows.append({"sequence_id": seq_id, "split": row["split"]})
        print(f"{seq_id}: {saved} frames [{row['split']}]")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "splits.json").write_text(json.dumps(sequence_rows, indent=2), encoding="utf-8")
    print(f"Preprocessing complete: {len(sequence_rows)} sequences -> {out_root}")
if __name__ == "__main__":
    main()
