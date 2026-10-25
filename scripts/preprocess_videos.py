import argparse
import json
import re
from pathlib import Path
import cv2
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Decode videos into grayscale frame sequences.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--out", default="data/processed/kth64", help="Output directory for processed frames")
    parser.add_argument("--size", type=int, default=64, help="Spatial resolution (square)")
    parser.add_argument("--sample-every", type=int, default=1, help="Temporal subsampling stride")
    parser.add_argument("--min-frames", type=int, default=20, help="Minimum required frames to keep sequence")
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit number of sequences processed (for testing)")
    args = parser.parse_args()

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    sequence_rows = []

    # Sort rows to ensure deterministic processing order
    processed_count = 0

    for idx, row in enumerate(rows):
        if args.max_sequences is not None and processed_count >= args.max_sequences:
            print(f"Reached --max-sequences limit ({args.max_sequences}). Stopping preprocessing.")
            break

        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Sequence {row.get('sequence_id', row.get('relative_path'))} has no reviewed split. Rebuild/review manifest.")

        src = Path(row["path"])
        if not src.exists():
            print(f"WARNING: Video file not found: {src}. Skipping.")
            continue

        seq_id = row.get("sequence_id")
        if not seq_id:
            cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem)
            seq_id = f"{idx:05d}_{cleaned}"

        seq_dir = out_root / "sequences" / seq_id
        seq_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            print(f"WARNING: Could not open video: {src}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sequence boundary handling (1-based inclusive)
        start_frame_1based = row.get("start_frame")
        end_frame_1based = row.get("end_frame")

        if start_frame_1based is not None and end_frame_1based is not None:
            # Seek to start frame (0-based index: start_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_1based - 1)
            current_frame_1based = start_frame_1based
            saved = 0
            while current_frame_1based <= end_frame_1based:
                ok, frame = cap.read()
                if not ok:
                    break
                if (current_frame_1based - start_frame_1based) % args.sample_every == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_AREA)
                    Image.fromarray(resized).save(seq_dir / f"{saved:05d}.png")
                    saved += 1
                current_frame_1based += 1
        else:
            # Fallback for unannotated videos: entire video
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
            print(f"SKIP {seq_id}: only {saved} frames (< min-frames {args.min_frames})")
            continue

        meta = {
            "sequence_id": seq_id,
            "source_path": str(src),
            "source_filename": row.get("source_filename", src.name),
            "subject_id": row.get("subject_id"),
            "action": row.get("action"),
            "scenario": row.get("scenario"),
            "take_idx": row.get("take_idx"),
            "start_frame": start_frame_1based,
            "end_frame": end_frame_1based,
            "split": row["split"],
            "fps": fps,
            "saved_frames": saved,
            "sample_every": args.sample_every,
            "image_size": args.size,
            "normalization": "uint8 PNG; loader maps to [0,1]",
        }
        (seq_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        sequence_rows.append({"sequence_id": seq_id, "split": row["split"]})
        processed_count += 1
        print(f"[{processed_count}] {seq_id}: {saved} frames [{row['split']}]")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "splits.json").write_text(json.dumps(sequence_rows, indent=2), encoding="utf-8")
    print(f"Preprocessing complete: {len(sequence_rows)} sequences written to {out_root}")


if __name__ == "__main__":
    main()
