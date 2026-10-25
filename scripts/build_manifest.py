import argparse
import json
from pathlib import Path
from qpred.data.manifest import (
    build_manifest,
    assign_splits,
    parse_kth_sequences_txt,
    validate_manifest_splits,
)


def main():
    parser = argparse.ArgumentParser(description="Build and review a video manifest.")
    parser.add_argument("--root", required=True, help="Root folder containing raw video files")
    parser.add_argument("--out", default="data/manifest.json")
    parser.add_argument("--sequences-file", default=None, help="Path to 00sequences.txt boundary file")
    parser.add_argument(
        "--assign-kth-splits",
        action="store_true",
        help="Assign example subject split IDs 1-8/9-16/17-25; review before use",
    )
    args = parser.parse_args()

    root = Path(args.root)
    seq_file = Path(args.sequences_file) if args.sequences_file else root / "00sequences.txt"

    if seq_file.exists():
        print(f"Detected sequence boundary file: {seq_file}")
        _, missing = parse_kth_sequences_txt(seq_file)
        if missing:
            print(f"Known missing KTH entries in 00sequences.txt ({len(missing)}): {missing}")

    rows = build_manifest(args.root, sequences_txt_path=seq_file if seq_file.exists() else None)

    if args.assign_kth_splits:
        rows = assign_splits(rows)
        validation = validate_manifest_splits(rows)
        print(f"Split validation report: {validation['split_counts']}, Disjoint: {validation['disjoint']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    unknown = [r for r in rows if r.get("subject_id") is None]
    print(f"Sequences generated in manifest: {len(rows)}; unknown subject IDs: {len(unknown)}")
    if unknown:
        print("Review these paths and manually assign subject IDs before preprocessing:")
        for row in unknown[:50]:
            print(" -", row.get("relative_path", row.get("path")))

    print(f"Manifest successfully written to: {out_path}")


if __name__ == "__main__":
    main()
