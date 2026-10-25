import argparse, csv, json
from pathlib import Path
from qpred.data.manifest import build_manifest, assign_splits

def main():
    parser = argparse.ArgumentParser(description="Build and review a video manifest.")
    parser.add_argument("--root", required=True, help="Root folder containing raw video files")
    parser.add_argument("--out", default="data/manifest.json")
    parser.add_argument("--assign-kth-splits", action="store_true",
                        help="Assign example subject split IDs 1-8/9-16/17-25; review before use")
    args = parser.parse_args()
    rows = build_manifest(args.root)
    if args.assign_kth_splits:
        rows = assign_splits(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    unknown = [r for r in rows if r.get("subject_id") is None]
    print(f"Videos found: {len(rows)}; unknown subject IDs: {len(unknown)}")
    if unknown:
        print("Review these paths and manually assign subject IDs before preprocessing:")
        for row in unknown[:50]:
            print(" -", row["relative_path"])
    print(f"Manifest written: {args.out}")
if __name__ == "__main__":
    main()
