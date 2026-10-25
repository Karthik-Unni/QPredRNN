import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split-file", default=None)
    a = p.parse_args()
    root = Path(a.root)
    split_file = Path(a.split_file) if a.split_file else root / "splits.json"
    rows = json.loads(split_file.read_text(encoding="utf-8"))
    counts = {}
    seen = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        seen.setdefault(r["sequence_id"], set()).add(r["split"])
    leakage = [k for k, v in seen.items() if len(v) > 1]
    if leakage:
        raise SystemExit(f"Split leakage: {leakage[:10]}")
    print("Sequence counts:", counts)
    print("No sequence appears in multiple splits.")
    for r in rows:
        folder = root / "sequences" / r["sequence_id"]
        if not folder.exists():
            raise SystemExit(f"Missing sequence directory: {folder}")
    print("All sequence directories exist.")
if __name__ == "__main__":
    main()
