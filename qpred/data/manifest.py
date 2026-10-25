from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
KTH_ACTIONS = ("boxing", "handclapping", "handwaving", "jogging", "running", "walking")


def parse_kth_filename(name: str) -> Dict[str, Optional[any]]:
    """Extract subject, action, and scenario from a filename or relative path.

    Expected format: personXX_action_dY_uncomp.avi or personXX/action/...
    Leaves fields as None if they cannot be safely matched.
    """
    stem = Path(name).stem.lower()
    full_path_str = str(name).lower()

    # Match subject: person01, person_01, person1, etc.
    subject_match = re.search(r"person_?(\d+)", stem) or re.search(r"person_?(\d+)", full_path_str)
    subject_id = int(subject_match.group(1)) if subject_match else None

    # Match action
    action = None
    for act in KTH_ACTIONS:
        if act in stem or act in full_path_str:
            action = act
            break

    # Match scenario condition: d1 (outdoors), d2 (outdoors scale), d3 (outdoors clothes), d4 (indoors)
    scenario_match = re.search(r"[\b_](d[1-4])[\b_]", stem)
    scenario = scenario_match.group(1) if scenario_match else None

    return {
        "subject_id": subject_id,
        "action": action,
        "scenario": scenario,
    }


def parse_kth_sequences_txt(filepath: str | Path) -> Tuple[Dict[str, Optional[List[Tuple[int, int]]]], List[str]]:
    """Parse official KTH 00sequences.txt file.

    Extracts 1-based inclusive frame ranges for each annotated sequence/take.
    Identifies explicitly marked missing videos (e.g., person13_handclapping_d3).

    Returns:
        tuple of (entries_dict, missing_prefixes_list)
        where entries_dict maps video_prefix (e.g. 'person01_boxing_d1') to
        either a list of (start_frame, end_frame) tuples (1-based inclusive) or None if missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sequences file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: Dict[str, Optional[List[Tuple[int, int]]]] = {}
    missing_list: List[str] = []

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("person"):
            continue

        if "*missing*" in line or "missing" in line.lower():
            prefix = line.split()[0].strip()
            entries[prefix] = None
            missing_list.append(prefix)
            continue

        if "frames" in line:
            prefix, ranges_str = line.split("frames", 1)
            prefix = prefix.strip()
            parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
            ranges: List[Tuple[int, int]] = []
            for part in parts:
                m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
                if not m:
                    raise ValueError(f"Invalid frame range format '{part}' in line: {line}")
                start_frame = int(m.group(1))
                end_frame = int(m.group(2))
                if start_frame > end_frame:
                    raise ValueError(f"Inverted frame range: {start_frame}-{end_frame} in {line}")
                ranges.append((start_frame, end_frame))
            entries[prefix] = ranges

    return entries, missing_list


def build_manifest(
    root: str | Path,
    sequences_txt_path: str | Path | None = None,
    read_video_info: bool = False,
) -> list[dict]:
    """Create a structured manifest from raw video files and optional sequence boundaries.

    If 00sequences.txt is provided or discovered in root, separate manifest rows
    are generated for each annotated take/sequence boundary.
    Missing videos (e.g. person13_handclapping_d3) are skipped without error.
    """
    root = Path(root)
    rows = []

    # Check for 00sequences.txt
    seq_file = Path(sequences_txt_path) if sequences_txt_path else root / "00sequences.txt"
    has_sequences_file = seq_file.exists()

    if has_sequences_file:
        entries, missing = parse_kth_sequences_txt(seq_file)
        # Index raw video files by prefix for fast, precise lookup
        video_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        prefix_to_file: Dict[str, Path] = {}
        for vf in video_files:
            # Map stems like 'person01_boxing_d1_uncomp' or 'person01_boxing_d1'
            clean_stem = re.sub(r"(_uncomp)?$", "", vf.stem, flags=re.IGNORECASE)
            prefix_to_file[clean_stem] = vf

        for prefix, ranges in sorted(entries.items()):
            if ranges is None:
                # Officially missing video (e.g. person13_handclapping_d3)
                continue

            vf = prefix_to_file.get(prefix)
            if vf is None:
                # File not found on disk; skip rather than fabricating
                continue

            parsed = parse_kth_filename(vf.name)

            for take_idx, (start, end) in enumerate(ranges, start=1):
                seq_id = f"{prefix}_take{take_idx:02d}"
                rows.append({
                    "sequence_id": seq_id,
                    "video_prefix": prefix,
                    "source_filename": vf.name,
                    "path": str(vf.resolve()),
                    "relative_path": str(vf.relative_to(root)).replace("\\", "/"),
                    "subject_id": parsed["subject_id"],
                    "action": parsed["action"],
                    "scenario": parsed["scenario"],
                    "take_idx": take_idx,
                    "start_frame": start,        # 1-based inclusive
                    "end_frame": end,            # 1-based inclusive
                    "start_idx": start - 1,      # 0-based inclusive
                    "end_idx": end,              # 0-based exclusive (slice limit)
                    "frame_count": end - start + 1,
                    "extension": vf.suffix.lower(),
                })
        return rows

    # Fallback when no sequence boundaries file is present (e.g. unannotated / toy datasets)
    cv2 = None
    if read_video_info:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            cv2 = None

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            parsed = parse_kth_filename(p.name)
            fps = None
            frame_count = None
            width = None
            height = None

            if cv2 is not None:
                cap = cv2.VideoCapture(str(p.resolve()))
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

            rows.append({
                "sequence_id": p.stem,
                "path": str(p.resolve()),
                "relative_path": str(p.relative_to(root)).replace("\\", "/"),
                "subject_id": parsed["subject_id"],
                "action": parsed["action"],
                "scenario": parsed["scenario"],
                "filename": p.name,
                "extension": p.suffix.lower(),
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            })
    return rows


def write_manifest(rows: Iterable[dict], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")


def kth_subject_split(subject_id: Optional[int]) -> str:
    """Canonical 3-way subject split:
      - Subjects 1–8: Train (8 subjects)
      - Subjects 9–16: Validation (8 subjects)
      - Subjects 17–25: Test (9 subjects)
    Strictly disjoint to eliminate test-set leakage.
    """
    if subject_id is None:
        raise ValueError("Subject ID is missing; manually review and map this video before splitting.")
    if 1 <= subject_id <= 8:
        return "train"
    if 9 <= subject_id <= 16:
        return "val"
    if 17 <= subject_id <= 25:
        return "test"
    raise ValueError(f"Unexpected KTH subject ID: {subject_id} (expected 1–25)")


def assign_splits(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        new = dict(row)
        new["split"] = kth_subject_split(new.get("subject_id"))
        result.append(new)
    return result


def validate_manifest_splits(rows: list[dict]) -> dict:
    """Verify that split assignments are completely disjoint at subject level."""
    subject_to_splits: dict[int, set[str]] = {}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    action_counts: dict[str, int] = {}
    unknown_subjects = 0

    for r in rows:
        sub = r.get("subject_id")
        split = r.get("split")
        if sub is None:
            unknown_subjects += 1
            continue
        if split:
            subject_to_splits.setdefault(sub, set()).add(split)
            split_counts[split] = split_counts.get(split, 0) + 1
        act = r.get("action")
        if act:
            action_counts[act] = action_counts.get(act, 0) + 1

    leakage = {sub: splits for sub, splits in subject_to_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Cross-split leakage detected for subjects: {leakage}")

    return {
        "total_sequences": len(rows),
        "unknown_subjects": unknown_subjects,
        "split_counts": split_counts,
        "action_counts": action_counts,
        "disjoint": True,
    }
