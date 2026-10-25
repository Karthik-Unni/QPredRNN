from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

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

def build_manifest(root: str | Path, read_video_info: bool = False) -> list[dict]:
    """Create a structured manifest from raw video files without destructive modifications."""
    root = Path(root)
    rows = []
    
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
        "total_videos": len(rows),
        "unknown_subjects": unknown_subjects,
        "split_counts": split_counts,
        "action_counts": action_counts,
        "disjoint": True
    }
