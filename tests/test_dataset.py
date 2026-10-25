import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from qpred.data.dataset import FrameSequenceDataset
from qpred.data.manifest import (
    parse_kth_filename,
    parse_kth_sequences_txt,
    kth_subject_split,
    assign_splits,
    validate_manifest_splits,
    build_manifest,
)


def test_dataset_shapes_and_ranges(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "make_toy_dataset.py"
    out = tmp_path / "toy"
    subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--sequences", "10", "--frames", "12", "--size", "32"],
        check=True
    )
    ds = FrameSequenceDataset(out, "train", input_frames=5, output_frames=5, stride=1, image_size=32)
    x, y = ds[0]
    assert x.shape == (5, 1, 32, 32)
    assert y.shape == (5, 1, 32, 32)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert 0.0 <= y.min() and y.max() <= 1.0


def test_manifest_filename_parsing():
    parsed = parse_kth_filename("person03_boxing_d1_uncomp.avi")
    assert parsed["subject_id"] == 3
    assert parsed["action"] == "boxing"
    assert parsed["scenario"] == "d1"

    parsed2 = parse_kth_filename("person22_handwaving_d4_uncomp.avi")
    assert parsed2["subject_id"] == 22
    assert parsed2["action"] == "handwaving"
    assert parsed2["scenario"] == "d4"

    # Unknown subject should return None rather than guessing silently
    parsed_unknown = parse_kth_filename("random_video_clip.avi")
    assert parsed_unknown["subject_id"] is None


def test_kth_canonical_subject_splits():
    # Train: 1-8 (8 subjects)
    for s in range(1, 9):
        assert kth_subject_split(s) == "train"
    # Val: 9-16 (8 subjects)
    for s in range(9, 17):
        assert kth_subject_split(s) == "val"
    # Test: 17-25 (9 subjects)
    for s in range(17, 26):
        assert kth_subject_split(s) == "test"

    with pytest.raises(ValueError):
        kth_subject_split(99)
    with pytest.raises(ValueError):
        kth_subject_split(None)


def test_split_leakage_detector():
    clean_manifest = [
        {"subject_id": 1, "split": "train", "action": "walking"},
        {"subject_id": 9, "split": "val", "action": "walking"},
        {"subject_id": 17, "split": "test", "action": "walking"},
    ]
    report = validate_manifest_splits(clean_manifest)
    assert report["disjoint"] is True

    leaky_manifest = [
        {"subject_id": 1, "split": "train", "action": "walking"},
        {"subject_id": 1, "split": "test", "action": "walking"},
    ]
    with pytest.raises(ValueError, match="Cross-split leakage detected"):
        validate_manifest_splits(leaky_manifest)


def test_dataset_clip_level_augmentation(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "make_toy_dataset.py"
    out = tmp_path / "toy_aug"
    subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--sequences", "4", "--frames", "10", "--size", "32"],
        check=True
    )
    ds = FrameSequenceDataset(out, "train", input_frames=5, output_frames=5, stride=1, image_size=32, augment=True)
    x, y = ds[0]
    assert x.shape == (5, 1, 32, 32)
    assert y.shape == (5, 1, 32, 32)


def test_parse_kth_sequences_txt(tmp_path):
    mock_seq = tmp_path / "00sequences.txt"
    mock_seq.write_text(
        "Header comments...\n\n"
        "person01_boxing_d1\tframes\t1-95, 96-185, 186-245, 246-360\n"
        "person13_handclapping_d3\t*missing*\n"
        "person13_handwaving_d3\tframes\t1-125, 126-240, 241-336\n",
        encoding="utf-8"
    )

    entries, missing = parse_kth_sequences_txt(mock_seq)
    assert "person13_handclapping_d3" in missing
    assert entries["person13_handclapping_d3"] is None

    assert len(entries["person01_boxing_d1"]) == 4
    assert entries["person01_boxing_d1"][0] == (1, 95)
    assert entries["person01_boxing_d1"][3] == (246, 360)

    assert len(entries["person13_handwaving_d3"]) == 3
    assert entries["person13_handwaving_d3"][0] == (1, 125)


def test_boundary_handling_and_no_fabrication(tmp_path):
    # Setup mock raw directory
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "person01_boxing_d1_uncomp.avi").write_text("dummy")

    mock_seq = raw_dir / "00sequences.txt"
    mock_seq.write_text(
        "person01_boxing_d1\tframes\t1-95, 96-185\n"
        "person13_handclapping_d3\t*missing*\n",
        encoding="utf-8"
    )

    manifest = build_manifest(raw_dir)
    assert len(manifest) == 2
    # Verify take 1
    t1 = manifest[0]
    assert t1["sequence_id"] == "person01_boxing_d1_take01"
    assert t1["start_frame"] == 1
    assert t1["end_frame"] == 95
    assert t1["start_idx"] == 0
    assert t1["end_idx"] == 95
    assert t1["frame_count"] == 95
    assert t1["take_idx"] == 1

    # Verify take 2
    t2 = manifest[1]
    assert t2["sequence_id"] == "person01_boxing_d1_take02"
    assert t2["start_frame"] == 96
    assert t2["end_frame"] == 185
    assert t2["start_idx"] == 95
    assert t2["end_idx"] == 185
    assert t2["frame_count"] == 90

    # Ensure missing video is NOT fabricated
    missing_items = [m for m in manifest if "person13_handclapping_d3" in m["sequence_id"]]
    assert len(missing_items) == 0


def test_annotated_manifest_real_kth():
    kth_raw = Path(__file__).parents[1] / "data" / "raw" / "KTH"
    if not (kth_raw / "00sequences.txt").exists():
        pytest.skip("data/raw/KTH/00sequences.txt not available")

    rows = build_manifest(kth_raw)
    assert len(rows) == 2391, f"Expected 2391 annotated takes, got {len(rows)}"

    rows_with_splits = assign_splits(rows)
    report = validate_manifest_splits(rows_with_splits)
    assert report["disjoint"] is True
    assert report["unknown_subjects"] == 0
    assert report["split_counts"]["train"] == 768
    assert report["split_counts"]["val"] == 760
    assert report["split_counts"]["test"] == 863
