import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from qpred.data.dataset import FrameSequenceDataset
from qpred.data.manifest import parse_kth_filename, kth_subject_split, validate_manifest_splits

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
    # Augmentation enabled
    ds = FrameSequenceDataset(out, "train", input_frames=5, output_frames=5, stride=1, image_size=32, augment=True)
    x, y = ds[0]
    assert x.shape == (5, 1, 32, 32)
    assert y.shape == (5, 1, 32, 32)
