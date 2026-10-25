# Dataset preparation and data contract

## Expected processed layout
```text
data/processed/kth64/
  splits.json
  sequences/
    <sequence_id>/
      00000.png
      00001.png
      ...
```

`splits.json` is a JSON list:
```json
[
  {"sequence_id": "subject01_action_clip01", "split": "train"},
  {"sequence_id": "subject09_action_clip02", "split": "val"},
  {"sequence_id": "subject17_action_clip03", "split": "test"}
]
```

Do not upload raw KTH or processed frames to GitHub unless the dataset license explicitly permits it. `.gitignore` excludes data and runs.

## Preprocessing decisions to record
- Source archive/version and download date.
- Subject/action identifiers and video-to-subject mapping.
- Subject-disjoint split rule.
- Decoder/version and frame sampling policy.
- FPS or temporal stride.
- Resize dimensions and interpolation.
- Grayscale conversion.
- Pixel normalization.
- Input and prediction lengths.
- Any augmentation, applied consistently across all frames in a clip.

## KTH Sequence Boundaries and Known Missing Video

### 1. Sequence Boundaries (`00sequences.txt`)
Each raw KTH `.avi` video consists of 4 separate takes/performances separated by camera cuts or static pauses.
- The official `00sequences.txt` file specifies 1-based inclusive frame ranges for each valid take (e.g., `frames 1-95, 96-185, 186-245, 246-360`).
- The preprocessing pipeline converts each annotated take into an independent sequence directory (e.g. `person01_boxing_d1_take01/`).
- This guarantees that temporal sliding windows (`input_frames` + `output_frames`) strictly stay inside an active performance and never cross camera cuts.

### 2. Known Missing Video: `person13_handclapping_d3`
- Official KTH distributions omit `person13_handclapping_d3_uncomp.avi`.
- This is officially marked as `*missing*` in `00sequences.txt`.
- There are **599 available AVI files** (out of 600 combinations) on disk, generating **2,391 annotated takes**.
- The pipeline acknowledges this missing file and never fabricates synthetic frames for it.

### 3. Subject-Disjoint 3-Way Split Protocol
- **Train:** Subjects 1–8 (768 takes)
- **Validation:** Subjects 9–16 (760 takes)
- **Test:** Subjects 17–25 (863 takes)
- **Total:** 2,391 takes.
- *Comparison with literature:* The original paper (Schuldt et al., 2004) and common video benchmarks (Srivastava et al., 2015) used a 16-train / 9-test split without a partitioned validation set. We use a strictly disjoint 8/8/9 split to ensure early stopping and model selection are conducted on validation data without test-set leakage. All reported metrics should be compared against models trained on this exact partition.


## Raw-video preprocessing commands

Install the optional video decoder:
```bash
pip install -e ".[video]"
```

Build a manifest:
```bash
python scripts/build_manifest.py --root /path/to/raw_kth --out data/manifest.json
```

Review all rows and manually map unknown subject IDs. For the reviewed KTH subject split:
```bash
python scripts/build_manifest.py --root /path/to/raw_kth --out data/manifest.json --assign-kth-splits
```
The automatic parser is only a starting point; do not use it until you have inspected the JSON and corrected paths/subjects/splits.

Convert reviewed videos to frame directories:
```bash
python scripts/preprocess_videos.py --manifest data/manifest.json --out data/processed/kth64 --size 64 --sample-every 1 --min-frames 20
python scripts/check_data.py --root data/processed/kth64
```
