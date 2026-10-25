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

## KTH caution
Filename patterns differ across mirrors/archives. Never trust an automatic subject parser without reviewing its output. The helper split function is an example only, not an authoritative guarantee for every archive. If a subject cannot be mapped, stop and manually resolve it.


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
