# Start here — Antigravity workflow

## Before opening Antigravity
1. Extract this ZIP into a working folder.
2. Keep your original `predrnn-pytorch-master.zip` separately. Unpack it beside this folder or clone upstream as explained in `upstream/README.md`.
3. Open the project folder in Antigravity.
4. Paste the global tutor prompt from `docs/ANTIGRAVITY_PROMPTS.md`.
5. Ask for **Prompt 1 only**. Review all proposed changes before accepting them.

## Local setup
Use Python 3.10–3.12 for the first setup unless the selected PyTorch build specifically supports another version. The project's runtime dependency is PyTorch; install a build appropriate for your OS and CUDA setup from the official PyTorch instructions first if pip's default is unsuitable.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For raw video decoding:
```bash
pip install -e ".[video]"
```

For the optional quantum simulator:
```bash
pip install -e ".[quantum]"
```

## Cheap checks on your laptop
```bash
python scripts/make_toy_dataset.py --out data/toy --sequences 24 --frames 20 --size 64
python scripts/check_data.py --root data/toy
pytest -q
python -m qpred.train --config configs/smoke.yaml --data-root data/toy --run-dir runs/smoke --device cpu
```

These commands use synthetic moving-square data and a tiny model. They prove the software path runs, not that it predicts real videos well.

## KTH preprocessing (after manual review of manifest)
```bash
python scripts/build_manifest.py --root /path/to/raw_kth --out data/manifest.json
```
Review every subject/action mapping. If the parser leaves unknown IDs, fix them manually. Only after reviewing the mapping and setting the subject-disjoint split:
```bash
pip install -e ".[video]"
python scripts/preprocess_videos.py --manifest data/manifest.json --out data/processed/kth64 --size 64 --sample-every 1 --min-frames 20
python scripts/check_data.py --root data/processed/kth64
```
Do not train until the data validation gate in `docs/PROJECT_PLAN.md` passes.

## Push to GitHub before expensive training
Follow `docs/COMMIT_PLAN.md` for seven suggested commits. Do not upload raw data, frame folders, checkpoints, secrets, or credentials. Run `git status`, `git diff --check`, and `pytest -q` before each push.

## Training on a cloud GPU
Use `docs/CLOUD_GPU.md`. Run a 1–3 epoch pilot first, save checkpoints to persistent storage, and record the GPU/software/config/git hash. Never assume a notebook session's disk persists.

## Honest claims
Read `docs/ORIGINALITY_AND_WEIGHTS.md` and `docs/PROVENANCE.md`. This starter does not contain a pretrained checkpoint or real KTH metrics. The compact baseline is not the official PredRNN implementation.
