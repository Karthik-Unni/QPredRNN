# Seven-commit GitHub plan (push before full training)

Make each commit coherent and testable. The point is to show development history before expensive training begins.

## Commit 1 — project specification
Files: README, project plan, provenance placeholder, `.gitignore`.
```bash
git add README.md .gitignore docs/PROJECT_PLAN.md docs/PROVENANCE.md
git commit -m "docs: define QPredRNN research scope and provenance"
git push -u origin main
```

## Commit 2 — environment and configs
Files: `pyproject.toml`, `configs/`.
```bash
git add pyproject.toml configs
git commit -m "build: add reproducible environment and experiment configs"
git push
```

## Commit 3 — data and split validation
Files: `qpred/data/`, `scripts/check_data.py`, tests for splits.
```bash
git add qpred/data scripts/check_data.py tests
git commit -m "data: add sequence dataset and split integrity checks"
git push
```

## Commit 4 — classical model and metrics
Files: `qpred/models/convlstm.py`, `qpred/metrics.py`, model tests.
```bash
git add qpred/models/convlstm.py qpred/metrics.py tests
git commit -m "model: add compact ConvLSTM baseline and metrics"
git push
```

## Commit 5 — training and checkpoint lifecycle
Files: `qpred/train.py`, smoke config.
```bash
git add qpred/train.py configs/smoke.yaml
git commit -m "train: add validation loop and resumable checkpoints"
git push
```

## Commit 6 — quantum branch and tests
Files: `qpred/models/quantum_context.py`, optional-dependency docs/tests.
```bash
git add qpred/models/quantum_context.py pyproject.toml docs
git commit -m "model: add optional quantum-context research branch"
git push
```

## Commit 7 — smoke workflow and cloud-training protocol
Files: toy generator, README, Antigravity prompts, cloud guide, experiment protocol.
```bash
git add scripts/make_toy_dataset.py README.md docs
git commit -m "docs: add smoke workflow and cloud training protocol"
git push
```

**Important:** Do not commit real videos, generated frame folders, checkpoints, API keys, or cloud credentials. Commit code, configs, documentation, and small synthetic test fixtures only. If you choose a different sequence, keep the same principle: each commit passes tests and is pushed before full training.


## Before any expensive training

Use the seven commits above to publish the scaffold, tests, docs, and model code. The upstream PredRNN integration should be a separate reviewed change after the upstream license/provenance audit. Do not push upstream source files unless you have confirmed redistribution terms. Push code before full training; commit only experiment configs and scripts, not private data or checkpoints.
