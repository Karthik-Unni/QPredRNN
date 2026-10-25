# Training without a powerful laptop

## Recommended workflow
Use your laptop for editing, code review, data inspection, and smoke tests. Use a GPU notebook/VM for pilot and full training. Candidate platforms include Google Colab, Kaggle notebooks, RunPod, Lambda, or a university GPU server. Availability, quotas, pricing, and GPU types change, so check current terms before choosing.

## Before moving data
1. Confirm dataset license/terms allow uploading to the chosen service.
2. Upload only the data required for the experiment.
3. Keep data private if the dataset or university rules require it.
4. Never put credentials or tokens in GitHub.

## Cloud checklist
- [ ] Clone the GitHub repository.
- [ ] Select a GPU runtime and check `torch.cuda.is_available()`.
- [ ] Install the same project version and optional dependencies.
- [ ] Mount persistent storage or configure checkpoint upload.
- [ ] Run toy smoke test first.
- [ ] Run a tiny KTH pilot with a small subset and batch size 1–2.
- [ ] Record GPU model, VRAM, package versions, config and git commit hash.
- [ ] Launch training; periodically save `last.pt`, `best.pt`, and `metrics.jsonl`.
- [ ] Copy checkpoints and logs to persistent storage.
- [ ] Verify copied artifacts can be downloaded before ending the session.

## Commands
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd QPredRNN_Antigravity_Project
python -m pip install --upgrade pip
pip install -e ".[dev]"
# Install quantum extras only for the quantum run:
pip install -e ".[quantum]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
python -m qpred.train --config configs/kth_classical.yaml --data-root /path/to/preprocessed_data --run-dir /persistent/runs/kth_classical --device cuda
```

For quantum model:
```bash
python -m qpred.train --config configs/kth_qpredrnn.yaml --data-root /path/to/preprocessed_data --run-dir /persistent/runs/kth_quantum --device cuda
```

## Compute-saving tactics
- Begin at 64x64, 10 input -> 10 output, small hidden size.
- Use 1–3 epoch pilot; estimate time per epoch and validation behavior.
- Use batch size 1–4 if memory is tight.
- Reduce workers if notebook crashes.
- Use mixed precision only after confirming numerical stability; quantum simulator/autograd may have limitations.
- Avoid huge grid searches. Tune classical baseline first, then test a small quantum ablation.
- Do not assume a cloud session persists. Checkpoint frequently and use persistent storage.
