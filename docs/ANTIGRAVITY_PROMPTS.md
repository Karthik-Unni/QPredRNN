# Antigravity prompts: use one stage at a time

Paste one prompt at a time. Review the diff and run the stated tests before proceeding. Do not ask the agent to implement the entire research project in one prompt.

## Global instruction (paste first)
You are my coding tutor for a research project called QPredRNN. Work in small, reviewable stages. Never claim a test passed unless you ran it and show the command/result. Do not invent metrics. Do not download/use pretrained weights unless I explicitly approve the exact checkpoint and its source. Keep raw data, checkpoints and secrets out of Git. Preserve all upstream license notices and attribution. Clearly distinguish this repository's compact ConvLSTM prototype from official THUML PredRNN. Do not claim quantum advantage. Before changing files, summarize the plan and list files to modify. After changes, explain the code and provide tests and commands. Stop after the requested stage; do not proceed automatically.

## Prompt 1 — repository and provenance audit (no edits)
Inspect this repository and the official `thuml/predrnn-pytorch` source if it is present locally. Do not modify files. Explain the data flow, model cells, training/evaluation code, checkpoint format, licenses, dependencies, and whether any trained checkpoint is included. Identify any missing pieces and propose a safe integration plan. Distinguish code reuse from weight reuse.

## Prompt 2 — environment
Check Python and dependency compatibility. Do not install CUDA-specific PyTorch blindly. Give separate commands for Windows PowerShell and Linux/macOS. Add exact setup instructions to docs. Do not start training.

## Prompt 3 — dataset manifest
Implement a manifest with relative path, video/sequence ID, subject, action, frame count, FPS, and split. Do not silently guess subject IDs; report unknowns for manual review. Split subjects/videos before creating windows. Add leakage tests.

## Prompt 4 — preprocessing and visual QA
Implement a deterministic preprocessing command for the dataset format we actually have. Add grayscale/resize/normalization and temporally consistent clip augmentation only if explicitly enabled. Save a metadata file. Create contact sheets for random train/validation clips. Do not invent a KTH directory structure; inspect it first.

## Prompt 5 — dataset validation
Add tests for frame order, shape, dtype, range [0,1], sequence lengths, and split disjointness. Add a command to print sample shapes and counts. No training yet.

## Prompt 6 — classical baseline
Implement or adapt the classical baseline and compare its tensor shapes with the official PredRNN model. Do not call the model exact PredRNN unless its architecture and training protocol match. Add a tiny overfit test and checkpoint save/load test.

## Prompt 7 — quantum branch
Add an optional PennyLane quantum recurrent/context component behind a config flag. Keep qubits and depth small. Test forward shape, finite outputs, gradient flow and runtime. Keep the classical path runnable without PennyLane. Do not claim the quantum version is better until measured.

## Prompt 8 — training validation gate
Add train/validation loops, metrics, best/last checkpoints, resume, early stopping if justified, and JSONL logs. Test resume and verify validation does not update gradients. Add a pre-training checklist. Do not run full dataset training.

## Prompt 9 — cloud execution
Prepare a notebook or script that installs dependencies, checks GPU, runs tests, runs a short pilot, saves artifacts to persistent storage, and records environment + git commit. Do not put credentials in the notebook.

## Prompt 10 — evaluation and dashboard
Implement held-out test evaluation at requested horizons, sample prediction exports, and a dashboard that refuses to show metrics if result files are missing. No fake/sample numbers presented as experiment results.
