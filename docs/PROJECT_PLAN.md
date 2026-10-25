# Project plan: stages, gates, and deliverables

## Research question
Does adding a compact quantum recurrent/context pathway to a classical spatiotemporal video predictor improve future-frame quality under a controlled experimental setup?

## Avoid scope creep
First make a reliable classical baseline and data pipeline. Add the quantum branch only after all classical data/model tests pass. Do not begin full training until the pre-training checklist is signed off.

## Stage 0 — Define the experiment
- Dataset: KTH as primary, Moving MNIST only for debugging.
- Main setting: 10 input frames -> 10 future frames at 64x64 grayscale initially.
- If reporting horizons 1, 5, 10, and 20, the target sequence must contain at least 20 frames; either use 10 input -> 20 output or report only horizons that fit the target.
- Metrics: MSE, MAE, PSNR, SSIM. Report metric implementation and averaging convention.
- Subject split: use a documented subject-disjoint split. This scaffold uses IDs 1–8 train, 9–16 validation, 17–25 test as an example mapping; verify that it matches your actual KTH archive/version and course protocol before locking it.
- No test-set tuning. Select checkpoints on validation only.

## Stage 1 — Upstream audit and provenance
- Clone or unpack the official `thuml/predrnn-pytorch` repository separately.
- Read its README, license, model code, data loader, training/evaluation scripts, checkpoint loading.
- Record exact commit hash and license in `docs/PROVENANCE.md`.
- Do not copy upstream code into this starter without preserving notices and license obligations.
- Establish whether a trained checkpoint is actually available. A code repo is not a checkpoint.

## Stage 2 — Environment
- Pin a tested Python/PyTorch/CUDA combination on the GPU runtime.
- Local CPU environment is for tests and toy data, not full KTH training.
- Save `pip freeze` and hardware details with each experiment.

## Stage 3 — Dataset audit and preprocessing
- Keep raw data outside Git (`data/raw/` ignored).
- Create a manifest with path, subject ID, action, source FPS, frame count, split.
- Manually review filenames the parser cannot map to a subject.
- Split at subject/video level first, then extract frames and create sliding windows.
- Decode videos consistently; grayscale; resize; scale to [0,1].
- Do not apply independent random transforms to individual frames. Any augmentation must be temporally consistent across the entire clip.
- Inspect random clips, frame order, dimensions, and normalized values.
- Store exact preprocessing parameters and source data version.

## Stage 4 — Feature extraction
For end-to-end frame prediction, a separate hand-engineered feature extraction stage is not mandatory. The CNN encoder learns spatial features jointly with the recurrent model. You may add optical flow as a diagnostic or separate baseline, but do not accidentally leak future frames into the input features.
- Verify encoder feature shapes and gradient flow.
- Save a visualization of input and target frame strips.
- If using a pretrained image encoder, explicitly name the source, checkpoint, license, and why it is suitable; otherwise initialize randomly.

## Stage 5 — Classical baseline
- Start with the included small ConvLSTM to validate the entire pipeline.
- Then integrate/reproduce the official PredRNN baseline as a separate experiment, with its own config and checkpoint naming.
- Compare model parameter counts, input/output shapes, runtime, and training setup.
- Do not call this compact baseline an exact PredRNN reproduction.

## Stage 6 — Quantum component
- Install `pip install -e '.[quantum]'` only after the classical model passes.
- Test the quantum layer on a tiny batch and verify finite forward output and non-zero gradients.
- Keep qubit count and circuit depth small initially (e.g. 4 qubits, depth 2).
- Run classical and quantum variants with matched data splits and comparable tuning budgets.
- Report simulator runtime and parameter count. A simulator does not demonstrate quantum hardware advantage.

## Stage 7 — Pre-training validation gate
Required before expensive training:
- [ ] Manifest reviewed; no unmapped subject IDs.
- [ ] Subject/video splits fixed and written to versioned metadata.
- [ ] No sequence appears in multiple splits.
- [ ] Preprocessing visual inspection completed.
- [ ] Dataset sample shapes and value range verified.
- [ ] Model output shape equals target shape.
- [ ] Loss decreases on a tiny overfit test.
- [ ] Gradients finite; no NaNs.
- [ ] Checkpoint save/resume works.
- [ ] Validation metrics are computed without gradient updates.
- [ ] Test split is not used for model selection.
- [ ] Config, seed, environment, and provenance are saved.
- [ ] GPU memory and step time estimated on a small subset.

## Stage 8 — Full training on cloud GPU
- Run a short pilot (1–3 epochs) first.
- Inspect train/validation loss and sample predictions.
- Adjust batch size/resolution before long run.
- Use validation loss for best checkpoint; keep last checkpoint for resuming.
- Save checkpoints to persistent cloud storage, not only ephemeral runtime disk.
- Stop if training loss is flat, NaN, or validation quality collapses; debug before spending more GPU time.

## Stage 9 — Evaluation and ablation
- Freeze the final model based on validation.
- Evaluate test exactly once per final configuration.
- Report per-horizon MSE/MAE/SSIM/PSNR and mean ± standard deviation across at least 3 seeds if compute allows.
- Ablations: classical only, quantum component on, qubits/depth, optional quantum branch removed.
- Include confidence/limitations; never claim improvement if the results do not show it.

## Stage 10 — Demo and final packaging
- Dashboard loads only real checkpoint and real test examples.
- Clearly label any demo/toy output as synthetic.
- Include commands, config, dataset setup, hardware, limitations, license/provenance.
