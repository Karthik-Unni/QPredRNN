# Pretraining, validation, and what those terms mean here

## Three different choices
1. **Random initialization / train from scratch:** model parameters start randomly initialized and are learned on your dataset. You can still use PyTorch and architecture ideas from prior work, but must acknowledge code/design reuse.
2. **Pretrained checkpoint initialization:** load weights trained by someone else, then train your custom model. This is transfer learning/fine-tuning or continued training, even if every layer is trainable.
3. **Pretraining your own model:** first train on a source dataset/task, then initialize a downstream/custom task with that checkpoint. This requires a separate documented pretraining stage and source dataset.

A repository containing model code is not a trained checkpoint. This starter contains no pretrained checkpoint. Do not say "pretrained PredRNN weights" unless you have an actual checkpoint file and can verify its origin and training setup.

## Recommended plan for a laptop-limited project
- Locally: toy-data smoke test, unit tests, data manifest validation, small shape/gradient checks.
- On GPU: train a small classical baseline on the actual training split.
- On GPU: train the quantum variant with matched data and compute budget.
- Optional pretraining: only if there is a suitable, license-compatible checkpoint and its architecture is compatible. Document every reused tensor group.
- Validation split: choose architecture/checkpoint/hyperparameters.
- Test split: final held-out evaluation only.

## What to validate before full training
- Input and target shapes match model contract.
- No subject/video leakage.
- Pixel range is [0,1].
- Train loss can decrease on a tiny subset (overfit sanity check).
- Gradients and losses are finite.
- Checkpoints save and resume correctly.
- Validation is in `eval()` and does not backpropagate.
- Config and git hash are stored with the run.
- Pilot runtime and VRAM use are acceptable.
