# Experiment protocol

## Primary comparison
- Model A: compact classical ConvLSTM (pipeline baseline).
- Model B: same classical model with optional quantum-context branch.
- Future work: official THUML PredRNN baseline, if integrated and configured faithfully.

## Fairness controls
- Same train/validation/test split.
- Same preprocessing, input/output horizon and evaluation code.
- Comparable optimization budget and early-stopping rule.
- Record random seed, model parameters, training time, GPU and software versions.
- Repeat at least three seeds if compute allows; if not, clearly label the result exploratory.

## Metrics
- MSE and MAE on normalized [0,1] pixels.
- PSNR with data range 1.0.
- SSIM should be calculated per predicted frame with an explicit data range and then averaged; add and test the chosen implementation before reporting it.
- Report metrics by forecast horizon, not only a single pooled average.
- Do not use classification accuracy/precision/recall/F1 for pixel regression.

## Result table template
| Model | Seed | Horizon | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ | GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Classical | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Quantum-context | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Never fill this table with toy smoke-test numbers as if they were KTH results.
