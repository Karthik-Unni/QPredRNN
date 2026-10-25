# Originality, pretrained weights, and honest claims

## Can the project be custom?
Yes. You can design a custom architecture using an established framework or architecture as a base. Your contribution may be the new recurrent/quantum component, model wiring, data pipeline, training protocol, ablations, or evaluation.

## What the terms mean
- **Framework/code reuse:** use PyTorch or adapt an upstream implementation. This is not weight reuse by itself.
- **Architecture reuse:** retain PredRNN/ST-LSTM equations or memory-flow design. Cite and describe it.
- **Pretrained weights:** load tensors from an existing checkpoint trained by someone else. Disclose the checkpoint source and what was initialized from it.
- **Training from scratch:** initialize the model parameters randomly and train on your dataset. This does not mean the architecture or all source code was invented independently.
- **Train end-to-end from pretrained initialization:** start from an existing checkpoint, leave layers trainable, and train the custom model on your dataset. This is not strictly training from scratch.

## What an examiner can discover
Do not rely on hiding the source. Git history, code structure, architecture equations, checkpoint metadata, and similarity checks can reveal reused components. A checkpoint may also contain names/metadata that identify its origin. The correct solution is accurate attribution, not trying to disguise reuse.

## Suggested defensible project description
"We implemented a custom video-prediction research pipeline inspired by THUML PredRNN and evaluated an optional quantum-context recurrent component. The classical recurrent baseline and quantum branch, training setup, preprocessing, and evaluation protocol are documented. Any reused code, equations, and checkpoints are identified in the provenance notes."

If a real upstream PredRNN checkpoint is used, replace the last sentence with a specific disclosure: source URL, commit/checkpoint identifier, tensor groups loaded, layers newly initialized, and which layers were trainable during training.

## Checkpoint compatibility
A checkpoint can only load tensors whose names and shapes match, or tensors that you explicitly map. Changed hidden sizes, gate order, layer count, or cell definitions can make weights incompatible or semantically wrong. New quantum layers need their own initialization. Matching tensor shapes alone does not prove that the weights have the intended semantics.

## Research integrity
Do not claim the entire model was independently invented if you reused PredRNN code or design. Do not claim quantum advantage from a classical simulator. Report negative or neutral results as well as positive ones.
