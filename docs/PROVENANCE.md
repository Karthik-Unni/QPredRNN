# Provenance and third-party components

Fill this out before publishing or presenting results.

## Upstream reference
- Project: THUML PredRNN-PyTorch (NeurIPS 2017, TPAMI 2022)
- Repository: https://github.com/thuml/predrnn-pytorch
- Archive reviewed: `predrnn-pytorch-master.zip` (TPAMI 2022 PredRNN-V2 release)
- License and notice reviewed: No formal LICENSE file in upstream archive; author attribution preserved (`yunbo`, `gaozhifeng` - Tsinghua University). Treated as external academic reference.
- Files or equations adapted: SpatioTemporalLSTMCell dual memory flow equations ($C_t$ and $M_t$ zigzag memory) and memory-decoupling loss.
- Modifications made: Standard PyTorch $(B, T, C, H, W)$ tensor format, 3-way subject-disjoint KTH split (01–08 train, 09–16 val, 17–25 test), modular quantum recurrent pathway.

## This repository
- Compact ConvLSTM baseline in `qpred/models/convlstm.py`: implemented in this starter as a small baseline; not claimed to reproduce PredRNN-V2.
- Quantum-context prototype in `qpred/models/quantum_context.py`: optional research branch using PennyLane's default.qubit simulator.
- Data and training scaffold: project-specific.
- Pretrained weights: none included by default.
- Trained results: none included by default.

## Attribution wording template
"This project implements a custom video-prediction research pipeline and a quantum-context experimental variant. It is inspired by and evaluated in relation to THUML PredRNN. Any reused code, equations, checkpoints, and modifications are identified in the accompanying provenance notes and retain their applicable licenses. The quantum component is simulated on classical hardware; no quantum advantage is claimed."

Do not fill in a license claim without checking the exact upstream license file and obligations.
