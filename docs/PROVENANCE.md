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
- Compact ConvLSTM baseline in `qpred/models/convlstm.py`: small pipeline baseline for smoke tests.
- PredRNN-V2 implementation in `qpred/models/predrnn_v2.py`:
  - Architecture: SpatioTemporal LSTM cells with dual memory flows (temporal memory $C_t^l$ and zigzag spatiotemporal memory $M_t^l$).
  - Memory-decoupling auxiliary loss: Evaluates absolute cosine similarity between temporal ($\Delta C$) and spatiotemporal ($\Delta M$) memory increments with numerical safeguard ($\epsilon = 10^{-7}$).
  - Reverse scheduled sampling: Dual-phase curriculum learning strategy following official THUML `run.py`:
    - Context/encoder schedule ($r_\text{eta}$): Starts at 0.5 and exponentially increases to 1.0 between `r_sampling_step_1` and `r_sampling_step_2` with rate $\alpha$, encouraging early encoder spatiotemporal representation learning conditioned on imperfect predictions.
    - Future/decoder schedule ($\text{eta}$): Starts at 0.5 and linearly decreases to 0.0 across the same step window, gradually transitioning rollout from teacher-forced ground truth to autoregressive generation.
    - Evaluated in strictly autoregressive mode during validation/inference with zero ground-truth leakage.
  - Deviations and window-based adaptations from upstream official code:
    1. Upstream `run.py` generates flat masks for continuous sequences $(T_{\text{total}}, B, C, H, W)$. Here, data is partitioned into separate batch-first tensors $x \in (B, T_\text{in}, C, H, W)$ and $y \in (B, T_\text{out}, C, H, W)$. Sampling masks are adapted into `mask_context` $(B, T_\text{in} - 1)$ for context transitions and `mask_future` $(B, T_\text{out})$ for decoder rollout steps.
    2. Operates on standard PyTorch `(B, T, C, H, W)` batch-first layout instead of upstream legacy `(T, B, C, H, W)`.
    3. Uses pure PyTorch operations compatible with Python 3.11 and PyTorch 2.x without deprecated dependencies (`scipy.misc`, `Variable`, legacy CUDA extensions).
    4. Standalone modular loss and reverse scheduled sampling integrated into clean training loop.
- Quantum-context prototype in `qpred/models/quantum_context.py`: optional research branch using PennyLane's default.qubit simulator.
- Data and training scaffold: project-specific.
- Pretrained weights: none included by default.
- Trained results: none included by default.

## Attribution wording template
"This project implements a custom video-prediction research pipeline and a quantum-context experimental variant. It is inspired by and evaluated in relation to THUML PredRNN. Any reused code, equations, checkpoints, and modifications are identified in the accompanying provenance notes and retain their applicable licenses. The quantum component is simulated on classical hardware; no quantum advantage is claimed."

Do not fill in a license claim without checking the exact upstream license file and obligations.
