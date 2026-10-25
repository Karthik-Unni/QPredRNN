# Upstream PredRNN source — integration instructions

The original THUML archive supplied by the project owner is intentionally not copied into this scaffold. Keep it as a separate, unmodified directory until you have reviewed its license and notices. The supplied archive did not contain an obvious LICENSE file in its top-level file list, so do not assume that redistribution is automatically permitted.

Recommended local layout:
```text
workspace/
  QPredRNN_Antigravity_Project/
  pred rnn-pytorch-master/       # unpacked separately, unmodified
```

Or clone the upstream repository separately:
```bash
git clone https://github.com/thuml/predrnn-pytorch.git ../predrnn-pytorch-upstream
```

Then ask Antigravity to audit the upstream repository and prepare a design for integrating it as a separately named baseline. Do not copy upstream files into `qpred/` blindly. Preserve the upstream source tree, exact commit hash, notices, and license terms. Add a provenance entry in `docs/PROVENANCE.md` before publishing any derived code.

The included `qpred.models.convlstm.VideoConvLSTM` is a small pipeline baseline; it is **not** the official PredRNN or PredRNN-V2 architecture. The full upstream integration is a deliberate stage, not something this starter falsely claims has already completed.
