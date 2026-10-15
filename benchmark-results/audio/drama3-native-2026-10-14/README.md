# Native Drama 3 development evidence

These receipts establish scoped native execution. **They do not admit a public
model/backend/tier or establish industry-grade performance/control fidelity.**

- The complete pinned checkpoint was verified by the shared bundle installer.
- Prompt rows match Fish's actual pinned prompt builder, not just a second local
  implementation. The upstream builder was isolated for reference evaluation.
- Codec and encoder checks use independent PyTorch operators over real trained
  tensors. They do not execute Fish's packaged serving executable.
- `isolation/` retains cancellation/frame-limit discard and identical repeated
  utterances. Its runner stopped on an unexpected two-speaker frame limit.
- `controls/` retains the second runner's complete result, including the failing
  unreferenced speaker case and the two-reference output that says only the
  first line. Its summary intentionally fails. The Chinese ASR receipt comes
  from an English-only verifier and cannot qualify Chinese speech.
- `cpu/` retains full CPU-only execution: the same waveform hash as the GPU
  runs, with all GPU admission counters at zero. Its performance is unqualified.
- `fused/` and `pooled/` retain warm/cold request timings and GPU lease cleanup.
  The GPU arena byte count is a reservation, not a physical peak measurement.

The initial receipts use temperature 1.0 / top-p 0.9. That adapter-default
mismatch was later corrected to Fish SDK temperature 0.7 / top-p 0.7; do not
read the historical control failures or timings as current-default results.

Later runs use eight Rayon threads. The initial run's thread configuration was
not retained, so the progression is not an isolated causal benchmark. No public
speed claim follows from this directory. Model verification/loading and
generation times are reported separately.

Large weights, executable builds and WAV/F32 files stay outside the source tree.
The local retained WAVs are under `X:/tmp/drama3-native-qualification-*` and their
hashes are recorded in the receipts. No reference voice from an external person
was used: reference tests condition on this model's own earlier generated audio.

See [implementation and open admission work](../../../docs/DRAMA3-NATIVE-ISG.md).
