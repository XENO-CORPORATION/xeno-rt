# Interim multilingual, numerical and delivery evidence

**NOT ISG; model admission remains open.** These receipts preserve component
passes and failed qualification without converting either into a model pass.
`inventory.json` binds copied receipts to their source bytes.

Local regression: 453 workspace tests pass with 39 explicit ignores; 68
native-audio, 43 server and seven publisher-enabled CLI tests pass. Rust 1.83
checks the combined server/CUDA and publisher CLI all-targets graph. Formatting,
Python syntax and the unchanged 30-file shared-crate inventory are checked.
Strict clippy fails on the existing approximate LOG2_E literal; capped-warning
clippy completes. Dependency policy retains the unwaived `paste` maintenance
advisory; bans, licenses and sources pass.

Twelve actual-driver FP8/dense tests pass. The independent actual-Triton fixtures
include partial block groups; the synthetic warm expert benchmark measures
Tensor Core and SIMT separately. These prove component behavior, not narrated
quality or whole-model throughput. The prompt comparison still fails its
fidelity checks: 14/26 top-20 log-probability comparisons pass and 24/26 argmaxes
match. The teacher-forced acoustic probe is built; no acoustic oracle execution
or admission is implied by that build receipt.

The earlier ten-language matrix has 13/20 accepted cases at 256 frames/two
attempts and passes three negative controls. It remains a failed matrix.
The later 600-frame/three-attempt restart with a 13 GiB expert cache encounters
native GPU headroom failures before establishing quality; its harness aborts
when the wrong-script control encounters a resource error. Those failures are
retained separately. A smaller 8 GiB cache profile is being evaluated on the
shared card. No complete result from that run is included here.

The dedicated OS-protected publisher signs a locally installed, rehashed
candidate and rejects the wrong signer. Its public identity is tracked;
private material never enters these receipts. No trust-feed rollout, managed
codec delivery, public release or Fish weight publication is established.
MOS, speaker identity, emotion, long narration, final installed lifecycle,
exclusive memory/performance evidence and full numerical qualification remain
open. See the [admission ledger](../../../../docs/DRAMA3-NATIVE-ISG.md).
