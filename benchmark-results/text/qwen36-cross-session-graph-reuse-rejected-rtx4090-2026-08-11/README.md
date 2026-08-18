# Qwen3.6 cross-session graph reuse rejection (RTX 4090, 2026-08-11)

This experiment retained one complete idle CUDA backend session so its
pointer-bound Qwen MTP verifier graphs could be reused by the next independent
request. A companion row-aware verifier cache prevented a short tail window
from evicting the full-depth graph family.

The structural objective was partially achieved: verifier captures fell from
two per corpus case to four total (two pointer generations at each of two KV
capacities). The candidate nevertheless failed correctness after the retained
KV cache grew. The fourth case collapsed to 5 accepted of 573 drafted tokens,
and two later cases failed with invalid encoded row-argmax indices. Corpus mean
fell to 41.7330 tok/s and exact token parity failed.

Cross-request backend-session pooling is rejected and removed. Qwen graph
executables are now explicitly reset when a session's logical state is
cleared. Row-aware graph identity remains useful and safe inside one live
session, where it prevents a tail shape from launching a graph captured for a
different row count.

This is negative benchmark evidence, not an advertised capability.
