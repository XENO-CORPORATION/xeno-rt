# Qwen3.6 NextN/MTP admission

Status: experimental, disabled by default.

XENO RT recognizes integrated Qwen3.5-compatible GGUF artifacts whose physical
block count includes appended `nextn_predict_layers`. The target decoder trunk
and appended predictor blocks are tracked separately so the predictor can never
be executed as an ordinary target layer.

The first execution lane targets the one-layer Qwen3.6 NextN layout:

- `qwen35.nextn_predict_layers = 1`;
- one full-attention predictor block appended after the target trunk;
- `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj`, and
  `nextn.shared_head_norm` tensors; and
- shared token embeddings and output projection.

Set `XRT_QWEN_MTP=on` to opt into CUDA greedy drafting. Draft depth defaults to
one token and can be bounded from one through three with
`XRT_QWEN_MTP_MAX_DRAFT_TOKENS`; deeper recursion requires separate performance
admission because rollback cost increases with depth. The complete target model
verifies every proposal through the existing transactional hybrid KV/DeltaNet rollback path.
Non-greedy requests remain on target-only or prompt-lookup decoding until exact
speculative rejection sampling is implemented.

Full greedy parity requires verification through the configured target sampler,
including repetition penalty and EOS handling. Raw-logit argmax is not an
equivalent verifier when request-time sampling transforms are active.

This lane must remain disabled by default until a pinned real artifact passes:

1. target-only parity with the same artifact;
2. deterministic MTP-on/off output parity for greedy decoding;
3. accepted-boundary KV and recurrent-state rollback tests;
4. no OOM within the documented 24 GB RTX 4090 profile;
5. repeated throughput measurements showing a material decode improvement; and
6. the ordinary text runtime correctness, compatibility, and packaging gates in
   `RUNTIME_DOMAINS.md`.

## RTX 4090 result (2026-08-08)

Commit `4ec4f4a` was exercised on a 24 GB RTX 4090 with the pinned
`Qwen3.6-27B-Q4_K_S.gguf` artifact recorded in
`benchmark-results/text/qwen36-mtp-rtx4090-q4_k_s-2026-08-08.json`.

- The 16-token CLI A/B produced byte-identical output and the three-run
  benchmark produced the same eight-token output in every target and MTP run.
- The CUDA transaction test passed with the default `1.1` repetition penalty.
- The real model loaded without OOM and peaked at 21,157,251,872 tracked bytes
  with MTP enabled.
- The acceptance sample accepted 5 of 10 drafts (50%).
- Warm decode fell from 7.276 to 3.544 tokens/second, a 51.29% regression.

The performance admission gate therefore failed. MTP remains experimental and
disabled by default. Before rerunning admission, remove enough predictor and
verifier overhead to demonstrate a material speedup, then run a multi-prompt
quality/parity suite and the remaining text release gates. Human quality review
is not useful until those automated performance and coverage gates pass.
