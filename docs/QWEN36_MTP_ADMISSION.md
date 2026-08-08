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

Set `XRT_QWEN_MTP=on` to opt into CUDA greedy drafting. The predictor recursively
proposes at most three tokens and the complete target model verifies every
proposal through the existing transactional hybrid KV/DeltaNet rollback path.
Non-greedy requests remain on target-only or prompt-lookup decoding until exact
speculative rejection sampling is implemented.

This lane must remain disabled by default until a pinned real artifact passes:

1. target-only parity with the same artifact;
2. deterministic MTP-on/off output parity for greedy decoding;
3. accepted-boundary KV and recurrent-state rollback tests;
4. no OOM within the documented 24 GB RTX 4090 profile;
5. repeated throughput measurements showing a material decode improvement; and
6. the ordinary text runtime correctness, compatibility, and packaging gates in
   `RUNTIME_DOMAINS.md`.

