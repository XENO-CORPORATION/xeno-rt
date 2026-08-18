# Device-resident MTP draft chain rejection

## Outcome

The device-resident draft-chain candidate preserved exact generated-token
parity on all 12 frozen Qwen3.6 greedy-admission cases, but did not improve
throughput and was removed.

| Arm | Mean tok/s | Median | Minimum | Maximum | Draft time | Verify time |
|---|---:|---:|---:|---:|---:|---:|
| Retained control | 127.3386 | 120.6250 | 80.9977 | 197.5432 | 1.446073 s | 4.559146 s |
| Device draft chain | 127.3270 | 119.0574 | 81.7851 | 195.8455 | 1.457831 s | 4.535992 s |

Both arms drafted 1,161 tokens, accepted 639, and executed 125 verification
windows. The candidate replaced every per-token MTP argmax download and decode
parameter upload with a stream-ordered device kernel, then downloaded the
complete draft once per window. Its -0.0091% mean change is neutral-to-negative
and the draft phase itself was 0.81% slower in this matched run.

## Conclusion

The per-token readback is a causal boundary but not the throughput limiter:
the next MTP step cannot start until the previous argmax exists, and the tiny
transfer cost is hidden beneath predictor compute. The rejected source and PTX
were removed. Further work should target target-verifier weight bandwidth or a
proposal method that increases accepted tokens per verifier pass.

## Scope

- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Model: Qwen3.6-27B Q4_K_S GGUF
- Corpus: `qwen36-greedy-admission-v1`, SHA-256
  `196b64dcdf2c56b9d162080b28e1a8f3385b00454ef2bb32098c2cb11ff0ce25`
- Fixed depth: 10
- Exact output-token parity: passed, 12/12 cases

