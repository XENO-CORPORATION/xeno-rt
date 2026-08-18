# Qwen3.6 stable-graph MTP shape screen

This record re-screens proposal depth and draft-vocabulary prefix size after
the stable verifier graph and heterogeneous projection schedule changed the
cost curve. It is a negative-control record: the retained depth-eight,
65,536-row configuration remained best.

## Registered tuple

- GPU: NVIDIA GeForce RTX 4090, 24 GB, 450 W limit
- remote worker: RunPod pod `b3xo3ohcu4uw1b`
- model: `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes
- model SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- source base: `77911ff37ee8a8e94c11815726a4008dd949f1e0` plus the recorded workspace candidate
- prompt: `Write the numbers from 1 to 100 in order, separated by commas, and do not stop early.`
- decode: greedy, seed 424242, 64 output tokens, F32 KV, stable verifier graph
- repetitions: six per shape; the first was discarded and five retained

## Results

| Depth | Draft rows | Mean tok/s | Min-max | Accepted / drafted | Verify batches |
|---:|---:|---:|---:|---:|---:|
| 6 | 65,536 | 121.9919 | 121.4967-122.2423 | 52 / 70 | 12 |
| **8** | **65,536** | **149.5396** | **148.3755-150.0233** | **55 / 68** | **9** |
| 10 | 65,536 | 126.4838 | 125.3834-126.8674 | 54 / 94 | 10 |
| 12 | 65,536 | 120.1741 | 119.9949-120.2520 | 54 / 109 | 10 |
| 8 | 57,344 | 127.1838 | 126.7028-127.6868 | 53 / 77 | 11 |
| 8 | 73,728 | 147.7185 | 147.4883-147.9299 | 55 / 68 | 9 |
| 8 | 81,920 | 145.6039 | 145.4150-145.8012 | 55 / 68 | 9 |

Every retained run completed without an error. Smaller row prefixes reduced
coverage enough to require more target windows. Larger prefixes preserved
acceptance but added projection work. Deeper proposals added draft cost without
enough accepted target tokens. No parameter change was admitted.

## Reproduction

```bash
XRT_CLI_BIN=/workspace/xeno-rt/target/release/xrt-cli \
XRT_QWEN_MTP_SCREEN_DEPTHS="6 8 10 12" \
XRT_QWEN_MTP_SCREEN_VOCAB_ROWS="57344 65536 73728 81920" \
scripts/screen-qwen36-mtp-shape.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/profiles/xrt-mtp-shape-final \
  6
```

The raw JSON files are the canonical samples. Their hashes are available with
`sha256sum *.json`; the checked-in files are immutable evidence, not a product
support declaration.
