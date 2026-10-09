# Constrained Tool Qualification

Scope: xrt-text. Measured 2026-10-05. Source candidate, not a published release.

## Benchmark

The benchmark is sampling-time enforcement, explicit dialect negotiation,
bounded matcher work, preserved schema semantics, non-execution of incomplete
calls, and independent semantic validation.

- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
  uses grammar backends and exposes structured-output controls. Its removed
  `guided_*` request fields and backend-specific regex dialects show why a
  consumer must negotiate a precise contract, not assume a server name implies
  support.
- [SGLang structured outputs](https://docs.sglang.ai/advanced_features/structured_outputs.html)
  supports XGrammar, llguidance and Outlines with explicit backend selection.
  Format support differs by backend; this candidate advertises only GBNF and
  separate streaming support rather than an undifferentiated structured flag.
- [llama.cpp GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
  documents Unicode grammars and warns that repeated nullable expressions can
  make sampling extremely slow. Real-vocabulary qualification here exposed the
  same work explosion; regular character sequences now compile into lexer
  expressions while the 50,000-item matcher bound stays enforced.
- [llguidance schema semantics](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md)
  declares supported schema semantics and explicit departures. This candidate
  rejects compiler warnings and unsupported assertions rather than silently
  claiming exact enforcement. A grammar never replaces tool validation.

Overall: PARTIAL. The local source/backend contract has real-model evidence.
Cross-platform, accelerator, performance baseline, exact-package and release
qualification are separate and remain open.

## Evidence

Commands were executed in the isolated runtime/SDK checkouts:

```powershell
cargo +1.83.0 check -p xrt-server -p xrt-cli -p xrt-capi -p xrt-runtime --locked
cargo test -p xrt-server -p xrt-runtime -p xrt-tokenizer --locked
cargo test -p xrt-workspace-tests --test smoke_e2e grammar_sampling --locked
cargo build -p xrt-workspace-tests --example qualify_patch_grammar --release --locked
node scripts/qualify-patch-grammar-corpus.mjs <qualify_patch_grammar.exe>
node --import ./tests/_setup.mjs scripts/qualify-xrt-tool-grammars.mjs --server-binary <xrt-server.exe> --model <supported.gguf> --output <evidence-directory>
```

- Rust compiler floor: 1.83.0 passed for runtime/server/CLI/C bindings. The new
  transitive indexmap is locked to 2.7.1; 2.14.2 required edition-2024 Cargo.
- Runtime/server/tokenizer: 165 passed, zero failures, eight explicit ignored
  model-fixture cases. Synthetic sampling covers exact output, completion,
  truncation, cancellation and speculation disabled for constrained requests.
- Current CUDA-feature runtime/server compilation passed with the same locked
  dependency graph. This does not prove GPU execution or backend performance.
- Generated Patch grammar: 22 accepted/rejected shape cases agree with the
  SDK strict parser, including deletion, empty addition, move-only update,
  hunk changes, multiple sections, Unicode and CRLF. Duplicate paths and file
  matching remain semantic checks in the strict parser/application layer.
- Workspace CPU integration matrix: 53 passed, zero failures, one explicit
  ignored smoke. Image-feature-disabled files ran zero tests, not qualification.
- SDK final full suite: 1,840 passed, zero failures, 12 explicit skips, after
  the parallel-call wire field and PID-incarnation fixture fix. The final
  focused conformance/socket run passed 58 tests with zero skips. Live gateway,
  external provider and POSIX checks are not inferred from this Windows run.
- Real model: cached Qwen2.5-0.5B-Instruct Q8_0, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Twelve real-model checks passed: default demand-loaded catalogue compilation,
  durable quota settlement, whole-response UTF-8, generated Patch grammar,
  nested required function schemas in JSON/SSE, exact-count mixed/repeated
  custom/function batches in JSON/SSE, budgeted custom-Patch negotiation
  and measured settlement, JSON/SSE exact file edit, JSON/SSE truncation refusal,
  and real AgentLoop execution with canonical-name permission checks. The owned
  server exited.
- Tested server SHA-256:
  `0b48a4f2c632d6d59fb82663fb989ad90e2e797bbc7841c1c4d0729e3ddc9dac`.
- Three isolated SDK mutations were rejected by named behavioral assertions:
  raw namespace, partial-call dispatch, and completion without a stream terminal.
  The unchanged build passed and the working source hash stayed unchanged.
- `cargo deny --offline --locked check licenses bans sources`: passed with
  duplicate-version and unmatched-allowance warnings retained. This is not a
  complete artifact/SBOM/snippet audit or legal clearance. New dependency licence
  texts are in the existing packaged NOTICE; canonical policy is unchanged.
- Final SDK tarball SHA-256:
  `bc13339ff121245f6c04cff83a9e7e304826c75103e1f0c5ca87a1f5a4208a07`.
  An offline clean npm consumer passed 49 installed HTTP/Patch tests and public
  declarations with source/dist hashes unchanged. The standard package smoke
  passed; ownership remains `pre-P1`, not its P2 target.
- A synchronized CLI/SDK candidate passed the complete installed engineering
  smoke. This does not promote it to a released or globally installed product.
- Two independent read-only reviews and a hostile pass produced no new concrete
  findings. The lead added actual recursive-language and bounded equivalence
  tests rather than accepting a review claim citing unrelated test coverage.

The retained report is under the qualification run's `real-model-grammar-mixed2`
directory. Failed attempts are retained: unsupported Q4 tensor format, the
pre-compaction matcher work bound, an auto-choice harness expecting a required
tool, and incomplete AgentLoop fixture configuration. None is reported as a
passing check.

One later full SDK run failed F9's numeric-PID exit check despite a retained
termination receipt and an isolated pass. The harness now compares PID plus
creation time on Windows/POSIX and never signals a reused PID. The exact cause
of that historical timeout is not proven; the failure remains in
`sdk-full-grammar-wire-final.log`. The corrected F9/C2/identity tests and full
suite passed. No production process-cleanup change was inferred from the timeout.

Mixed raw-string boundaries are recovered from llguidance captures, not text
delimiter searches. Repeated captures preserve every call, including payloads
containing envelope-like text. Batch output is bounded and non-executable until
the complete constrained document is received. Explicit call-count controls
keep `required` (at least one) distinct from an exact-cardinality request.

## Remaining Gates

Supported backend performance/accelerator and cross-platform checks;
documentation and generated-catalogue checks;
exact final artifacts with complete provenance/snippet/SBOM/notices audits;
review/merge and runtime/SDK/CLI release sequencing; registry and installed proof.
No release, deployment, trust-key addition or compliance sign-off occurred.

## Windows CUDA Qualification (2026-10-06)

Isolated locked `cargo build -p xrt-server --release --features cuda` passed in
`E:/xeno-work/qualification-20261006/runtime-cuda-target`. The server's actual
status reported requested AND active `cuda-resident`, CUDA available, NVIDIA
RTX 4090 and resident model allocations. No CPU fallback or global toolkit
installation was used. Server SHA-256:
`9ecb78d29d2f6ca6e6b968df6232dda13253c9baf475a7ed29d79dfe0ec8cecc`.
Model SHA remains
`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.

All 12 actual-model checks passed on CUDA: real default catalogue, durable
quota admission/usage settlement, whole UTF-8 GBNF, generated Patch, nested
function-schema semantics, mixed/repeated calls and JSON/SSE IDs, budgeted
custom negotiation, strict exact-file edits, truncation refusal and actual
AgentLoop canonical-name permission execution. Report:
`E:/xeno-work/qualification-20261006/real-model-grammar-cuda/report.json`.
The owned server exited. Qualifier supports explicit `--backend cpu` or
`--backend cuda-resident` and asserts actual runtime status before checks.

This closes the Windows RTX 4090 grammar correctness subset, not a representative
performance baseline, other accelerators/platforms, exact release provenance or
the broader live provider/workflow matrix. 11.6/14.3 remain partial and unreleased.

## Full CLI Prompt Measurement

Exact captured CLI prompt of 3,079 tokens completed fixed diagnostic SSE text
in 139.332 seconds on the CUDA backend, with one admitted prefill and no queue.
Raw-SSE string assertion originally failed despite a complete stop/DONE; corrected
retained-byte parser readback passed. Report plus original failure and corrected
readback: `E:/xeno-work/qualification-20261006/full-prompt-cuda-3cgNtw/`.
Dense CUDA forward_batch currently iterates tokens sequentially. This is a real
performance gap for agent prompts, not a deadlock or permission to trim context.

Separately, actual built CLI named-child context probe passed three native CUDA
calls under declared measured 180-second per-call / 540-second total bounds:
`E:/xeno-work/qualification-20261006/real-model-child-context-hmlHj6/report.json`.
Full parent/child prompts and intended shared memory were preserved; private
transcripts excluded. Forced legal tool/text outputs mean transport/context proof,
not autonomous model quality or broad workflow performance. All owned model
processes exited; source changes remain unreleased and uncommitted.
