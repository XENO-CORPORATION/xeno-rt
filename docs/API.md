# HTTP API

`xrt-server` exposes an OpenAI-compatible subset plus xeno-rt lifecycle and
image-task routes. Compatibility means that the documented request and
response shapes can be used by OpenAI-style clients; it does not mean that
every OpenAI endpoint or field is implemented.

## Start the Server

```bash
cargo run --release --locked -p xrt-server -- \
  --model ./models/model.gguf \
  --backend cpu \
  --host 127.0.0.1 \
  --port 3000
```

The model is optional at process start. A runtime can be loaded later through
`POST /v1/runtime/load`.

## Security Boundary

The server has no built-in inbound API-key validation or TLS termination. It
binds to `127.0.0.1` by default. For non-loopback use, put it behind an
authenticating TLS reverse proxy, restrict network access, and set request/body
limits at that boundary.

The `XRT_EXTERNAL_API_KEY` setting authenticates outbound proxy requests; it
does not protect inbound xrt-server requests.

## `GET /v1/models`

Returns an OpenAI-style list object. When no runtime is loaded, `data` is
empty.

## `POST /v1/completions`

Request fields:

| Field | Type | Default | Notes |
|---|---|---:|---|
| `model` | string | loaded model | Optional routing label |
| `prompt` | string | required | Completion prompt |
| `max_tokens` | integer | `128` | Maximum generated tokens |
| `temperature` | number | `0.8` | `0` selects greedy behavior |
| `top_k` | integer | `40` | Local extension |
| `top_p` | number | `0.95` | Nucleus threshold |
| `repetition_penalty` | number | `1.1` | Local extension |
| `seed` | integer | random | Deterministic seed when supplied |
| `stream` | boolean | `false` | SSE when true |
| `cache_policy` | string | `default_chat` | Local extension |
| `recent_window_tokens` | integer | policy default | Local extension |
| `grammar` | string | absent | Local GBNF enforced during sampling; malformed input returns 400 |

Non-streaming responses include `id`, `object`, `created`, `model`, `choices`,
and token `usage`. Streaming responses use Server-Sent Events with completion
chunks and terminate with the standard `[DONE]` marker.

## `POST /v1/chat/completions`

Accepts the generation fields above plus:

| Field | Type | Notes |
|---|---|---|
| `messages` | array | Required role/content messages |
| `tools` | array | Function object schemas or custom GBNF string tools; at most 128 |
| `tool_choice` | value | `auto` (default), `none`, `required`, or a named function/custom declaration |
| `parallel_tool_calls` | boolean | `true` by default; `false` constrains a function response to one call |

The additive `xeno.min_tool_calls` / `xeno.max_tool_calls` controls constrain
batch cardinality during sampling, with `1 <= min <= max <= 128`. Supplying a
minimum requires a tool response even under `auto`. Count limits require
declared tools, cannot be combined with `tool_choice: "none"`, and cannot exceed
one when `parallel_tool_calls` is false. Whole-text completions refuse them.
Exact chat prompt preflight applies the same constraints and validation.

Message `content` may be text or an array of content parts. Tool-call fields on
assistant/tool messages are preserved for template construction. Image content
requires a loaded mmproj file and a compatible model/template path.

Local tool selection and arguments share one sampling-time constraint. Function
arguments retain their JSON Schema; custom tools use GBNF. Model relevance and
the semantic correctness of tool input remain model-dependent. Consumers must
validate input, apply permission policy and confirm effects before reporting
success. A grammar is not permission to execute a tool.
Function batches retain ordered, distinct call IDs and SSE indices, bounded to
128 calls. Function and mixed custom/function batches are emitted after their
entire constrained document completes. Parser captures, not payload delimiter
searches, recover mixed and repeated custom calls. Single custom input streams
incrementally. Streamed batches and single calls are not executable previews.

### Constrained tools (source candidate, 2026-10-05)

```json
{
  "messages": [{ "role": "user", "content": "Use the example tool." }],
  "tools": [{
    "type": "custom",
    "custom": {
      "name": "Example",
      "description": "Return the required example text.",
      "format": { "type": "grammar", "syntax": "gbnf", "definition": "root ::= \"hello\"" }
    }
  }],
  "tool_choice": { "type": "custom", "custom": { "name": "Example" } },
  "max_tokens": 64
}
```

The result contains `tool_calls: [{ id, type: "custom", custom: { name, input },
xeno_grammar: { complete: true } }]` and `finish_reason: "tool_calls"`.
Custom SSE starts with `{ index, id, type: "custom", custom: { name, input } }`,
then carries `custom.input` fragments. Concatenated fragments reproduce the raw
input without stripping whitespace, tags or UTF-8 characters.

Truncation returns `finish_reason: "length"`. A partial JSON response contains
no executable tool call; an SSE preview carries `xeno_grammar.complete: false`
when a tool has been identified. Clients must wait for an intact stream terminal
and completion status before executing any call. Inference failure sends an SSE
error without prompt or argument payloads. Disconnects stop generation when the
next emitted piece encounters the closed stream.

Only `gbnf` is supported. Individual definitions and function schemas have a
64-KiB bound; compiled rule/element and matcher-work limits also apply. Invalid,
unsupported and excessive constraints are refused, never silently discarded.
Top-level `grammar` constrains a whole response and cannot be combined with
`tools`; use a custom tool when the model must select among tools.

`GET /v1/runtime/capabilities` reports `custom_tool_grammars: ["gbnf"]` and
`custom_tool_grammars_streaming: true` in local mode, and `[]`/`false` in external
proxy mode. It also reports prompt-token count/ceiling support. These flags
describe the local implementation, not whether a model is loaded; use runtime
status for readiness. Older servers lacking the fields are unsupported.
The external proxy explicitly refuses GBNF/custom requests it cannot enforce.
Constrained requests disable speculative decoding; unconstrained requests keep
their existing decoding path.

## Runtime Lifecycle

### `GET /v1/runtime/status`

Reports readiness, loaded model paths, requested/active backend, KV mode,
prefix-cache state, scheduler state, external backend state, and GPU resource
telemetry.

### `POST /v1/runtime/load`

Accepted fields:

```json
{
  "model_path": "./models/model.gguf",
  "mmproj_path": null,
  "backend": "auto",
  "hf_repo": null,
  "hf_file": null,
  "external_base_url": null,
  "external_api_key": null,
  "external_model": null
}
```

Use either `model_path`, the `hf_repo`/`hf_file` pair, or an
`external-openai` configuration. Loading replaces the active runtime only
through the lifecycle handler's validated path.

### `POST /v1/runtime/unload`

Releases the active runtime and returns `{ "success": true }`.

## `POST /v1/images/remove-background`

Experimental ONNX endpoint. Provide exactly one of `image_b64` or `image_url`.
Optional fields are `model_path` and `use_gpu` (default `true`). The response
contains base64-encoded PNG bytes plus output width and height.

`image_url` can cause server-side reads/fetches. Do not expose this endpoint to
untrusted callers without an upstream policy that restricts allowed schemes,
hosts, paths, payload sizes, and response sizes.

## Errors

Invalid requests return an HTTP error with a text explanation. Backend/model
errors are explicit. Clients should not parse error text as a stable machine
contract; status codes and successful JSON shapes are the compatibility
surface.

## External OpenAI-Compatible Backend

Set `--backend external-openai` with `XRT_EXTERNAL_BASE_URL` or the equivalent
load-request field. Targets are limited to loopback by default. Set
`XRT_EXTERNAL_ALLOW_REMOTE=1` only when the remote host, TLS, and credentials
are trusted.
