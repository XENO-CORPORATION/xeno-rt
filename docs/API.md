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
| `presence_penalty` | number | `0` | OpenAI-compatible additive penalty for tokens already present |
| `frequency_penalty` | number | `0` | OpenAI-compatible additive penalty per prior occurrence |
| `seed` | integer | random | Deterministic seed when supplied |
| `stream` | boolean | `false` | SSE when true |
| `cache_policy` | string | `default_chat` | Local extension |
| `recent_window_tokens` | integer | policy default | Local extension |

Non-streaming responses include `id`, `object`, `created`, `model`, `choices`,
and token `usage`. Streaming responses use Server-Sent Events with completion
chunks and terminate with the standard `[DONE]` marker.

## `POST /v1/chat/completions`

Accepts the generation fields above plus:

| Field | Type | Notes |
|---|---|---|
| `messages` | array | Required role/content messages |
| `tools` | array | Tool definitions accepted by the chat-template path |
| `tool_choice` | value | Accepted tool-choice strategy |
| `enable_thinking` | boolean | Optional Qwen chat-template control; omitted preserves the model template default |
| `chat_template_kwargs` | object | Optional vLLM-compatible envelope; currently supports `enable_thinking` |

When both forms are present, top-level `enable_thinking` wins. Setting it to
`false` requests the model's non-thinking chat template; it does not guarantee
that every model or prompt will have the same quality as thinking-enabled
generation. Models whose template does not consume the variable continue to
use their normal template behavior.

For Qwen3.8 chat models, omitted sampling fields use the model-native profile:
thinking uses temperature `1`, top-k `20`, top-p `0.95`, repetition penalty
`1`, and zero presence/frequency penalties; non-thinking uses temperature
`0.7`, top-k `20`, top-p `0.8`, repetition penalty `1`, presence penalty
`1.5`, and zero frequency penalty. Any field supplied by the caller overrides
only that field. Other architectures and earlier Qwen versions retain XRT's
existing defaults.

Thinking Qwen responses keep the existing OpenAI-compatible `content` field as
the final answer and add `reasoning_content` to the response message. Streaming
chunks likewise emit either `delta.reasoning_content` or `delta.content` and
never expose the model's internal `</think>` boundary in `content`. The
additional field is omitted for non-thinking responses, so clients that only
consume `content` remain compatible.

Message `content` may be text or an array of content parts. Tool-call fields on
assistant/tool messages are preserved for template construction. Image content
requires a loaded mmproj file and a compatible model/template path.

Tool fields are a compatibility surface, not a guarantee that every model will
produce a valid structured tool call. Model template and output quality remain
model-dependent.

Both completion routes return `finish_reason: "length"` when generation reaches
`max_tokens` and `finish_reason: "stop"` when generation stops earlier. Chat
responses may instead return `finish_reason: "tool_calls"` after a structured
tool call is extracted. The same terminal reason is emitted in the final SSE
chunk before `[DONE]`.

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
