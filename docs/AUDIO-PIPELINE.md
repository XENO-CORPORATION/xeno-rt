# Audio pipeline integration

Scope: **xrt-audio**. Status: **development integration; not a published release**.

This guide describes the local Chatterbox Multilingual v3 speech API. It does not
claim that music generation, speech-to-speech conversion, streaming speech, or a
public transcription API is implemented.

## Contract at a glance

| Operation | Endpoint |
|---|---|
| Check file presence and worker occupancy (not a load guarantee) | `GET /v1/audio/status` |
| Register a voice reference once | `POST /v1/audio/voices` |
| List registered voices | `GET /v1/audio/voices` |
| Read a voice's identity and checksum | `GET /v1/audio/voices/{id}` |
| Delete a registered reference | `DELETE /v1/audio/voices/{id}` |
| Generate narration using a saved voice or inline reference | `POST /v1/audio/speech` |

Registration is zero-shot voice cloning: it saves a reference recording. It does
**not** train a new model. Use recordings you have permission to use.

All inference runs locally in Rust through ONNX Runtime. Python is only an
optional provisioning/test/client tool, not part of the inference process.

## Installation (Windows x64)

The routes described here arrive in **0.4.0**; a 0.3.x `xrt-server.exe` does not
contain them. Until the 0.4.0 package is published, build from source
(`cargo build -p xrt-server -p xrt-cli --release --features cuda,transcription`;
omit `cuda` for a CPU build) and place the ONNX Runtime DLLs beside the
executables. `scripts/stage-audio-native.py` assembles and hash-verifies that
layout from the pinned runtime manifests in `reference/runtime/`.

**Native libraries.** The server loads `onnxruntime.dll` from beside
`xrt-server.exe`, or from `ORT_DYLIB_PATH` if set. It reads the library's own
version and refuses anything older than **1.23** (Chatterbox v3's attention
operator needs it) with an actionable error. CUDA additionally needs the ONNX
provider DLLs and the NVIDIA libraries in
`reference/runtime/cuda12-payload-xrt-audio-windows-x64.json` (about 2.3 GB, each
file pinned to its vendor wheel). A GPU driver alone is not enough. CUDA requests
fail rather than silently falling back when `device` is `cuda` or `cuda:N`; `auto`
may fall back, so inspect the reported provider.

**Models.** Install the two pinned bundles with the runtime's own installer. It
plans by default, downloads with resume, verifies every file's size and SHA-256,
and activates a bundle atomically only after all of its files verify:

```powershell
$hosts = '--allowed-host','huggingface.co','--allowed-host','us.aws.cdn.hf.co'
$cb = 'reference\audio\bundles\chatterbox-multilingual-v3.json'
$cbDigest = '862678bb134fc58679f1e9b24399b84b96edfa053adf7730d307d57b28e99dbd'
$wh = 'reference\audio\bundles\whisper-small-timestamped.json'
$whDigest = '9c287bbd6898cd6a766425c362212052fea76b6074ceeb299c71c790839a0aa7'

.\xrt-cli.exe bundle install --manifest $cb --digest $cbDigest @hosts            # plan only
.\xrt-cli.exe bundle install --manifest $cb --digest $cbDigest @hosts --confirm
.\xrt-cli.exe bundle install --manifest $wh --digest $whDigest @hosts --confirm
.\xrt-cli.exe bundle verify chatterbox-multilingual-v3                            # offline rehash
```

The digest is the identity of the whole bundle (every path, size and hash), so a
changed or partial download cannot be activated. Interrupted downloads resume;
an installed bundle is never overwritten. `bundle remove <id> --digest <d>
--confirm` deletes only the verified declared files and refuses a corrupted or
linked tree. The model cache defaults to `~/.cache/xrt/models` (`XRT_CACHE_DIR`
overrides it). With installed bundles the server needs **no model-path
variables**:

```powershell
$env:XRT_AUDIO_VOICES_DIR = 'D:\xrt\state'   # optional; defaults to ~/.xeno
.\xrt-server.exe --host 127.0.0.1 --port 3338
```

`XRT_AUDIO_MODEL_DIR` / `XRT_AUDIO_ASR_DIR` remain as developer overrides for a
custom directory; such directories are not integrity-managed.
`scripts/provision-audio-models.py` is a development utility, not the production
installer.

`XRT_AUDIO_VOICES_DIR` is the **store root**, not the final voices directory.
The layout is `<root>/voices/v1/<id>/{reference.wav,voice.json}`. Default root:
`~/.xeno`. Back up that directory; deleting a voice removes its stored reference.
Models are separate from voice state.

CUDA additionally needs the ONNX provider DLLs and the NVIDIA libraries listed in
`reference/runtime/cuda12-payload-xrt-audio-windows-x64.json`. The current payload
is about 2.3 GB. A driver alone is not enough. Do not distribute a random selection
of DLLs or assume a successful CPU fallback proves CUDA installation.

Keep this server on loopback. The voice library is server-wide local state, not a
multi-user authorized service. Do not expose it directly to the internet.

## Working client example

Python 3, standard library only. Save as `narrate.py`, then run
`python narrate.py reference.wav script.txt narration.wav`. The reference should
be clean, single-speaker WAV, at least 3 seconds; 6–10 seconds is recommended.
Longer references use only the first 10 seconds. MP3/MP4 decoding belongs in your
media pipeline before this call.

```python
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

BASE = 'http://127.0.0.1:3338'
VOICE = 'documentary-narrator'


def request(method, path, body=None):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={} if data is None else {'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def save_new(path, data):
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


reference, script, output = map(Path, sys.argv[1:4])
try:
    info = request('GET', '/v1/audio/voices/' + VOICE)
except urllib.error.HTTPError as error:
    if error.code != 404:
        raise
    info = request('POST', '/v1/audio/voices', {
        'id': VOICE,
        'name': 'Documentary narrator',
        'audio_b64': base64.b64encode(reference.read_bytes()).decode('ascii'),
    })

# Existing VOICE deliberately stays unchanged. Select a new id to use a new
# recording; do not silently delete/re-register a voice used by other jobs.
print('Voice:', info['id'], 'reference SHA-256:', info['sha256'])
result = request('POST', '/v1/audio/speech', {
    'model': 'chatterbox-multilingual-v3',
    'input': script.read_text(encoding='utf-8-sig'),
    'voice': VOICE,
    'language': 'en',
    'device': 'cuda',
    'preset': 'dramatic',
    'seed': 42,
    'word_check': True,
    'best_effort': False,
    'breath_reduction_db': 15,
    'response_format': 'json',
})
if any(chunk['best_effort'] for chunk in result['chunks']):
    raise RuntimeError('Unapproved best-effort take')
for chunk in result['chunks']:
    if chunk['word_problems']:
        print('Review transcript differences:', chunk['word_problems'])

save_new(output, base64.b64decode(result.pop('audio_b64'), validate=True))
save_new(output.with_suffix('.json'), json.dumps(result, indent=2).encode('utf-8'))
print('Saved', output, result['duration'], 'seconds; provider', result['provider'])
```

For long narration, or any pipeline that must survive a crash or a network
drop, use the maintained job client `examples/audio/narrate_job.py` (same
arguments). It checks readiness, registers or reuses the voice, submits a durable
job with an Idempotency-Key derived from the inputs, prints chunk progress, saves
the WAV plus timings, and deletes the job record. Re-running it after an
interruption waits on the same job rather than starting a second render.

For a one-off voice omit `voice` and send `voice_b64` containing a base64 WAV in
the speech request. Exactly one voice source must be present.

## Speech options

| Field | Default | Meaning |
|---|---|---|
| `input` | required | Script, at most 50,000 Unicode characters. Internally token-chunked. |
| `model` | v3 when omitted | Exact id `chatterbox-multilingual-v3`. Other ids are refused. |
| `voice` | none | Saved voice id. Alternative: `voice_b64` or `voice_url`, not both. |
| `language` | `en` | Model language code. English is the retained end-to-end quality evidence. |
| `preset` | `neutral` | `neutral`, `documentary`, `dramatic`; explicit numerical fields override. |
| `device` | `auto` | `cpu`, `cuda`, `cuda:N`, or `auto`. |
| `exaggeration` | 0.5 | Model conditioning intensity, 0–2. Not a word-level emphasis guarantee. |
| `cfg_weight` | 0.5 | Guidance, 0–3. Lower is not reliably better/slower; it can cause rambling. |
| `temperature` | 0.8 | Sampling temperature, 0–2. |
| `seed` | 42 | Base sampling seed; retries use different seeds. Not a cross-device identity guarantee. |
| `sentence_pause` | 0.28 s | Desired minimum sentence pause where a protected gap exists. |
| `paragraph_pause` | 0.63 s | Desired paragraph pause; both LF and CRLF paragraphs are recognized. |
| `pause_scale` | 1 | Scale existing clause/sentence pauses, 1–3; never stretches speech. |
| `breath_reduction_db` | 15 | Attenuation in estimated non-word gaps, 0–40; zero disables. |
| `word_check` | true | Whisper transcript comparison and retry. Disabling also disables default breath processing. |
| `names` | empty | Proper-noun hints for recognition comparison; inspect reported differences. |
| `best_effort` | false | Explicit opt-in to the closest failed word-check take after attempts exhaust. |
| `speed` | 1 | Other values are rejected; no time-stretching. |
| `response_format` | `wav` | `wav`, `pcm`, `json` (`verbose_json` alias). |

Presets:

| Preset | Exaggeration | CFG | Sentence / paragraph pause |
|---|---:|---:|---:|
| neutral | 0.5 | 0.5 | 0.28 / 0.63 s |
| documentary | 0.6 | 0.5 | 0.50 / 0.90 s |
| dramatic | 0.7 | 0.45 | 0.55 / 1.00 s |

A calm reference plus the dramatic preset was the selected listening setup.
This does not mean every script or speaker will produce the same quality.

## Optional narration director

`"direction": "auto"` calls the server's already-loaded **local text Runtime**
through its existing scheduler. Start the server with `--model <supported.gguf>`
to use this. No remote provider is selected silently; without a local model the
request returns 428. Invalid model-generated plans return 422 before speech.

The director returns **numeric controls**, never replacement script text:
per-generation-chunk exaggeration, sentence/paragraph pause targets, and optional
punctuation-boundary pauses indexed by original whitespace-word counts. It does
not provide pitch control or arbitrary per-word emotional acting. Short sentences
remain packed into stable chunks rather than being generated individually.

An explicit plan avoids the LLM call, for example for a script that packs into
one chunk (the plan must contain every actual chunk, in order):

```json
{
  "direction": {
    "chunks": [
      {"index": 0, "exaggeration": 0.6, "sentence_pause": 0.5, "paragraph_pause": 0.9}
    ],
    "pauses": [{"after_word": 9, "seconds": 1.0}]
  }
}
```

`after_word` is 1-based, unique, increasing, must follow punctuation, and must
not be the final word. Controls are bounded: exaggeration 0.3–0.9, sentence pause
0–2 s, paragraph/explicit pause 0–3 s. A requested pause is inserted only when
alignment provides a protected gap; it never licenses cutting into a word.
The JSON speech result returns `direction` so you can retain the plan alongside
the take. Automatic direction requires a model that can follow this JSON
contract; the path must be verified with your chosen text model. On 2026-09-28,
Qwen2.5-0.5B-Instruct Q8_0 on CPU produced a validated plan and a successful
14.67-second CUDA speech response from the clean-main audio candidate. This is a
single-path proof, not evidence that the small model directs every script well.
The 4B Qwen3.8 director also worked on the development text branch but its loader
is not yet on release `main`; do not depend on that model in a main-based install.

## Responses and captions

- `wav`: 24 kHz mono IEEE float32 RIFF/WAVE, `Content-Type: audio/wav`.
- `pcm`: 24 kHz mono **signed 16-bit little-endian**, no header, `audio/pcm`.
- `json`: `audio_b64` contains a WAV; also returns `sample_rate`, `duration`,
  `provider`, `asr_provider`, `words`, and `chunks`.

Word entries have `text`, `start`, `end` (seconds in final audio), and `heard`.
Text follows the **input script**, not an independent literal transcript.
Changed numbers and negations are hard rejections; supported English integer
spellings are compared by value. Other isolated differences can be tolerated but
are reported. If differences remain, internal breath/pause shaping is skipped
rather than trusting that alignment to cut audio.
A spelled number can occupy one timing entry. `heard: false` means an estimated
interpolation; do not use it as an exact edit point. Times are ASR estimates,
not sample-exact boundaries or proof of pronunciation.

Chunk reports include retry reasons, transcript differences, normalized word
error rate, `best_effort`, gaps softened, and seconds of added silence.
`breaths_softened` counts processed gaps, **not independently detected breaths**.

Binary and JSON responses carry `X-Xrt-Sample-Rate`, `X-Xrt-Duration-Seconds`,
`X-Xrt-Provider`, `X-Xrt-Chunks`, and `X-Xrt-Retries`.
`X-Xrt-Word-Check` is `matched`, `passed-with-differences`, or `best-effort` when
recognition ran; absent when disabled. `matched` is not a guarantee: two speech
models can make the same mistake.

## References, limits, and failure handling

Prefer inline base64 references. The request body cap is 32 MiB, including base64
and JSON overhead. WAV decoding rejects truncated, non-finite, out-of-range or
unsupported audio instead of forwarding it into inference.

`voice_url` / registration `audio_url` accepts WAV data URLs. File URLs require
operator-configured `XRT_AUDIO_REFERENCE_ROOT`, with canonical paths confined to
that directory. HTTP(S) URLs require operator-configured
`XRT_AUDIO_REFERENCE_ORIGINS` (comma-separated exact origins). Redirects are not
followed; downloads are bounded to 32 MiB and 30 seconds. Only grant trusted
origins. Never accept those grants from an API caller.

| Status | Caller action |
|---|---|
| 400 | Fix invalid input/options/audio; do not retry unchanged. |
| 404 | Register the voice or correct its id. |
| 409 | Voice id exists; deliberately choose/reuse an id, never overwrite implicitly. |
| 413 | Reduce reference/request size. |
| 422 | Local director returned an invalid plan; no speech was generated. |
| 428 | Required model files or local direction model are not installed/loaded. |
| 429 + `Retry-After` | The single speech worker is occupied. Retry with bounded backoff. |
| 500 | Inference/quality exhaustion or store failure; inspect the error and logs. |
| 503 | Native runtime is incompatible/unavailable. |

Synchronous speech cancels on client disconnect or `timeout_seconds` (default
1800, range 1–7200), using cooperative token/window checks and ONNX run termination.
The worker keeps its admission slot until native execution exits. Cancellation
returns 408 when the client remains connected; no partial audio is published.
`POST /v1/audio/unload` releases cached sessions (409 while busy), and
`POST /v1/audio/drain` cancels active work and refuses new work until restart.
`GET /v1/audio/status` distinguishes loaded providers from files merely present.

For longer pipeline tasks, use the durable local job API:

| Operation | Route |
|---|---|
| Submit a speech request with required `Idempotency-Key` header | `POST /v1/audio/jobs` |
| Poll status and persisted `completed_chunks` | `GET /v1/audio/jobs/{id}` |
| Fetch successful audio/timings JSON | `GET /v1/audio/jobs/{id}/result` |
| Request cancellation | `POST /v1/audio/jobs/{id}/cancel` |
| Delete a terminal job and retained result | `DELETE /v1/audio/jobs/{id}` |

Submission returns 202 with an opaque id. Repeating the same key and identical
request/reference/model bytes returns the same job; changed bytes return 409.
Reference clips are frozen at submission, so deleting a saved voice later does
not change an accepted job. Job state lives in SQLite under
`XRT_AUDIO_JOBS_DIR`, default `<voice-store-root>/audio-jobs/v1`. One server owns
a store at a time. Completed results survive restart; queued/running work becomes
`interrupted`, not falsely completed or silently resumed.

Limits: 8 pending jobs, 100 retained records, 2 GiB logical request/result budget,
and 128 MiB per JSON result. Capacity is reserved for pending results. Delete
terminal records deliberately when done. Only successful terminal jobs have a
result URL. There is no token-level crash resume or progress streaming; polling
reports completed chunks. The submission connection may close after acceptance
without cancelling the durable job; use its cancel route instead.

The 50,000-character admission cap is not a measured long-form performance
promise. Start with short narration paragraphs, establish your timeout and GPU
budget, then test multi-chunk scripts before automating long episodes.

## Verification

Run pure Rust tests:

```powershell
cargo test -p xrt-audio
cargo test -p xrt-server audio_api
```

Tokenizer/chunking model fixtures need `XRT_AUDIO_MODEL_DIR`; read test output for
fixture skips. Pure tests are not evidence that native ONNX or CUDA loaded.

For real HTTP + real models, start a **dedicated** test server with its own
`XRT_AUDIO_VOICES_DIR`, then:

```powershell
python scripts/test-audio-api.py --base http://127.0.0.1:3711 --reference reference.wav --output audio-evidence-001 --device cuda
```

This test creates only a unique temporary voice id and removes that id afterwards.
It retains the generated WAV, JSON timing/report, PCM output, and check results.
Missing assets fail rather than silently skipping generation.

Before release: run from the packaged artifact with a fresh state directory,
verify CPU fallback separately, test missing native libraries and corrupt model
assets, run long-form/failure/concurrency tests, and verify the published downloads
by installing them back. Development smoke tests do not substitute for these.

## Recovery in 0.4.0-rc.1

Validation errors, rejected takes, and CPU requests reusing CPU models retain healthy sessions. Native failures and cancellation clear affected sessions. Switching from CUDA to CPU releases CUDA sessions before releasing their memory reservation.

A corrupt partial download is never activated. Discard that one partial install explicitly, then retry:

```bash
xrt-cli bundle discard-partial <bundle-id> --digest <pinned-digest>
xrt-cli bundle discard-partial <bundle-id> --digest <pinned-digest> --confirm
```

The first command is a dry run. Discard does not remove installed bundles. Removal refuses undeclared files or directories before changing installed files. Transcription uploads have a 30-second deadline and two upload slots separate from native inference admission. A temporary job-store lock or filesystem failure can be retried without restarting the server. Jobs with `word_check:false` need only Chatterbox; the default word check also requires the separately installed timestamped Whisper bundle. Installed Whisper-base reloads from its verified cache without fetching the registry.
