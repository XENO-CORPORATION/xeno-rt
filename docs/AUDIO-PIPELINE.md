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

## Development setup (Windows x64)

The released distribution is not ready yet. Do not assume a downloaded older
`xrt-server.exe` contains these routes.

Build from the checkout containing `crates/xrt-audio`:

```powershell
cargo build -p xrt-server --release --features cuda
```

For a CPU build omit `--features cuda`. CUDA requests fail rather than silently
falling back when `device` is `cuda` or `cuda:N`. `auto` can fall back; inspect the
reported provider.

Required model layouts:

```text
chatterbox-v3/
  tokenizer.json
  onnx/
    speech_encoder.onnx
    embed_tokens.onnx
    language_model.onnx
    conditional_decoder_slim.onnx
    conditional_decoder_slim.onnx.data
whisper-small-timestamped/
  tokenizer.json
  generation_config.json
  onnx/
    encoder_model.onnx
    decoder_model.onnx
    decoder_with_past_model.onnx
    ... any external tensor files referenced by those graphs
```

Use the pinned ONNX exports, not arbitrary models with matching filenames.
The manifests in `reference/audio/` bind the files to immutable upstream revisions
and hashes verified against Hugging Face metadata. The stdlib installer is dry-run
by default and resumes partial downloads:

```powershell
python scripts/provision-audio-models.py --model chatterbox-multilingual-v3 --destination D:\xrt\models\chatterbox-v3
# Review the plan, then explicitly install:
python scripts/provision-audio-models.py --model chatterbox-multilingual-v3 --destination D:\xrt\models\chatterbox-v3 --install
python scripts/provision-audio-models.py --model whisper-small-timestamped --destination D:\xrt\models\whisper-small-timestamped --install
python scripts/provision-audio-models.py --model chatterbox-multilingual-v3 --destination D:\xrt\models\chatterbox-v3 --verify
```

An existing mismatched file is refused, not overwritten. A download lock is not
a stale-process detector: inspect its recorded PID before manually removing a lock.
These upstream downloads are not a claim that the models were published to XENO R2.
ONNX Runtime **1.23 or newer** is needed for Chatterbox v3's attention operator;
the measured Windows runtime is 1.23.0. The `ort` crate version is not the version
of the separately loaded native DLL.

Example session configuration (replace paths with your installed assets):

```powershell
$env:ORT_DYLIB_PATH = 'D:\xrt\onnxruntime-gpu\onnxruntime.dll'
$env:PATH = 'D:\xrt\onnxruntime-gpu;D:\xrt\cuda12;' + $env:PATH
$env:XRT_AUDIO_MODEL_DIR = 'D:\xrt\models\chatterbox-v3'
$env:XRT_AUDIO_ASR_DIR = 'D:\xrt\models\whisper-small-timestamped'
$env:XRT_AUDIO_VOICES_DIR = 'D:\xrt\state'
.\target\release\xrt-server.exe --host 127.0.0.1 --port 3338
```

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

A dropped HTTP connection does not currently cancel model execution. The worker
keeps its admission slot until it finishes; a retry receives 429 instead of
starting overlapping GPU work. This is **not** a durable job API: there is no
job id, result polling, idempotency key, progress stream, or crash-resume contract.
Keep requests bounded and persist completed output in the calling pipeline.

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
