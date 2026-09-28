//! Audio domain endpoints, served from `xrt-audio` (Chatterbox Multilingual v3).
//!
//! | route | purpose |
//! |---|---|
//! | `POST   /v1/audio/speech` | generate speech; voice = saved id, or a clip in the request |
//! | `GET    /v1/audio/voices` | list saved (cloned) voices |
//! | `POST   /v1/audio/voices` | clone a voice: register a reference clip under an id |
//! | `GET    /v1/audio/voices/{id}` | one voice's manifest |
//! | `DELETE /v1/audio/voices/{id}` | remove a saved voice |
//!
//! `POST /v1/audio/speech` details:
//!
//! OpenAI-compatible where the standard applies: `input`, `model`,
//! `response_format` (`wav` | `pcm`) and `speed` behave as in OpenAI's
//! `/v1/audio/speech`, and the response body is the audio itself. OpenAI's
//! `voice` names a built-in preset; we have none, so the voice is a reference
//! clip supplied by the caller (`voice_b64` / `voice_url`). XENO controls live
//! under additive fields and `X-Xrt-*` response headers.
//!
//! Every take is transcribed (Whisper, same device) and checked against the
//! script; a take that dropped, repeated or garbled words is regenerated.
//! The same word times make breath softening and pause lengthening
//! word-safe. `response_format: "json"` returns the audio (base64) together
//! with every script word's time in it and the per-chunk report, for
//! pipelines that cut, caption or verify.
//!
//! Inference is long (seconds per chunk on GPU, minutes on CPU) and runs in
//! `spawn_blocking`; requests are serialised because one Chatterbox instance
//! owns ~3 GB and generation is not re-entrant.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::Deserialize;
use tokio::sync::Semaphore;
use xrt_audio::{chatterbox::Device, AudioError, Preset, SpeechOptions};

use crate::AppState;

/// Upper bound on one request's script. Long-form narration is chunked
/// internally; this caps the work one HTTP request can queue.
pub(crate) const MAX_INPUT_CHARS: usize = 50_000;
/// Reference clips are trimmed to 10 s by the model; this bounds the upload.
pub(crate) const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Admission is bounded rather than queuing arbitrarily many WAV buffers.
/// The permit belongs to the blocking worker: dropping the HTTP future must
/// not release it while CUDA inference is still running.
fn speech_slot() -> &'static Arc<Semaphore> {
    static SLOT: std::sync::OnceLock<Arc<Semaphore>> = std::sync::OnceLock::new();
    SLOT.get_or_init(|| Arc::new(Semaphore::new(1)))
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpeechRequest {
    /// The text to speak.
    input: String,
    /// Accepted for OpenAI compatibility; must be absent or a Chatterbox id.
    #[serde(default)]
    model: Option<String>,
    /// Reference voice: a WAV as base64, or a `data:`/`file://`/`http(s)://` URL.
    #[serde(default)]
    voice_b64: Option<String>,
    #[serde(default)]
    voice_url: Option<String>,
    /// OpenAI field. Here it names a SAVED voice id (see `POST
    /// /v1/audio/voices`); OpenAI's built-in names are not provided.
    #[serde(default)]
    voice: Option<String>,
    /// `wav` (default, 32-bit float), `pcm` (24 kHz signed 16-bit little-endian), or
    /// `json` (`audio_b64` WAV + `words` + `chunks`).
    #[serde(default)]
    response_format: Option<String>,
    /// OpenAI field. Only 1.0 is accepted: xrt-audio never time-stretches
    /// speech (it warbles words); pace comes from the voice and the preset.
    #[serde(default)]
    speed: Option<f32>,
    /// `neutral` | `documentary` | `dramatic`.
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    exaggeration: Option<f32>,
    #[serde(default)]
    cfg_weight: Option<f32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Multiply the model's own clause/sentence pauses (1..=3, default 1).
    #[serde(default)]
    pause_scale: Option<f32>,
    /// dB to lower model-generated breaths by (0 disables; default 15).
    #[serde(default)]
    breath_reduction_db: Option<f32>,
    /// Transcribe and verify every take (default true). Off also disables
    /// breath and pause shaping, which need the word times.
    #[serde(default)]
    word_check: Option<bool>,
    /// Proper nouns in the script, matched more loosely by the word check.
    #[serde(default)]
    names: Option<Vec<String>>,
    /// Deliver the closest take when every attempt fails the word check,
    /// with its problems reported, instead of failing (default false).
    #[serde(default)]
    best_effort: Option<bool>,
    #[serde(default)]
    sentence_pause: Option<f32>,
    #[serde(default)]
    paragraph_pause: Option<f32>,
    /// `auto` (default), `cpu`, or `cuda` / `cuda:N` (fails rather than falls back).
    #[serde(default)]
    device: Option<String>,
    /// "auto" uses the loaded local text runtime; object supplies controls.
    #[serde(default)]
    direction: Option<serde_json::Value>,
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": { "message": msg.into(), "type": "xrt_audio_error" } })),
    )
        .into_response()
}

pub(crate) async fn audio_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let permit = match speech_slot().clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            let mut response = err(
                StatusCode::TOO_MANY_REQUESTS,
                "speech worker is busy; retry after the active request finishes",
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            return response;
        }
    };
    let auto_direction = req.direction.as_ref().and_then(|v| v.as_str()) == Some("auto");
    // Reference loading (including URL I/O) must not block a Tokio worker.
    let built = tokio::task::spawn_blocking(move || build(req)).await;
    let (mut opts, script, reference, rate, format) = match built {
        Ok(Ok(value)) => value,
        Ok(Err(response)) => return response,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reference worker failed: {e}"),
            )
        }
    };
    if auto_direction {
        match crate::audio_direction::direct(&state, &script, &opts).await {
            Ok(plan) => opts.direction = Some(plan),
            Err((status, message)) => return err(status, message),
        }
    }
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        xrt_audio::synthesize(&script, &reference, rate, &opts)
            .map(|out| (out, format))
            .map_err(audio_error)
    })
    .await;
    match result {
        Ok(Ok((out, format))) => speech_response(out, format),
        Ok(Err(response)) => response,
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("speech worker failed: {e}"),
        ),
    }
}

type Built = (SpeechOptions, String, Vec<f32>, u32, Format);

#[derive(Clone, Copy)]
enum Format {
    Wav,
    Pcm,
    Json,
}

fn build(req: SpeechRequest) -> Result<Built, Response> {
    let bad = |m: String| err(StatusCode::BAD_REQUEST, m);
    if req.input.trim().is_empty() {
        return Err(bad("`input` is empty".into()));
    }
    if req.input.chars().count() > MAX_INPUT_CHARS {
        return Err(bad(format!(
            "`input` exceeds {MAX_INPUT_CHARS} characters; split it across requests"
        )));
    }
    if let Some(m) = req.model.as_deref() {
        if m != "chatterbox-multilingual-v3" {
            return Err(bad(format!(
                "model `{m}` is not available; this endpoint serves chatterbox-multilingual-v3"
            )));
        }
    }
    let sources = [
        req.voice.is_some(),
        req.voice_b64.is_some(),
        req.voice_url.is_some(),
    ];
    if sources.iter().filter(|s| **s).count() > 1 {
        return Err(bad(
            "give exactly one of `voice` (saved id), `voice_b64` or `voice_url`".into(),
        ));
    }
    let (reference, rate) = if let Some(id) = req.voice.as_deref() {
        library().load(id).map_err(|e| match e {
            AudioError::InvalidRequest(m) if m.contains("not found") => err(StatusCode::NOT_FOUND, format!(
                "{m}. Built-in voices are not provided: clone one with POST /v1/audio/voices, or send `voice_b64`"
            )),
            other => audio_error(other),
        })?
    } else {
        let wav = if let Some(b64) = req.voice_b64.as_deref() {
            BASE64_STANDARD
                .decode(b64.trim())
                .map_err(|e| bad(format!("invalid `voice_b64`: {e}")))?
        } else if let Some(url) = req.voice_url.as_deref() {
            load_reference(url)?
        } else {
            return Err(bad(
                "a voice is required: `voice` (saved id), `voice_b64` or `voice_url`".into(),
            ));
        };
        xrt_audio::audio::read_wav(&wav).map_err(|e| bad(e.to_string()))?
    };

    let format = match req.response_format.as_deref().unwrap_or("wav") {
        "wav" => Format::Wav,
        "pcm" => Format::Pcm,
        "json" | "verbose_json" => Format::Json,
        other => {
            return Err(bad(format!(
                "response_format `{other}` is not supported (wav, pcm, json)"
            )))
        }
    };

    let mut opts = SpeechOptions::default();
    if let Some(p) = req.preset.as_deref() {
        Preset::parse(p)
            .ok_or_else(|| {
                bad(format!(
                    "unknown preset `{p}` (neutral, documentary, dramatic)"
                ))
            })?
            .apply(&mut opts);
    }
    if let Some(v) = req.language {
        opts.language = v;
    }
    if let Some(v) = req.exaggeration {
        opts.exaggeration = v;
    }
    if let Some(v) = req.cfg_weight {
        opts.cfg_weight = v;
    }
    if let Some(v) = req.temperature {
        opts.temperature = v;
    }
    if let Some(v) = req.seed {
        opts.seed = v;
    }
    if let Some(v) = req.speed {
        if (v - 1.0).abs() > 1e-6 {
            return Err(bad(format!(
                "speed {v} is not supported: xrt-audio does not time-stretch speech. \
                 Control pace with `preset`, `pause_scale` and a calmer reference voice"
            )));
        }
    }
    if let Some(v) = req.pause_scale {
        opts.pause_scale = v;
    }
    if let Some(v) = req.breath_reduction_db {
        opts.breath_reduction_db = v;
    }
    if let Some(v) = req.sentence_pause {
        opts.sentence_pause = v;
    }
    if let Some(v) = req.paragraph_pause {
        opts.paragraph_pause = v;
    }
    if let Some(v) = req.word_check {
        opts.word_check = v;
        if !v && req.breath_reduction_db.is_none() {
            // The breath default needs word times; switching the check off
            // switches it off too rather than failing the request.
            opts.breath_reduction_db = 0.0;
        }
    }
    if let Some(v) = req.names {
        opts.names = v;
    }
    if let Some(v) = req.best_effort {
        opts.accept_best_effort = v;
    }
    opts.device = match req.device.as_deref().unwrap_or("auto") {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        d => match d.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
            Some(n) => Device::Cuda(n),
            None => {
                return Err(bad(format!(
                    "device `{d}` is not one of auto, cpu, cuda, cuda:N"
                )))
            }
        },
    };
    if let Some(value) = req.direction {
        if value.as_str() != Some("auto") {
            opts.direction = Some(
                serde_json::from_value(value)
                    .map_err(|e| bad(format!("invalid direction plan: {e}")))?,
            );
        }
        if !opts.word_check {
            return Err(bad("direction requires word_check".into()));
        }
    }
    xrt_audio::speech::validate_options(&opts).map_err(audio_error)?;
    Ok((opts, req.input, reference, rate, format))
}

fn speech_response(out: xrt_audio::SpeechOutput, format: Format) -> Response {
    let (body, ctype) = match format {
        Format::Wav => (
            xrt_audio::audio::write_wav(&out.samples, out.sample_rate),
            "audio/wav",
        ),
        Format::Pcm => (pcm16(&out.samples), "audio/pcm"),
        Format::Json => {
            let wav = xrt_audio::audio::write_wav(&out.samples, out.sample_rate);
            let body = serde_json::json!({
                "object": "audio.speech",
                "model": "chatterbox-multilingual-v3",
                "sample_rate": out.sample_rate,
                "duration": out.seconds,
                "provider": out.provider,
                "asr_provider": out.asr_provider,
                "audio_b64": BASE64_STANDARD.encode(wav),
                "words": out.words,
                "chunks": out.chunks,
                "direction": out.direction,
            });
            match serde_json::to_vec(&body) {
                Ok(bytes) => (bytes, "application/json"),
                Err(e) => {
                    return err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("cannot serialize speech result: {e}"),
                    )
                }
            }
        }
    };
    let retries: u32 = out
        .chunks
        .iter()
        .map(|c| c.attempts.saturating_sub(1))
        .sum();
    let mut resp = Response::new(Body::from(body));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(ctype));
    let mut put = |k: &'static str, v: String| {
        if let Ok(v) = HeaderValue::from_str(&v) {
            h.insert(k, v);
        }
    };
    put("x-xrt-sample-rate", out.sample_rate.to_string());
    put("x-xrt-duration-seconds", format!("{:.3}", out.seconds));
    put("x-xrt-provider", out.provider.clone());
    put("x-xrt-chunks", out.chunks.len().to_string());
    put("x-xrt-retries", retries.to_string());
    if out.asr_provider.is_some() {
        let verbatim = out.chunks.iter().all(|c| c.word_problems.is_empty());
        let best_effort = out.chunks.iter().filter(|c| c.best_effort).count();
        // Recognition is evidence, not a guarantee of the words spoken.
        put(
            "x-xrt-word-check",
            if best_effort > 0 {
                "best-effort"
            } else if verbatim {
                "matched"
            } else {
                "passed-with-differences"
            }
            .to_string(),
        );
    }
    resp
}

/// References are untrusted request data, not authority to read the host or
/// contact arbitrary services. Network origins and a local directory must be
/// explicitly granted by the operator. Inline data needs neither grant.
fn load_reference(reference: &str) -> Result<Vec<u8>, Response> {
    use std::io::Read;
    let bad = |m| err(StatusCode::BAD_REQUEST, m);
    if let Some(data) = reference.strip_prefix("data:") {
        let (mime, payload) = data
            .split_once(',')
            .ok_or_else(|| bad("malformed audio data URL"))?;
        if !matches!(
            mime,
            "audio/wav;base64" | "audio/x-wav;base64" | "audio/wave;base64"
        ) {
            return Err(bad("reference data URL must be base64 WAV"));
        }
        let bytes = BASE64_STANDARD
            .decode(payload)
            .map_err(|_| bad("invalid reference base64"))?;
        return bounded_reference(bytes);
    }
    let url = url::Url::parse(reference)
        .map_err(|_| bad("reference must be an absolute URL; use audio_b64 for local clips"))?;
    if !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
        return Err(bad(
            "reference URLs must not carry credentials or fragments",
        ));
    }
    let mut bytes = Vec::new();
    match url.scheme() {
        "file" => {
            let root = std::env::var_os("XRT_AUDIO_REFERENCE_ROOT")
                .ok_or_else(|| bad("file references require operator-configured XRT_AUDIO_REFERENCE_ROOT; otherwise send base64"))?;
            let root = std::fs::canonicalize(root)
                .map_err(|_| bad("configured reference root is unavailable"))?;
            let path = url.to_file_path().map_err(|_| bad("invalid file URL"))?;
            let path =
                std::fs::canonicalize(path).map_err(|_| bad("reference file is unavailable"))?;
            if !path.starts_with(root) {
                return Err(bad(
                    "reference file is outside the configured reference root",
                ));
            }
            std::fs::File::open(path)
                .map_err(|_| bad("cannot open reference file"))?
                .take(MAX_REQUEST_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| bad("cannot read reference file"))?;
        }
        "https" | "http" => {
            let origins = std::env::var("XRT_AUDIO_REFERENCE_ORIGINS").unwrap_or_default();
            if !origins.split(',').any(|origin| {
                url::Url::parse(origin.trim())
                    .ok()
                    .is_some_and(|allowed| allowed.origin() == url.origin())
            }) {
                return Err(bad("reference origin is not in operator-configured XRT_AUDIO_REFERENCE_ORIGINS; otherwise send base64"));
            }
            let response = ureq::AgentBuilder::new()
                .redirects(0)
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .get(url.as_str())
                .call()
                .map_err(|_| bad("reference download failed"))?;
            if response.status() != 200 {
                return Err(bad(
                    "reference download must return 200; redirects are not followed",
                ));
            }
            response
                .into_reader()
                .take(MAX_REQUEST_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| bad("cannot read reference response"))?;
        }
        _ => return Err(bad("unsupported reference URL scheme")),
    }
    bounded_reference(bytes)
}

fn bounded_reference(bytes: Vec<u8>) -> Result<Vec<u8>, Response> {
    if bytes.len() > MAX_REQUEST_BYTES {
        Err(err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "reference exceeds 32 MiB",
        ))
    } else {
        Ok(bytes)
    }
}

fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|s| {
            let value = (s.clamp(-1.0, 1.0) * 32768.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            value.to_le_bytes()
        })
        .collect()
}

fn audio_error(e: AudioError) -> Response {
    let status = match &e {
        AudioError::InvalidRequest(_) | AudioError::InvalidReference(_) => StatusCode::BAD_REQUEST,
        AudioError::ModelMissing { .. } => StatusCode::PRECONDITION_REQUIRED,
        AudioError::RuntimeIncompatible(_) => StatusCode::SERVICE_UNAVAILABLE,
        AudioError::Truncated { .. } | AudioError::Tokenizer(_) | AudioError::Inference(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    err(status, e.to_string())
}

/// Installation state is not a claim that the model loaded successfully.
pub(crate) async fn audio_status() -> Json<serde_json::Value> {
    let model_dir = xrt_audio::speech::default_model_dir();
    let paths = xrt_audio::chatterbox::ModelPaths::from_dir(&model_dir);
    let speech_present = [
        &paths.speech_encoder,
        &paths.embed_tokens,
        &paths.language_model,
        &paths.decoder,
        &paths.tokenizer,
    ]
    .iter()
    .all(|p| p.is_file());
    let asr = xrt_audio::whisper::Recognizer::default_dir();
    let asr_present = [
        "tokenizer.json",
        "generation_config.json",
        "onnx/encoder_model.onnx",
        "onnx/decoder_model.onnx",
        "onnx/decoder_with_past_model.onnx",
    ]
    .iter()
    .all(|p| asr.join(p).is_file());
    Json(serde_json::json!({
        "object": "audio.status",
        "model": "chatterbox-multilingual-v3",
        "model_files_present": speech_present,
        "recognizer_files_present": asr_present,
        "load_verified": false,
        "busy": speech_slot().available_permits() == 0,
        "sample_rate": 24000,
        "response_formats": ["wav", "pcm", "json"],
        "streaming": false,
        "scope": "local-server",
        "quality_check": "ASR heuristic; not a pronunciation guarantee"
    }))
}

fn library() -> xrt_audio::voices::VoiceLibrary {
    xrt_audio::voices::VoiceLibrary::open(&xrt_audio::voices::VoiceLibrary::default_root())
}

// ─── voice library ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct CreateVoiceRequest {
    /// Stable id pipelines refer to, e.g. `inanna-teaching`.
    id: String,
    #[serde(default)]
    name: Option<String>,
    /// Reference clip: WAV as base64, or a `data:`/`file://`/`http(s)://` URL.
    #[serde(default)]
    audio_b64: Option<String>,
    #[serde(default)]
    audio_url: Option<String>,
}

pub(crate) async fn list_voices() -> Response {
    match library().list() {
        Ok(v) => Json(serde_json::json!({ "object": "list", "data": v })).into_response(),
        Err(e) => audio_error(e),
    }
}

pub(crate) async fn get_voice(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    match library().info(&id) {
        Ok(v) => Json(v).into_response(),
        Err(AudioError::InvalidRequest(m)) if m.contains("not found") => {
            err(StatusCode::NOT_FOUND, m)
        }
        Err(e) => audio_error(e),
    }
}

pub(crate) async fn delete_voice(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    match library().delete(&id) {
        Ok(()) => Json(serde_json::json!({ "id": id, "deleted": true })).into_response(),
        Err(AudioError::InvalidRequest(m)) if m.contains("not found") => {
            err(StatusCode::NOT_FOUND, m)
        }
        Err(e) => audio_error(e),
    }
}

pub(crate) async fn create_voice(Json(req): Json<CreateVoiceRequest>) -> Response {
    match tokio::task::spawn_blocking(move || create_voice_sync(req)).await {
        Ok(response) => response,
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("voice worker failed: {e}"),
        ),
    }
}

fn create_voice_sync(req: CreateVoiceRequest) -> Response {
    let wav = match (req.audio_b64.as_deref(), req.audio_url.as_deref()) {
        (Some(b64), None) => match BASE64_STANDARD.decode(b64.trim()) {
            Ok(b) => b,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("invalid `audio_b64`: {e}")),
        },
        (None, Some(url)) => match load_reference(url) {
            Ok(b) => b,
            Err(response) => return response,
        },
        _ => {
            return err(
                StatusCode::BAD_REQUEST,
                "give exactly one of `audio_b64` or `audio_url`",
            )
        }
    };
    let name = req.name.unwrap_or_default();
    let id = req.id;
    match library().create(&id, &name, &wav) {
        Ok(v) => (StatusCode::CREATED, Json(v)).into_response(),
        Err(AudioError::InvalidRequest(m)) if m.contains("already exists") => {
            err(StatusCode::CONFLICT, m)
        }
        Err(e) => audio_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: serde_json::Value) -> SpeechRequest {
        serde_json::from_value(json).unwrap()
    }

    fn wav_b64() -> String {
        let x: Vec<f32> = (0..24_000 * 4)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        BASE64_STANDARD.encode(xrt_audio::audio::write_wav(&x, 24_000))
    }

    fn status(r: Result<Built, Response>) -> StatusCode {
        match r {
            Ok(_) => StatusCode::OK,
            Err(resp) => resp.status(),
        }
    }

    #[test]
    fn references_require_valid_data_or_operator_authority() {
        let wav = wav_b64();
        assert!(load_reference(&format!("data:audio/wav;base64,{wav}")).is_ok());
        for value in [
            "data:text/plain;base64,AA==",
            "data:audio/wav;base64,???",
            "C:/private.wav",
            "ftp://example.com/a.wav",
            "https://user:password@example.com/a.wav",
        ] {
            let response = load_reference(value).unwrap_err();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{value}");
        }
        assert_eq!(
            bounded_reference(vec![0; MAX_REQUEST_BYTES + 1])
                .unwrap_err()
                .status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[test]
    fn pcm_is_signed_16_bit_little_endian() {
        assert_eq!(
            pcm16(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
            [-32768i16, -32768, -16384, 0, 16384, 32767, 32767]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn admission_survives_a_disconnected_waiter() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = slots.clone().try_acquire_owned().unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
        });
        started_rx.await.unwrap();
        worker.abort(); // A running blocking worker is not cancelled by abort.
        assert!(slots.clone().try_acquire_owned().is_err());
        finish_tx.send(()).unwrap();
        worker.await.unwrap();
        assert!(slots.try_acquire_owned().is_ok());
    }

    #[test]
    fn empty_input_is_rejected() {
        assert_eq!(
            status(build(req(
                serde_json::json!({ "input": "  ", "voice_b64": wav_b64() })
            ))),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn missing_voice_is_rejected_with_guidance() {
        // An unknown saved id is a 404 that tells the caller how to clone one.
        assert_eq!(
            status(build(req(
                serde_json::json!({ "input": "hi", "voice": "definitely-not-a-saved-voice-xyz" })
            ))),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status(build(req(serde_json::json!({ "input": "hi" })))),
            StatusCode::BAD_REQUEST
        );
        // Two voice sources at once is ambiguous and refused.
        assert_eq!(
            status(build(req(
                serde_json::json!({ "input": "hi", "voice": "x", "voice_b64": wav_b64() })
            ))),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn unknown_preset_format_device_and_model_are_rejected() {
        for extra in [
            serde_json::json!({ "preset": "shouty" }),
            serde_json::json!({ "response_format": "mp3" }),
            serde_json::json!({ "device": "tpu" }),
            serde_json::json!({ "model": "tts-1" }),
            serde_json::json!({ "speed": 0.9 }),
        ] {
            let mut j = serde_json::json!({ "input": "hi", "voice_b64": wav_b64() });
            j.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert_eq!(
                status(build(req(j.clone()))),
                StatusCode::BAD_REQUEST,
                "{j}"
            );
        }
    }

    #[test]
    fn word_check_options_reach_the_pipeline() {
        let (o, _, _, _, f) = build(req(serde_json::json!({
            "input": "hi", "voice_b64": wav_b64(), "names": ["Ma'at"], "best_effort": true, "response_format": "json"
        })))
        .ok()
        .unwrap();
        assert!(o.word_check, "on by default");
        assert_eq!(o.names, ["Ma'at"]);
        assert!(o.accept_best_effort);
        assert!(matches!(f, Format::Json));
        // Turning the check off also turns off the breath default that needs it...
        let (o, ..) = build(req(
            serde_json::json!({ "input": "hi", "voice_b64": wav_b64(), "word_check": false }),
        ))
        .ok()
        .unwrap();
        assert_eq!(o.breath_reduction_db, 0.0);
        // ...but an explicit request for both is refused before model loading.
        assert_eq!(
            status(build(req(serde_json::json!({
                "input": "hi", "voice_b64": wav_b64(), "word_check": false, "breath_reduction_db": 12.0
            })))),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn documentary_preset_then_overrides() {
        let (o, _, _, rate, _) = build(req(serde_json::json!({
            "input": "hi", "voice_b64": wav_b64(), "preset": "documentary", "pause_scale": 1.2, "device": "cuda:1"
        })))
        .ok()
        .unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(o.cfg_weight, 0.5, "preset applied");
        assert_eq!(o.pause_scale, 1.2, "explicit field wins over preset");
        assert_eq!(o.device, Device::Cuda(1));
    }
}
