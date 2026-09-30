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
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use xrt_audio::{chatterbox::Device, AudioError, Preset, SpeechOptions};

use crate::AppState;

/// Upper bound on one request's script. Long-form narration is chunked
/// internally; this caps the work one HTTP request can queue.
pub(crate) const MAX_INPUT_CHARS: usize = 50_000;
/// Reference clips are trimmed to 10 s by the model; this bounds the upload.
pub(crate) const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Server-owned model lifetime and admission, shared by all audio routes.
pub(crate) struct AudioServerState {
    runtime: Arc<xrt_audio::speech::AudioRuntime>,
    slot: Arc<Semaphore>,
    draining: std::sync::atomic::AtomicBool,
    active: std::sync::Mutex<Option<Arc<xrt_audio::control::InferenceControl>>>,
    gpu_lease: std::sync::Mutex<Option<xrt_runtime::GpuAllocationLease>>,
}

impl Default for AudioServerState {
    fn default() -> Self {
        Self {
            runtime: Arc::new(Default::default()),
            slot: Arc::new(Semaphore::new(1)),
            draining: std::sync::atomic::AtomicBool::new(false),
            active: std::sync::Mutex::new(None),
            gpu_lease: std::sync::Mutex::new(None),
        }
    }
}

impl AudioServerState {
    fn admit_device(
        &self,
        resources: &xrt_runtime::GpuResourceManager,
        requested: Device,
    ) -> Result<Device, AudioError> {
        if requested == Device::Cpu {
            self.runtime.unload()?;
            self.gpu_lease
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            return Ok(Device::Cpu);
        }
        let ordinal = resources.config().device_ordinal;
        if let Device::Cuda(id) = requested {
            if id as usize != ordinal {
                return Err(AudioError::InvalidRequest(
                    "audio CUDA device must match the server's XRT_CUDA_DEVICE shared budget"
                        .into(),
                ));
            }
        }
        let mut slot = self.gpu_lease.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Ok(Device::Cuda(ordinal as i32));
        }
        // Sum of the seven ONNX session arena caps (14592 MiB), plus a separate
        // 512 MiB driver/library allowance. Actual admission is conservative.
        const RESERVATION: u64 = (14592 + 512) * 1024 * 1024;
        let attempt = (|| {
            let device = std::panic::catch_unwind(|| xrt_cuda::CudaDevice::new(ordinal))
                .map_err(|_| {
                    AudioError::RuntimeIncompatible("CUDA driver libraries unavailable".into())
                })?
                .map_err(|e| AudioError::RuntimeIncompatible(e.to_string()))?;
            let (free, total) = device
                .memory_info()
                .map_err(|e| AudioError::Inference(e.to_string()))?;
            let config = resources.config();
            if free.saturating_sub(config.reserved_bytes()) < RESERVATION {
                return Err(AudioError::Inference(format!("audio needs {RESERVATION} bytes of GPU headroom plus server reserve; {free} free")));
            }
            let budget = free
                .min((total as f64 * config.memory_fraction as f64) as u64)
                .saturating_sub(config.reserved_bytes());
            let arena = resources.allocation_arena();
            arena
                .initialize_budget(budget)
                .map_err(|e| AudioError::Inference(e.to_string()))?;
            arena
                .reserve(xrt_runtime::GpuAllocationClass::ModelWeights, RESERVATION)
                .map_err(|e| AudioError::Inference(e.to_string()))
        })();
        match attempt {
            Ok(lease) => {
                *slot = Some(lease);
                Ok(Device::Cuda(ordinal as i32))
            }
            Err(error) if requested == Device::Auto => {
                tracing::warn!(%error,"audio auto device uses CPU after GPU admission refusal");
                Ok(Device::Cpu)
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) fn drain(&self) {
        self.draining
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(control) = self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            control.cancel();
        }
    }
}

struct CancelOnDrop(Arc<xrt_audio::control::InferenceControl>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct ActiveRequest(Arc<AudioServerState>);
impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.0
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
}

struct DeadlineTask(tokio::task::JoinHandle<()>);
impl Drop for DeadlineTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(default)]
    timeout_seconds: Option<u64>,
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
    speech_with_control(state, req, Arc::new(Default::default()), || Ok(()), |_| {}).await
}

pub(crate) async fn speech_with_control(
    state: AppState,
    req: SpeechRequest,
    control: Arc<xrt_audio::control::InferenceControl>,
    on_admitted: impl FnOnce() -> Result<(), String> + Send + 'static,
    progress: impl FnMut(&xrt_audio::ChunkReport) + Send + 'static,
) -> Response {
    if state
        .audio
        .draining
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "audio runtime is draining; restart to admit new work",
        );
    }
    let timeout = req.timeout_seconds.unwrap_or(1800);
    if !(1..=7200).contains(&timeout) {
        return err(StatusCode::BAD_REQUEST, "timeout_seconds must be 1..=7200");
    }
    let permit = match state.audio.slot.clone().try_acquire_owned() {
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
    if let Err(message) = on_admitted() {
        return err(StatusCode::CONFLICT, message);
    }
    *state.audio.active.lock().unwrap_or_else(|e| e.into_inner()) = Some(control.clone());
    let active = ActiveRequest(state.audio.clone());
    let _cancel_on_disconnect = CancelOnDrop(control.clone());
    // Cover drain racing with admission registration.
    if state
        .audio
        .draining
        .load(std::sync::atomic::Ordering::Acquire)
    {
        control.cancel();
    }
    let deadline_control = control.clone();
    let deadline = DeadlineTask(tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(timeout)).await;
        deadline_control.cancel();
    }));
    let auto_direction = req.direction.as_ref().and_then(|v| v.as_str()) == Some("auto");
    // Share the admission lifetime across every blocking stage, including
    // reference loading and the optional text director. Dropping the HTTP
    // future never makes an orphaned native worker invisible to admission.
    let lease = Arc::new((permit, active, deadline));
    let loading_lease = lease.clone();
    let loading_control = control.clone();
    let built = tokio::task::spawn_blocking(move || {
        let _lease = loading_lease;
        loading_control
            .check()
            .map_err(|e| Box::new(audio_error(e)))?;
        build(req)
    })
    .await;
    let (mut opts, script, reference, rate, format) = match built {
        Ok(Ok(value)) => value,
        Ok(Err(response)) => return *response,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reference worker failed: {e}"),
            )
        }
    };
    if auto_direction {
        match crate::audio_direction::direct(&state, &script, &opts, control.clone(), lease.clone())
            .await
        {
            Ok(plan) => opts.direction = Some(plan),
            Err((status, message)) => return err(status, message),
        }
    }
    let audio = state.audio.clone();
    let resources = state.gpu_resources.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        opts.device = audio.admit_device(&resources, opts.device)?;
        let result = audio.runtime.synthesize_controlled(
            &script,
            &reference,
            rate,
            &opts,
            &xrt_audio::speech::SignalValidator::default(),
            &control,
            progress,
        );
        if result.is_err() {
            // Controlled synthesis drops failed/cancelled sessions before the
            // reservation is released, so other modalities cannot reuse it early.
            if audio.runtime.unload().is_ok() {
                audio
                    .gpu_lease
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
            }
        }
        result.map(|out| (out, format))
    })
    .await;
    match result {
        Ok(Ok((out, format))) => speech_response(out, format),
        Ok(Err(error)) => audio_error(error),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("speech worker failed: {e}"),
        ),
    }
}

#[cfg(feature = "transcription")]
pub(crate) async fn transcriptions(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> Response {
    if state
        .audio
        .draining
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return err(StatusCode::SERVICE_UNAVAILABLE, "audio runtime is draining");
    }
    let Ok(permit) = state.audio.slot.clone().try_acquire_owned() else {
        return err(StatusCode::TOO_MANY_REQUESTS, "audio runtime is busy");
    };
    let mut file = None;
    let mut format = "json".to_string();
    let mut fields = std::collections::HashSet::new();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(_) => return err(StatusCode::BAD_REQUEST, "invalid audio multipart body"),
        };
        let name = field.name().unwrap_or("").to_string();
        if !fields.insert(name.clone()) {
            return err(StatusCode::BAD_REQUEST, "duplicate multipart field");
        }
        if name == "file" {
            file = match field.bytes().await {
                Ok(bytes) => Some(bytes),
                Err(_) => return err(StatusCode::BAD_REQUEST, "invalid audio file part"),
            };
            continue;
        }
        let value = match field.text().await {
            Ok(value) => value,
            Err(_) => return err(StatusCode::BAD_REQUEST, "invalid multipart text"),
        };
        match name.as_str() {
            "response_format" if matches!(value.as_str(), "json" | "verbose_json" | "text") => {
                format = value
            }
            "model" if matches!(value.as_str(), "whisper-base" | "whisper-1") => {}
            "language" if value == "en" => {}
            "temperature" if value == "0" || value == "0.0" => {}
            _ => {
                return err(
                    StatusCode::BAD_REQUEST,
                    format!("unsupported transcription option `{name}`"),
                )
            }
        }
    }
    let Some(file) = file else {
        return err(StatusCode::BAD_REQUEST, "missing file part");
    };
    let control = Arc::new(xrt_audio::control::InferenceControl::default());
    *state.audio.active.lock().unwrap_or_else(|e| e.into_inner()) = Some(control.clone());
    let active = ActiveRequest(state.audio.clone());
    let _cancel = CancelOnDrop(control.clone());
    if state
        .audio
        .draining
        .load(std::sync::atomic::Ordering::Acquire)
    {
        control.cancel();
    }
    let runtime = state.audio.runtime.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = active;
        let (samples, rate) = xrt_audio::audio::read_wav(&file)?;
        if samples.len() as f64 / rate as f64 > 600.0 {
            return Err(AudioError::InvalidRequest(
                "transcription is limited to 600 seconds per request".into(),
            ));
        }
        let duration = samples.len() as f32 / rate as f32;
        runtime
            .transcribe_base_controlled(&samples, rate, &control)
            .map(|out| (out, duration))
    })
    .await;
    match result {
        Ok(Ok((out, duration))) => match format.as_str() {
            "text" => ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], out.text).into_response(),
            "verbose_json" => Json(serde_json::json!({"task":"transcribe","language":out.language,"duration":duration,"text":out.text,
                "segments":out.segments.iter().enumerate().map(|(id,s)| serde_json::json!({"id":id,"start":s.start,"end":s.end,"text":s.text})).collect::<Vec<_>>()})).into_response(),
            _ => Json(serde_json::json!({"text":out.text})).into_response(),
        },
        Ok(Err(e)) => audio_error(e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("transcription worker failed: {e}")),
    }
}

pub(crate) fn freeze_job(req: SpeechRequest) -> Result<SpeechRequest, Box<Response>> {
    let (_, _, samples, rate, _) = build(req.clone())?;
    let mut frozen = req;
    frozen.voice = None;
    frozen.voice_url = None;
    frozen.voice_b64 = Some(BASE64_STANDARD.encode(xrt_audio::audio::write_wav(&samples, rate)));
    frozen.response_format = Some("json".into());
    Ok(frozen)
}

pub(crate) fn accepts_jobs(state: &AppState) -> bool {
    !state
        .audio
        .draining
        .load(std::sync::atomic::Ordering::Acquire)
}

type Built = (SpeechOptions, String, Vec<f32>, u32, Format);

#[derive(Clone, Copy)]
enum Format {
    Wav,
    Pcm,
    Json,
}

fn build(req: SpeechRequest) -> Result<Built, Box<Response>> {
    let bad = |m: String| Box::new(err(StatusCode::BAD_REQUEST, m));
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
        library().load(id).map_err(|e| Box::new(match e {
            AudioError::InvalidRequest(m) if m.contains("not found") => err(StatusCode::NOT_FOUND, format!(
                "{m}. Built-in voices are not provided: clone one with POST /v1/audio/voices, or send `voice_b64`"
            )),
            other => audio_error(other),
        }))?
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
    xrt_audio::speech::validate_options(&opts).map_err(|e| Box::new(audio_error(e)))?;
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
fn load_reference(reference: &str) -> Result<Vec<u8>, Box<Response>> {
    use std::io::Read;
    let bad = |m| Box::new(err(StatusCode::BAD_REQUEST, m));
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

fn bounded_reference(bytes: Vec<u8>) -> Result<Vec<u8>, Box<Response>> {
    if bytes.len() > MAX_REQUEST_BYTES {
        Err(Box::new(err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "reference exceeds 32 MiB",
        )))
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
        AudioError::Cancelled => StatusCode::REQUEST_TIMEOUT,
        AudioError::ModelMissing { .. } => StatusCode::PRECONDITION_REQUIRED,
        AudioError::RuntimeIncompatible(_) => StatusCode::SERVICE_UNAVAILABLE,
        AudioError::Truncated { .. } | AudioError::Tokenizer(_) | AudioError::Inference(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    err(status, e.to_string())
}

/// Installation state is not a claim that the model loaded successfully.
pub(crate) async fn audio_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let (speech_provider, asr_provider) = state.audio.runtime.status();
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
        "load_verified": speech_provider.is_some(),
        "provider": speech_provider,
        "asr_provider": asr_provider,
        "draining": state.audio.draining.load(std::sync::atomic::Ordering::Acquire),
        "busy": state.audio.slot.available_permits() == 0,
        "sample_rate": 24000,
        "response_formats": ["wav", "pcm", "json"],
        "streaming": false,
        "scope": "local-server",
        "quality_check": "ASR heuristic; not a pronunciation guarantee"
    }))
}

pub(crate) async fn audio_unload(State(state): State<AppState>) -> Response {
    let Ok(_permit) = state.audio.slot.clone().try_acquire_owned() else {
        return err(
            StatusCode::CONFLICT,
            "audio runtime is busy; cancel or drain before unloading",
        );
    };
    match state.audio.runtime.unload() {
        Ok(()) => {
            state
                .audio
                .gpu_lease
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            Json(serde_json::json!({"object":"audio.unload", "unloaded":true})).into_response()
        }
        Err(e) => audio_error(e),
    }
}

pub(crate) async fn audio_drain(State(state): State<AppState>) -> Response {
    state.audio.drain();
    Json(serde_json::json!({"object":"audio.drain", "draining":true,
        "active":state.audio.slot.available_permits() == 0}))
    .into_response()
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
            Err(response) => return *response,
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

    fn status(r: Result<Built, Box<Response>>) -> StatusCode {
        match r {
            Ok(_) => StatusCode::OK,
            Err(resp) => resp.status(),
        }
    }

    #[test]
    fn dropping_request_cancels_without_releasing_worker_admission() {
        let state = Arc::new(AudioServerState::default());
        let permit = state.slot.clone().try_acquire_owned().unwrap();
        let control = Arc::new(xrt_audio::control::InferenceControl::default());
        *state.active.lock().unwrap() = Some(control.clone());
        let worker = Arc::new((permit, ActiveRequest(state.clone())));
        let caller = worker.clone();
        drop(CancelOnDrop(control.clone()));
        drop(caller);
        assert!(control.is_cancelled());
        assert_eq!(state.slot.available_permits(), 0);
        drop(worker);
        assert_eq!(state.slot.available_permits(), 1);
        assert!(state.active.lock().unwrap().is_none());
    }

    #[test]
    fn draining_cancels_active_work_and_remains_closed() {
        let state = AudioServerState::default();
        let control = Arc::new(xrt_audio::control::InferenceControl::default());
        *state.active.lock().unwrap() = Some(control.clone());
        state.drain();
        assert!(control.is_cancelled());
        assert!(state.draining.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(state.runtime.status(), (None, None));
        assert!(state.runtime.unload().is_ok());
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
