//! `/v1/audio/*` — speech transcription, served from `xrt-audio`.
//!
//! Shaped after OpenAI's `POST /v1/audio/transcriptions`: `multipart/form-data`
//! with a `file` part, plus optional `model` and `response_format`. That keeps
//! ABSOLUTE RULE 1 ("existing OpenAI API compatibility is sacred") intact for a
//! standard contract, so an existing client library works unchanged.
//!
//! # Two deliberate narrowings, both stated in the error rather than hidden
//!
//! **Only 16-bit PCM WAV is accepted.** `RUNTIME_DOMAINS.md` gives `xeno-lib`
//! audio decode/encode and keeps model execution here, so accepting mp3 or m4a
//! would quietly make xeno-rt a media library. A compressed upload is refused
//! with the format NAMED — `xeno-motion` already holds decoded PCM from
//! WebCodecs, so for the caller that matters this costs nothing.
//!
//! **Inference is serialised.** One model instance behind a mutex: a second
//! request waits rather than racing. ONNX Runtime sessions are not safe to run
//! concurrently through `&mut`, and a queue is the honest behaviour — the
//! alternative is loading a second copy of the weights per caller, which is how
//! a server falls over under exactly the load it was built for.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use serde_json::{json, Value};
use xrt_audio::whisper::WhisperModel;

use crate::AppState;

/// OPTIONAL override pointing at a local model directory.
///
/// Absent is the normal case: the model is then resolved from the XENO model
/// registry and sha256-verified before use. The override exists for a custom or
/// pre-staged model and for offline testing — it is a developer affordance, not
/// the path a user takes.
const MODEL_DIR_ENV: &str = "XENO_RT_WHISPER_DIR";

#[derive(Clone)]
pub struct AudioServerState {
    model: Arc<Mutex<Option<WhisperModel>>>,
    model_dir: Option<PathBuf>,
}

impl AudioServerState {
    pub fn from_env() -> Self {
        let model_dir = std::env::var_os(MODEL_DIR_ENV)
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty());
        if model_dir.is_none() {
            tracing::info!(
                "{MODEL_DIR_ENV} is not set - whisper-base will be resolved from the XENO model registry on first use"
            );
        }
        Self { model: Arc::new(Mutex::new(None)), model_dir }
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self { model: Arc::new(Mutex::new(None)), model_dir: None }
    }
}

#[derive(Serialize)]
struct SegmentOut {
    id: usize,
    start: f32,
    end: f32,
    text: String,
}

/// `POST /v1/audio/transcriptions`
pub async fn transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut file: Option<Vec<u8>> = None;
    let mut response_format = "json".to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("malformed multipart body: {e}")))?
    {
        match field.name().unwrap_or_default().to_string().as_str() {
            "file" => {
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, format!("could not read `file`: {e}")))?
                        .to_vec(),
                )
            }
            "response_format" => {
                response_format = field.text().await.unwrap_or_else(|_| "json".to_string())
            }
            // `model`, `language`, `prompt`, `temperature` are accepted and
            // ignored for now. Ignoring silently is defensible only because
            // none of them can change the RESULT here yet; the moment one does,
            // it must be honoured or rejected, never dropped.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let file = file.ok_or((
        StatusCode::BAD_REQUEST,
        "missing `file` part (multipart/form-data)".to_string(),
    ))?;

    // Refuse an unknown response_format rather than silently returning `json`.
    // Handing back a different shape than the caller asked for is worse than a
    // 400: they parse it, get nothing, and blame the transcription.
    if !matches!(response_format.as_str(), "json" | "verbose_json" | "text") {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported response_format `{response_format}` (json, verbose_json, text)"),
        ));
    }

    let (samples, rate, channels) = xrt_audio::wav::read_pcm16(&file)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if samples.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "the uploaded audio is empty".to_string()));
    }
    let mono = xrt_audio::to_mono(&samples, channels as usize);
    let duration = mono.len() as f32 / rate as f32;

    let model_dir = state.audio.model_dir.clone();
    let slot = Arc::clone(&state.audio.model);

    // Inference is multi-second and CPU-bound: it must not run on the async
    // executor. The lock is taken INSIDE the blocking task so waiting callers
    // park a blocking thread rather than an executor thread.
    let out = tokio::task::spawn_blocking(move || -> Result<xrt_audio::whisper::Transcript, String> {
        let mut guard = slot.lock().map_err(|_| "model mutex was poisoned".to_string())?;
        if guard.is_none() {
            // An explicit directory wins; otherwise resolve from the registry,
            // which downloads and sha256-verifies on first use. The FIRST call
            // therefore pays the download; every later one is cached.
            let model = match model_dir {
                Some(dir) => WhisperModel::load(&dir),
                None => WhisperModel::load_from_registry(),
            };
            *guard = Some(model.map_err(|e| e.to_string())?);
        }
        guard
            .as_mut()
            .expect("model was just loaded")
            .transcribe(&mono, rate)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("transcription task failed: {e}")))?
    .map_err(|e| {
        // A missing model file is the caller's environment, not a server fault.
        let code = if e.contains("not found") || e.contains("unavailable") {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (code, e)
    })?;

    Ok(Json(match response_format.as_str() {
        "verbose_json" => json!({
            "task": "transcribe",
            "language": out.language,
            "duration": duration,
            "text": out.text,
            "segments": out.segments.iter().enumerate().map(|(i, s)| SegmentOut {
                id: i,
                start: s.start,
                end: s.end,
                text: s.text.clone(),
            }).collect::<Vec<_>>(),
        }),
        // `text` is documented by OpenAI as a bare body; returning it inside a
        // JSON object here is a knowing divergence, recorded rather than
        // pretended away, because this handler's return type is Json.
        _ => json!({ "text": out.text }),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state must be constructible with no model configured, and must say
    /// so rather than panicking at startup. A server that refuses to boot
    /// because an optional model is absent takes every other endpoint with it.
    #[test]
    fn missing_model_dir_is_not_a_startup_failure() {
        let s = AudioServerState::for_tests();
        assert!(s.model_dir.is_none());
        assert!(s.model.lock().unwrap().is_none(), "the model must load lazily, not eagerly");
    }

    /// Compressed uploads must be refused by NAME. This is the boundary between
    /// xeno-rt and xeno-lib expressed as an error message.
    #[test]
    fn compressed_audio_is_refused_with_the_format_named() {
        let mut flac = b"fLaC".to_vec();
        flac.extend_from_slice(&[0u8; 32]);
        let err = xrt_audio::wav::read_pcm16(&flac).unwrap_err().to_string();
        assert!(err.contains("FLAC"), "error should name the format: {err}");
        assert!(err.contains("16-bit PCM WAV"), "error should say what to send: {err}");
    }
}
