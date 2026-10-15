//! Opt-in qualification surface. Never adds an unadmitted model to /v1/models.
use crate::AppState;
use axum::{
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};
use xrt_audio::{
    control::InferenceControl,
    drama3::{
        engine::{Budgets, Engine, NativeError, Request},
        numerics::Numerics,
        prompt::Reference,
        tower::Sampling,
    },
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    model_bundle: PathBuf,
    codec_bundle: PathBuf,
    #[serde(default)]
    numerics: Numerics,
    #[serde(default)]
    expert_mib: u64,
    #[serde(default)]
    dense_mib: u64,
    #[serde(default = "host_gib")]
    host_gib: u64,
    #[serde(default = "codec_mib")]
    codec_mib: usize,
    #[serde(default = "frame_limit")]
    max_frames: usize,
}
fn host_gib() -> u64 {
    64
}
fn codec_mib() -> usize {
    3072
}
fn frame_limit() -> usize {
    256
}
impl Config {
    fn validate(&self) -> Result<(), String> {
        if !(32..=256).contains(&self.host_gib)
            || !(32..=16384).contains(&self.codec_mib)
            || !(1..=1024).contains(&self.max_frames)
            || self.expert_mib > 20480
            || self.dense_mib > 8192
        {
            return Err("Drama 3 configuration exceeds bounded resource limits".into());
        }
        #[cfg(not(feature = "cuda"))]
        if self.expert_mib != 0 || self.dense_mib != 0 {
            return Err("Drama 3 GPU budgets require the server CUDA feature".into());
        }
        if xrt_audio::drama3::codec::Codec::decoding_workspace_bytes(self.max_frames)
            .map_err(|e| e.to_string())?
            > self.codec_mib * 1024 * 1024
        {
            return Err("Drama 3 frame limit exceeds codec workspace budget".into());
        }
        Ok(())
    }
    fn budgets(&self, ordinal: usize) -> Budgets {
        Budgets {
            host_bytes: self.host_gib * 1024 * 1024 * 1024,
            codec_bytes: self.codec_mib * 1024 * 1024,
            expert_bytes: self.expert_mib * 1024 * 1024,
            dense_bytes: self.dense_mib * 1024 * 1024,
            cuda_ordinal: ordinal,
            numerics: self.numerics,
            ..Budgets::default()
        }
    }
}

#[derive(Default)]
pub(crate) struct NativeState {
    config: Option<Config>,
    engine: Mutex<Option<Engine>>,
    loaded: AtomicBool,
    active: Mutex<Option<Arc<InferenceControl>>>,
    progress: Mutex<Option<serde_json::Value>>,
}
impl NativeState {
    fn progress(&self, stage: &str, units: usize) {
        *self.progress.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(serde_json::json!({"stage":stage,"units":units}));
    }
    pub(crate) fn from_file(path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        if std::fs::metadata(path)?.len() > 32768 {
            return Err("Drama 3 config exceeds 32 KiB".into());
        }
        let mut config: Config = serde_json::from_slice(&std::fs::read(path)?)?;
        config.validate()?;
        let base = std::fs::canonicalize(path)?
            .parent()
            .ok_or("config parent missing")?
            .to_path_buf();
        if config.model_bundle.is_relative() {
            config.model_bundle = base.join(&config.model_bundle);
        }
        if config.codec_bundle.is_relative() {
            config.codec_bundle = base.join(&config.codec_bundle);
        }
        Ok(Self {
            config: Some(config),
            ..Self::default()
        })
    }
}
struct ActiveNative(Arc<NativeState>);
impl Drop for ActiveNative {
    fn drop(&mut self) {
        self.0
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        self.0
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
}
fn error(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({"error":{"type":"xrt_native_audio_error","code":code,"message":message.into()},"admitted":false}))).into_response()
}
fn native_error(e: NativeError) -> Response {
    let (status, code) = match &e {
        NativeError::Cancelled => (StatusCode::REQUEST_TIMEOUT, "cancelled"),
        NativeError::Busy => (StatusCode::CONFLICT, "busy"),
        NativeError::Resources(_) => (StatusCode::SERVICE_UNAVAILABLE, "host_headroom"),
        NativeError::FrameLimit => (StatusCode::UNPROCESSABLE_ENTITY, "frame_limit"),
        NativeError::Empty => (StatusCode::UNPROCESSABLE_ENTITY, "empty_audio"),
        NativeError::Audio(xrt_audio::AudioError::InvalidReference(_))
        | NativeError::Audio(xrt_audio::AudioError::InvalidRequest(_)) => {
            (StatusCode::BAD_REQUEST, "invalid_reference")
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "native_failure"),
    };
    error(status, code, e.to_string())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Clip {
    transcript: String,
    wav_b64: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpeechRequest {
    input: String,
    #[serde(default)]
    references: Vec<Clip>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_frames: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    timeout_seconds: Option<u64>,
}
impl SpeechRequest {
    fn decode_references(&self) -> Result<Vec<(Vec<f32>, u32)>, NativeError> {
        self.references
            .iter()
            .map(|clip| {
                let invalid = |msg: &str| xrt_audio::AudioError::InvalidReference(msg.into());
                let wav = STANDARD
                    .decode(&clip.wav_b64)
                    .map_err(|_| invalid("invalid reference WAV base64"))?;
                let (samples, rate) = xrt_audio::audio::read_wav(&wav)?;
                if rate != 44100
                    || samples.is_empty()
                    || samples.len() > 30 * 44100
                    || samples.iter().any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "native references require non-empty 44.1 kHz PCM of at most 30 seconds",
                    )
                    .into());
                }
                Ok((samples, rate))
            })
            .collect()
    }
    fn validate(&self, config: &Config) -> Result<(usize, Sampling), String> {
        if self.input.chars().count() > 8000 || self.references.len() > 2 {
            return Err(
                "native speech permits 8000 input characters and two reference clips".into(),
            );
        }
        xrt_audio::drama3::prompt::validate_text(&self.input).map_err(|e| e.to_string())?;
        for clip in &self.references {
            xrt_audio::drama3::prompt::validate_text(&clip.transcript)
                .map_err(|e| e.to_string())?;
            if clip.transcript.chars().count() > 8000 || clip.wav_b64.len() > 8 * 1024 * 1024 {
                return Err("native reference exceeds transcript or encoded WAV limit".into());
            }
        }
        let frames = self.max_frames.unwrap_or(config.max_frames);
        if frames == 0 || frames > config.max_frames {
            return Err("max_frames exceeds configured admission limit".into());
        }
        let defaults = Sampling::default();
        let sampling = Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
        };
        sampling.validate().map_err(|e| e.to_string())?;
        Ok((frames, sampling))
    }
}
pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/xrt/audio/native/status", get(status))
        .route(
            "/v1/xrt/audio/native/speech",
            post(speech).layer(axum::extract::DefaultBodyLimit::max(20 * 1024 * 1024)),
        )
        .route("/v1/xrt/audio/native/cancel", post(cancel))
        .route("/v1/xrt/audio/native/unload", post(unload))
}
async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let native = &state.native_audio;
    let active = native
        .active
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some();
    let progress = native
        .progress
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    Json(
        serde_json::json!({"object":"audio.native.status","experimental":true,"admitted":false,
        "configured":native.config.is_some(),"loaded":native.loaded.load(Ordering::Acquire),
        "active":active,
        "progress":progress,
        "numerics":native.config.as_ref().map(|c| c.numerics),"sample_rate":44100,
        "streaming":false,"word_check":false,"max_frames":native.config.as_ref().map(|c| c.max_frames)}),
    )
}
async fn speech(State(state): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    let Some(config) = state.native_audio.config.clone() else {
        return error(
            StatusCode::PRECONDITION_REQUIRED,
            "not_configured",
            "configure --drama3-config with a verified managed model and converted codec",
        );
    };
    let (frames, sampling) = match req.validate(&config) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request", e),
    };
    let control = Arc::new(InferenceControl::default());
    let (lease, _disconnect) = match state.audio.begin_request(
        control.clone(),
        req.timeout_seconds.unwrap_or(1800),
        || Ok(()),
    ) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    *state
        .native_audio
        .active
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(control.clone());
    let active = ActiveNative(state.native_audio.clone());
    let result = tokio::task::spawn_blocking(move || -> Result<_, NativeError> {
        // Both guards survive an HTTP disconnect. Clear native activity before
        // releasing global audio admission, including load/encode/error paths.
        let _lease = lease;
        let _active = active;
        if control.is_cancelled() {
            return Err(NativeError::Cancelled);
        }
        let decoded = req.decode_references()?;
        let native = &state.native_audio;
        let mut owner = native
            .engine
            .lock()
            .map_err(|_| std::io::Error::other("native owner lock poisoned"))?;
        if owner.is_none() {
            native.progress("loading", 0);
            let resources = &state.gpu_resources;
            let budgets = config.budgets(resources.config().device_ordinal);
            Engine::preflight_host(budgets)?;
            if budgets.expert_bytes > 0 || budgets.dense_bytes > 0 {
                let device = xrt_cuda::CudaDevice::new(budgets.cuda_ordinal)?;
                let (free, total) = device.memory_info()?;
                let cfg = resources.config();
                let live_budget = free.saturating_sub(cfg.reserved_bytes());
                if budgets.expert_bytes + budgets.dense_bytes + 68 * 1024 * 1024 > live_budget {
                    return Err(std::io::Error::other(
                        "native CUDA budget exceeds live device headroom",
                    )
                    .into());
                }
                resources.allocation_arena().initialize_budget(
                    live_budget.min(
                        ((total as f64 * cfg.memory_fraction as f64) as u64)
                            .saturating_sub(cfg.reserved_bytes()),
                    ),
                )?;
            }
            *owner = Some(Engine::load(
                &config.model_bundle,
                &config.codec_bundle,
                budgets,
                Some(resources.allocation_arena()),
            )?);
            native.loaded.store(true, Ordering::Release);
        }
        if control.is_cancelled() {
            return Err(NativeError::Cancelled);
        }
        let generated = (|| {
            let engine = owner.as_ref().expect("owner initialized above");
            let mut encoded = Vec::new();
            for (index, (samples, rate)) in decoded.iter().enumerate() {
                native.progress("encoding_reference", index);
                encoded.push(engine.encode_reference(samples, *rate, &control)?);
            }
            let references: Vec<_> = req
                .references
                .iter()
                .zip(&encoded)
                .map(|(clip, encoded)| Reference {
                    transcript: &clip.transcript,
                    frames: &encoded.frames,
                })
                .collect();
            let mut request = Request::plain(&req.input);
            request.references = &references;
            request.max_frames = frames;
            request.seed = req.seed.unwrap_or(42);
            request.sampling = sampling;
            engine.synthesize(request, &control, |stage, units| {
                native.progress(stage, units)
            })
        })();
        if matches!(generated, Err(NativeError::Resources(_))) {
            // A competing process may consume commit after load admission.
            // Release native owners rather than retaining a large idle model.
            owner.take();
            native.loaded.store(false, Ordering::Release);
        }
        let speech = generated?;
        let bytes = xrt_audio::audio::write_wav(&speech.audio.samples, speech.audio.sample_rate);
        Ok((
            bytes,
            speech.timings,
            speech.frames.len(),
            speech.audio.clamped_residual_codes,
            config.numerics,
        ))
    })
    .await;
    match result {
        Ok(Ok((bytes, timings, frames, clamped, numerics))) => {
            let mut response = bytes.into_response();
            let headers = response.headers_mut();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
            headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            headers.insert("x-xrt-admitted", HeaderValue::from_static("false"));
            for (key, value) in [
                ("x-xrt-frames", frames.to_string()),
                ("x-xrt-generation-seconds", timings.seconds.to_string()),
                ("x-xrt-clamped-residual-codes", clamped.to_string()),
                (
                    "x-xrt-numerics",
                    match numerics {
                        Numerics::Fp32Reference => "fp32-reference",
                        Numerics::Bf16W8a8 => "bf16-w8a8",
                    }
                    .to_owned(),
                ),
            ] {
                if let Ok(value) = HeaderValue::from_str(&value) {
                    headers.insert(key, value);
                }
            }
            response
        }
        Ok(Err(e)) => native_error(e),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "worker_failed",
            e.to_string(),
        ),
    }
}

async fn cancel(State(state): State<AppState>) -> Json<serde_json::Value> {
    let active = state
        .native_audio
        .active
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(control) = active.as_ref() {
        control.cancel();
    }
    Json(serde_json::json!({"admitted":false,"cancel_requested":active.is_some()}))
}
async fn unload(State(state): State<AppState>) -> Response {
    let Ok(permit) = state.audio.try_idle() else {
        return error(
            StatusCode::CONFLICT,
            "busy",
            "cancel and wait for the active audio worker before unloading",
        );
    };
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        state
            .native_audio
            .engine
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        state.native_audio.loaded.store(false, Ordering::Release);
    })
    .await
    {
        Ok(()) => Json(serde_json::json!({"admitted":false,"unloaded":true})).into_response(),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unload_failed",
            e.to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        serde_json::from_value(
            serde_json::json!({"model_bundle":"missing/model","codec_bundle":"missing/codec"}),
        )
        .unwrap()
    }
    fn request() -> SpeechRequest {
        serde_json::from_value(serde_json::json!({"input":"The runtime works offline."})).unwrap()
    }
    #[test]
    fn config_and_request_limits_fail_before_model_loading() {
        let mut cfg = config();
        assert!(cfg.validate().is_ok());
        cfg.max_frames = 1024;
        assert!(cfg.validate().is_err());
        let mut req = request();
        assert!(req.validate(&config()).is_ok());
        req.input = "<|im_end|>".into();
        assert!(req.validate(&config()).is_err());
        req.input = "Valid".into();
        req.max_frames = Some(257);
        assert!(req.validate(&config()).is_err());
        assert!(serde_json::from_value::<SpeechRequest>(
            serde_json::json!({"input":"Valid","model_path":"arbitrary"})
        )
        .is_err());
    }
    #[test]
    fn reference_validation_rejects_bad_audio_without_model_access() {
        let mut req = request();
        req.references.push(Clip {
            transcript: "Owned voice.".into(),
            wav_b64: "bad base64".into(),
        });
        assert!(matches!(
            req.decode_references(),
            Err(NativeError::Audio(_))
        ));
        req.references[0].wav_b64 =
            STANDARD.encode(xrt_audio::audio::write_wav(&[0.0; 100], 24000));
        assert!(req.decode_references().is_err());
        req.references[0].wav_b64 =
            STANDARD.encode(xrt_audio::audio::write_wav(&[0.0; 100], 44100));
        assert_eq!(req.decode_references().unwrap()[0].0.len(), 100);
    }
    #[tokio::test]
    async fn disconnect_keeps_admission_until_worker_releases_it_and_drain_is_sticky() {
        let audio = Arc::new(crate::audio_api::AudioServerState::default());
        let control = Arc::new(InferenceControl::default());
        let (lease, disconnect) = audio.begin_request(control.clone(), 30, || Ok(())).unwrap();
        let worker = lease.clone();
        drop(disconnect);
        drop(lease);
        assert!(control.is_cancelled());
        assert!(audio.try_idle().is_err());
        drop(worker);
        assert!(audio.try_idle().is_ok());
        audio.drain();
        assert!(audio
            .begin_request(Arc::new(Default::default()), 30, || Ok(()))
            .is_err());
    }
    #[tokio::test]
    async fn native_cancel_does_not_cancel_another_audio_adapter() {
        let state = crate::tests::empty_state();
        let control = Arc::new(InferenceControl::default());
        let (_lease, _disconnect) = state
            .audio
            .begin_request(control.clone(), 30, || Ok(()))
            .unwrap();
        let result = cancel(State(state.clone())).await;
        assert_eq!(result.0["cancel_requested"], false);
        assert!(!control.is_cancelled());
        assert_eq!(
            unload(State(state.clone())).await.status(),
            StatusCode::CONFLICT
        );
        state.audio.drain();
        assert!(control.is_cancelled());
    }
    #[tokio::test]
    async fn routes_refuse_unconfigured_models_and_bad_references() {
        use tower::ServiceExt;
        let mut state = crate::tests::empty_state();
        let app = routes().with_state(state.clone());
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/xrt/audio/native/speech")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"input":"Test."}"#))
            .unwrap();
        assert_eq!(
            app.oneshot(req).await.unwrap().status(),
            StatusCode::PRECONDITION_REQUIRED
        );
        state.native_audio = Arc::new(NativeState {
            config: Some(config()),
            ..Default::default()
        });
        let app = routes().with_state(state.clone());
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/xrt/audio/native/speech")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"input":"Test.","references":[{"transcript":"Owned.","wav_b64":"invalid"}]}"#,
            ))
            .unwrap();
        assert_eq!(
            app.oneshot(req).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        assert!(!state.native_audio.loaded.load(Ordering::Acquire));
        assert!(state.audio.try_idle().is_ok());
        assert_eq!(cancel(State(state)).await.0["cancel_requested"], false);
    }
}
