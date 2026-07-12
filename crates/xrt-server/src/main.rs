use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use clap::{ArgGroup, Parser};
use image::{imageops::FilterType, DynamicImage};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    io::{self, Read as _, Write as _},
    ops::ControlFlow,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    signal,
    sync::{mpsc, RwLock},
    task,
};
use tokio_stream::wrappers::ReceiverStream;
use xrt_hub::{resolve_model_alias_or_path, DownloadProgress, ModelHub};
use xrt_runtime::{
    BackendKind, GenerateRequest, GpuResourceManager, GpuResourceStatus, PromptSpan,
    PromptSpanKind, RequestScheduler, Runtime, SchedulerAcquireError, SchedulerConfig,
    SchedulerPermit, SchedulerStatus,
};
use xrt_tokenizer::{apply_chat_template, ChatMessage as TemplateChatMessage, CHATML_TEMPLATE};

#[derive(Parser)]
#[command(name = "xrt-server", about = "xeno-rt OpenAI-compatible server")]
#[command(group(
    ArgGroup::new("model_source")
        .args(["model", "hf_repo"])
))]
struct Cli {
    /// Path to a local GGUF model file
    #[arg(long, conflicts_with_all = ["hf_repo", "hf_file"])]
    model: Option<String>,
    /// Path to a local multimodal projection GGUF file
    #[arg(long)]
    mmproj: Option<String>,
    /// HuggingFace repo to download model from (e.g. "Qwen/Qwen3-0.6B-GGUF")
    #[arg(long, requires = "hf_file", conflicts_with = "model")]
    hf_repo: Option<String>,
    /// GGUF filename within the HuggingFace repo (e.g. "qwen3-0.6b-q4_k_m.gguf")
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 3000)]
    port: u16,
    #[arg(long, env = "XRT_BACKEND", default_value = "auto")]
    backend: String,
    #[arg(long, env = "XRT_MAX_ACTIVE_SEQUENCES", default_value_t = 1)]
    max_active_sequences: usize,
    #[arg(long, env = "XRT_MAX_QUEUED_SEQUENCES", default_value_t = 32)]
    max_queued_sequences: usize,
    #[arg(long, env = "XRT_STREAM_BUFFER_CAPACITY", default_value_t = 32)]
    stream_buffer_capacity: usize,
    #[arg(long, env = "XRT_PREFILL_CHUNK_TOKENS", default_value_t = 128)]
    prefill_chunk_tokens: usize,
    #[arg(long, env = "XRT_MAX_DECODE_TURNS_BEFORE_PREFILL", default_value_t = 8)]
    max_decode_turns_before_prefill: usize,
    #[arg(long, env = "XRT_MAX_DECODE_BATCH_SIZE", default_value_t = 4)]
    max_decode_batch_size: usize,
    #[arg(long, env = "XRT_DECODE_BATCH_WAIT_MICROS", default_value_t = 20_000)]
    decode_batch_wait_micros: u64,
}

#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<Option<Arc<Runtime>>>>,
    loaded_model_name: Arc<RwLock<Option<String>>>,
    loaded_model_path: Arc<RwLock<Option<String>>>,
    loaded_mmproj_path: Arc<RwLock<Option<String>>>,
    scheduler: Arc<RequestScheduler>,
}

// --- OpenAI-compatible request/response types ---

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    cache_policy: Option<String>,
    recent_window_tokens: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    stream: Option<bool>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatRequestMessage>,
    cache_policy: Option<String>,
    recent_window_tokens: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    stream: Option<bool>,
    seed: Option<u64>,
    /// Tool definitions for function calling.
    #[serde(default)]
    #[allow(dead_code)]
    tools: Option<Vec<serde_json::Value>>,
    /// Tool choice strategy.
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatRequestMessage {
    role: String,
    #[serde(default)]
    content: Option<ChatRequestContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ChatRequestContent {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    function: Option<ChatToolFunction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatToolFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct UsageInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: UsageInfo,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Serialize)]
struct CompletionChunkChoice {
    text: String,
    index: usize,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: UsageInfo,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatDelta {
    role: Option<&'static str>,
    content: Option<String>,
}

// --- /v1/models response types ---

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Debug, Deserialize)]
struct RuntimeLoadRequest {
    model_path: Option<String>,
    hf_repo: Option<String>,
    hf_file: Option<String>,
    mmproj_path: Option<String>,
    backend: Option<String>,
}

#[derive(Serialize)]
struct RuntimeStatusResponse {
    object: &'static str,
    ready: bool,
    kv_cache_mode: &'static str,
    requested_backend: String,
    active_backend: Option<String>,
    gpu_resource: GpuResourceStatus,
    scheduler: SchedulerStatus,
    loaded_model: Option<String>,
    loaded_model_path: Option<String>,
    loaded_mmproj_path: Option<String>,
}

#[derive(Serialize)]
struct RuntimeLoadResponse {
    success: bool,
    loaded_model: String,
    loaded_model_path: String,
    loaded_mmproj_path: Option<String>,
    requested_backend: String,
    active_backend: String,
    gpu_resource: GpuResourceStatus,
}

#[derive(Serialize)]
struct RuntimeUnloadResponse {
    success: bool,
}

#[derive(Debug, Clone)]
struct PreparedTemplateMessage {
    message: TemplateChatMessage,
    span_kind: PromptSpanKind,
}

#[derive(Debug, Clone)]
struct PreparedChatRequest {
    messages: Vec<ChatMessage>,
    images: Vec<Vec<f32>>,
}

// ─── Image-domain task handlers (xrt-vision) ────────────────────────────────

/// Request payload for `POST /v1/images/remove-background`.
///
/// Accepts the source image as either:
///   - `image_b64`: base64-encoded bytes (PNG / JPEG / WebP — anything the
///     `image` crate decodes), OR
///   - `image_url`: `http(s)://`, `file://`, or `data:` URL — fetched server-side.
///
/// Optional fields:
///   - `model_path`: override the BiRefNet ONNX file location. Defaults to
///     `~/.xeno/models/birefnet-general/model.onnx`.
///   - `use_gpu`: try CUDA first when the build supports it.
///
/// Response: JSON with `image_b64` (PNG bytes, base64-encoded) carrying the
/// alpha-cut result, plus `width` / `height` of the output for callers that
/// don't want to decode just to measure.
#[derive(Debug, Deserialize)]
struct RemoveBackgroundRequest {
    image_b64: Option<String>,
    image_url: Option<String>,
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default = "default_use_gpu")]
    use_gpu: bool,
}

fn default_use_gpu() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct RemoveBackgroundResponse {
    image_b64: String,
    width: u32,
    height: u32,
}

async fn remove_background(
    State(_state): State<AppState>,
    Json(req): Json<RemoveBackgroundRequest>,
) -> Result<Json<RemoveBackgroundResponse>, (StatusCode, String)> {
    // Resolve input bytes — either a base64 payload or a URL we fetch.
    let input_bytes: Vec<u8> = if let Some(b64) = req.image_b64.as_ref() {
        BASE64_STANDARD
            .decode(b64.as_bytes())
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid base64: {e}")))?
    } else if let Some(url) = req.image_url.as_ref() {
        load_image_bytes(url).map_err(|e| (e.0, e.1))?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "must provide either `image_b64` or `image_url`".to_string(),
        ));
    };

    // Build the inference options. Cloning into the spawn_blocking closure
    // keeps the handler independent of the Tokio executor — inference is
    // CPU- and GPU-bound and would otherwise stall the runtime for seconds.
    let mut opts = xrt_vision::background_removal::RemoveBackgroundOptions::default();
    if let Some(p) = req.model_path.as_ref() {
        opts.model_path = std::path::PathBuf::from(p);
    }
    opts.use_gpu = req.use_gpu;

    let result_bytes = task::spawn_blocking(move || {
        xrt_vision::background_removal::remove_background(&input_bytes, &opts)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panic: {e}"),
        )
    })?
    .map_err(|e| match e {
        xrt_vision::VisionError::ModelMissing { path, message } => (
            StatusCode::PRECONDITION_REQUIRED,
            format!("model file not found at {path}: {message}"),
        ),
        xrt_vision::VisionError::InvalidImage(msg) => (StatusCode::BAD_REQUEST, msg),
        xrt_vision::VisionError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        xrt_vision::VisionError::EncodeFailed(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
    })?;

    // Decode the produced PNG just to read dimensions back. Cheap relative
    // to the inference cost; saves callers from re-decoding to measure.
    let (out_w, out_h) = match image::load_from_memory(&result_bytes) {
        Ok(img) => (img.width(), img.height()),
        Err(_) => (0, 0),
    };

    Ok(Json(RemoveBackgroundResponse {
        image_b64: BASE64_STANDARD.encode(&result_bytes),
        width: out_w,
        height: out_h,
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let scheduler_config = SchedulerConfig::new(
        cli.max_active_sequences,
        cli.max_queued_sequences,
        cli.stream_buffer_capacity,
    )?
    .with_execution_policy(
        cli.prefill_chunk_tokens,
        cli.max_decode_turns_before_prefill,
    )?
    .with_decode_batching(cli.max_decode_batch_size, cli.decode_batch_wait_micros)?;
    let state = AppState {
        runtime: Arc::new(RwLock::new(None)),
        loaded_model_name: Arc::new(RwLock::new(None)),
        loaded_model_path: Arc::new(RwLock::new(None)),
        loaded_mmproj_path: Arc::new(RwLock::new(None)),
        scheduler: Arc::new(RequestScheduler::new(scheduler_config)),
    };

    if cli.model.is_some() || cli.hf_repo.is_some() {
        load_runtime_from_cli(&state, &cli).await?;
    }

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/runtime/status", get(runtime_status))
        .route("/v1/runtime/load", post(runtime_load))
        .route("/v1/runtime/unload", post(runtime_unload))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // Image-domain task endpoints, served from `xrt-vision`. These run
        // ONNX inference (BiRefNet et al.) inside `tokio::task::spawn_blocking`
        // so the async executor stays unblocked across the multi-second hits.
        .route("/v1/images/remove-background", post(remove_background))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cli.host, cli.port)).await?;
    tracing::info!("listening on {}", listener.local_addr()?);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = signal::ctrl_c().await;
        })
        .await?;

    Ok(())
}

async fn load_runtime_from_cli(
    state: &AppState,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = resolve_model_path(cli)?;
    let requested_backend = parse_backend_value(&cli.backend)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let runtime = if let Some(mmproj_path) = cli.mmproj.as_deref() {
        Runtime::load_with_backend(&model_path, requested_backend)?.load_vision(mmproj_path)?
    } else {
        Runtime::load_with_backend(&model_path, requested_backend)?
    };

    let model_name = runtime.model_name().to_string();
    state
        .scheduler
        .configure_kv_budget(runtime.gpu_resource_status().kv_budget_bytes);
    *state.runtime.write().await = Some(runtime);
    *state.loaded_model_name.write().await = Some(model_name);
    *state.loaded_model_path.write().await = Some(model_path);
    *state.loaded_mmproj_path.write().await = cli.mmproj.clone();
    Ok(())
}

/// Resolve model path from --model or --hf-repo/--hf-file flags,
/// downloading from HuggingFace if needed.
fn resolve_model_path(cli: &Cli) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(model) = &cli.model {
        return Ok(resolve_model_alias_or_path(model)
            .to_string_lossy()
            .to_string());
    }
    let repo = cli.hf_repo.as_deref().ok_or("missing --hf-repo")?;
    let file = cli.hf_file.as_deref().ok_or("missing --hf-file")?;
    let hub = ModelHub::new()?;
    let mut reporter = progress_reporter(repo, file);
    let model = hub.download_with_progress(repo, file, &mut reporter)?;
    finish_download(
        &model.repo_id,
        &model.filename,
        &model.path,
        model.size,
        model.was_cached,
    )?;
    Ok(model.path.to_string_lossy().to_string())
}

fn progress_reporter<'a>(repo: &'a str, target: &'a str) -> impl FnMut(DownloadProgress) + 'a {
    move |progress| {
        let percent = progress.percent().unwrap_or(0.0);
        eprint!(
            "\rDownloading {repo}/{target} {:>6.2}% ({}/{})",
            percent,
            format_bytes(progress.downloaded),
            format_bytes(progress.total)
        );
        let _ = io::stderr().flush();
    }
}

fn finish_download(
    repo: &str,
    file: &str,
    path: &std::path::Path,
    size: u64,
    was_cached: bool,
) -> io::Result<()> {
    if was_cached {
        eprintln!(
            "Using cached {repo}/{file} ({}) at {}",
            format_bytes(size),
            path.display()
        );
    } else {
        eprintln!(
            "\rDownloaded {repo}/{file} ({}) to {}",
            format_bytes(size),
            path.display()
        );
    }
    io::stderr().flush()
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for next in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

// --- Route handlers ---

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let model_name = state.loaded_model_name.read().await.clone();
    Json(ModelList {
        object: "list",
        data: model_name
            .into_iter()
            .map(|id| ModelInfo {
                id,
                object: "model",
                created: unix_timestamp(),
                owned_by: "xeno-rt",
            })
            .collect(),
    })
}

async fn runtime_status(State(state): State<AppState>) -> Json<RuntimeStatusResponse> {
    let runtime = state.runtime.read().await.clone();
    let loaded_model = state.loaded_model_name.read().await.clone();
    let loaded_model_path = state.loaded_model_path.read().await.clone();
    let loaded_mmproj_path = state.loaded_mmproj_path.read().await.clone();
    let requested_backend = runtime
        .as_ref()
        .map(|runtime| runtime.requested_backend().as_str().to_string())
        .unwrap_or_else(|| BackendKind::from_env().as_str().to_string());
    let active_backend = runtime
        .as_ref()
        .map(|runtime| runtime.active_backend().as_str().to_string());
    let gpu_resource = runtime
        .as_ref()
        .map(|runtime| runtime.gpu_resource_status())
        .unwrap_or_else(|| GpuResourceManager::from_env().status());
    Json(RuntimeStatusResponse {
        object: "runtime.status",
        ready: runtime.is_some(),
        kv_cache_mode: xrt_runtime::KvCacheMode::from_env().as_str(),
        requested_backend,
        active_backend,
        gpu_resource,
        scheduler: state.scheduler.status(),
        loaded_model,
        loaded_model_path,
        loaded_mmproj_path,
    })
}

async fn runtime_load(
    State(state): State<AppState>,
    Json(request): Json<RuntimeLoadRequest>,
) -> Result<Response, (StatusCode, String)> {
    let requested_backend = request
        .backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(parse_backend_value)
        .transpose()
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?
        .unwrap_or_else(BackendKind::from_env);
    let mmproj_path = request
        .mmproj_path
        .clone()
        .filter(|value| !value.trim().is_empty());
    let model_path = if let Some(path) = request
        .model_path
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        resolve_model_alias_or_path(&path)
            .to_string_lossy()
            .to_string()
    } else if let (Some(repo), Some(file)) = (request.hf_repo.clone(), request.hf_file.clone()) {
        task::spawn_blocking(move || {
            let cli = Cli {
                model: None,
                mmproj: None,
                hf_repo: Some(repo),
                hf_file: Some(file),
                host: "127.0.0.1".to_string(),
                port: 0,
                backend: "auto".to_string(),
                max_active_sequences: 1,
                max_queued_sequences: 32,
                stream_buffer_capacity: 32,
                prefill_chunk_tokens: 128,
                max_decode_turns_before_prefill: 8,
                max_decode_batch_size: 4,
                decode_batch_wait_micros: 20_000,
            };
            resolve_model_path(&cli).map_err(|err| err.to_string())
        })
        .await
        .map_err(internal_error)?
        .map_err(internal_error)?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Provide either model_path or hf_repo + hf_file.".to_string(),
        ));
    };

    let runtime = task::spawn_blocking({
        let model_path = model_path.clone();
        let mmproj_path = mmproj_path.clone();
        move || {
            let runtime = Runtime::load_with_backend(&model_path, requested_backend)?;
            if let Some(mmproj_path) = mmproj_path {
                runtime.load_vision(&mmproj_path)
            } else {
                Ok(runtime)
            }
        }
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;

    let loaded_model = runtime.model_name().to_string();
    let requested_backend = runtime.requested_backend().as_str().to_string();
    let active_backend = runtime.active_backend().as_str().to_string();
    let gpu_resource = runtime.gpu_resource_status();
    state
        .scheduler
        .configure_kv_budget(gpu_resource.kv_budget_bytes);
    *state.runtime.write().await = Some(runtime);
    *state.loaded_model_name.write().await = Some(loaded_model.clone());
    *state.loaded_model_path.write().await = Some(model_path.clone());
    *state.loaded_mmproj_path.write().await = mmproj_path.clone();

    Ok(Json(RuntimeLoadResponse {
        success: true,
        loaded_model,
        loaded_model_path: model_path,
        loaded_mmproj_path: mmproj_path,
        requested_backend,
        active_backend,
        gpu_resource,
    })
    .into_response())
}

fn parse_backend_value(value: &str) -> Result<BackendKind, String> {
    BackendKind::parse(value).ok_or_else(|| format!("unsupported backend value: {value}"))
}

async fn runtime_unload(State(state): State<AppState>) -> Json<RuntimeUnloadResponse> {
    *state.runtime.write().await = None;
    state.scheduler.configure_kv_budget(None);
    *state.loaded_model_name.write().await = None;
    *state.loaded_model_path.write().await = None;
    *state.loaded_mmproj_path.write().await = None;
    Json(RuntimeUnloadResponse { success: true })
}

async fn loaded_runtime(state: &AppState) -> Result<Arc<Runtime>, (StatusCode, String)> {
    state.runtime.read().await.clone().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "xeno-rt is running but no model is loaded.".to_string(),
    ))
}

async fn acquire_inference_permit(
    state: &AppState,
) -> Result<SchedulerPermit, (StatusCode, String)> {
    state.scheduler.acquire().await.map_err(|err| match err {
        SchedulerAcquireError::QueueFull => (StatusCode::TOO_MANY_REQUESTS, err.to_string()),
        SchedulerAcquireError::Closed => (StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
        SchedulerAcquireError::KvBudgetExceeded { .. } => {
            (StatusCode::TOO_MANY_REQUESTS, err.to_string())
        }
    })
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    if request.stream.unwrap_or(false) {
        completion_stream(state, request).await
    } else {
        completion_once(state, request).await
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    if request.stream.unwrap_or(false) {
        chat_stream(state, request).await
    } else {
        chat_once(state, request).await
    }
}

async fn completion_once(
    state: AppState,
    request: CompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let runtime = loaded_runtime(&state).await?;
    let prompt_text = request.prompt.clone();
    let generate = request_to_generate_request(request.prompt.clone(), &request, true);

    // Count prompt tokens for usage info
    let prompt_tokens = runtime
        .tokenizer()
        .encode_with_options(&prompt_text, true, true)
        .map(|t| t.len())
        .unwrap_or(0);

    let permit = acquire_inference_permit(&state).await?;
    let generate_runtime = runtime.clone();
    let scheduler = state.scheduler.clone();
    let text = task::spawn_blocking(move || {
        let _permit = permit;
        let mut session = generate_runtime.new_session();
        session.generate_scheduled(&generate, &scheduler)
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;

    let completion_tokens = runtime
        .tokenizer()
        .encode(&text)
        .map(|t| t.len())
        .unwrap_or(0);

    let created = unix_timestamp();
    let response = CompletionResponse {
        id: completion_id("cmpl"),
        object: "text_completion",
        created,
        model: request
            .model
            .unwrap_or_else(|| runtime.model_name().to_string()),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop",
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    Ok(Json(response).into_response())
}

async fn chat_once(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let runtime = loaded_runtime(&state).await?;
    let prepared_chat = prepare_chat_request(&request.messages, &runtime)?;
    let (prompt, prompt_spans) =
        chat_prompt_with_spans(&prepared_chat.messages, request.tools.as_deref(), &runtime);
    let mut generate = request_to_generate_request(prompt.clone(), &request, false);
    generate.prompt_spans = prompt_spans;
    generate.images = prepared_chat.images;
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && request.temperature.is_none()
    {
        generate.temperature = 0.2;
    }

    let prompt_tokens = runtime
        .tokenizer()
        .encode_with_options(&prompt, false, true)
        .map(|t| t.len())
        .unwrap_or(0);

    let permit = acquire_inference_permit(&state).await?;
    let generate_runtime = runtime.clone();
    let scheduler = state.scheduler.clone();
    let text = task::spawn_blocking(move || {
        let _permit = permit;
        let mut session = generate_runtime.new_session();
        session.generate_scheduled(&generate, &scheduler)
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;

    let completion_tokens = runtime
        .tokenizer()
        .encode(&text)
        .map(|t| t.len())
        .unwrap_or(0);

    let sanitized_text = sanitize_assistant_text(&text);
    let (response_text, response_tool_calls, finish_reason) =
        if let Some(tool_calls) = extract_tool_calls_from_text(&sanitized_text) {
            (String::new(), Some(tool_calls), "tool_calls")
        } else {
            (sanitized_text, None, "stop")
        };

    let created = unix_timestamp();
    let response = ChatCompletionResponse {
        id: completion_id("chatcmpl"),
        object: "chat.completion",
        created,
        model: request
            .model
            .unwrap_or_else(|| runtime.model_name().to_string()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
                tool_call_id: None,
                tool_calls: response_tool_calls,
            },
            finish_reason,
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    Ok(Json(response).into_response())
}

async fn completion_stream(
    state: AppState,
    request: CompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let runtime = loaded_runtime(&state).await?;
    let (tx, rx) =
        mpsc::channel::<Result<Event, Infallible>>(state.scheduler.config().stream_buffer_capacity);
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| runtime.model_name().to_string());
    let generate = request_to_generate_request(request.prompt.clone(), &request, true);
    let id = completion_id("cmpl");
    let created = unix_timestamp();
    let permit = acquire_inference_permit(&state).await?;
    let scheduler = state.scheduler.clone();

    task::spawn_blocking(move || {
        let _permit = permit;
        let mut session = runtime.new_session();
        let result =
            session.generate_stream_scheduled_with_control(&generate, &scheduler, |piece| {
                let chunk = CompletionChunk {
                    id: id.clone(),
                    object: "text_completion.chunk",
                    created,
                    model: model_name.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: piece.to_string(),
                        index: 0,
                        finish_reason: None,
                    }],
                };
                if let Ok(data) = serde_json::to_string(&chunk) {
                    if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            });

        if tx.is_closed() {
            return;
        }
        let finish = CompletionChunk {
            id,
            object: "text_completion.chunk",
            created,
            model: model_name,
            choices: vec![CompletionChunkChoice {
                text: String::new(),
                index: 0,
                finish_reason: Some(if result.is_ok() { "stop" } else { "error" }),
            }],
        };
        if let Ok(data) = serde_json::to_string(&finish) {
            let _ = tx.blocking_send(Ok(Event::default().data(data)));
        }
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn chat_stream(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let runtime = loaded_runtime(&state).await?;
    let (tx, rx) =
        mpsc::channel::<Result<Event, Infallible>>(state.scheduler.config().stream_buffer_capacity);
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| runtime.model_name().to_string());
    let prepared_chat = prepare_chat_request(&request.messages, &runtime)?;
    let (prompt, prompt_spans) =
        chat_prompt_with_spans(&prepared_chat.messages, request.tools.as_deref(), &runtime);
    let mut generate = request_to_generate_request(prompt, &request, false);
    generate.prompt_spans = prompt_spans;
    generate.images = prepared_chat.images;
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && request.temperature.is_none()
    {
        generate.temperature = 0.2;
    }
    let id = completion_id("chatcmpl");
    let created = unix_timestamp();
    let permit = acquire_inference_permit(&state).await?;
    let scheduler = state.scheduler.clone();

    task::spawn_blocking(move || {
        let _permit = permit;
        let bootstrap = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(data) = serde_json::to_string(&bootstrap) {
            if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                return;
            }
        }

        let mut session = runtime.new_session();
        let result =
            session.generate_stream_scheduled_with_control(&generate, &scheduler, |piece| {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_name.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(piece.to_string()),
                        },
                        finish_reason: None,
                    }],
                };
                if let Ok(data) = serde_json::to_string(&chunk) {
                    if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            });

        if tx.is_closed() {
            return;
        }
        let finish = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model: model_name,
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(if result.is_ok() { "stop" } else { "error" }),
            }],
        };
        if let Ok(data) = serde_json::to_string(&finish) {
            let _ = tx.blocking_send(Ok(Event::default().data(data)));
        }
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn prepare_chat_request(
    messages: &[ChatRequestMessage],
    runtime: &Runtime,
) -> Result<PreparedChatRequest, (StatusCode, String)> {
    let mut prepared_messages = Vec::with_capacity(messages.len());
    let mut images = Vec::new();

    for message in messages {
        let (content, mut message_images) = render_chat_request_content(&message.content, runtime)?;
        images.append(&mut message_images);
        prepared_messages.push(ChatMessage {
            role: message.role.clone(),
            content,
            tool_call_id: message.tool_call_id.clone(),
            tool_calls: message.tool_calls.clone(),
        });
    }

    Ok(PreparedChatRequest {
        messages: prepared_messages,
        images,
    })
}

fn render_chat_request_content(
    content: &Option<ChatRequestContent>,
    runtime: &Runtime,
) -> Result<(String, Vec<Vec<f32>>), (StatusCode, String)> {
    let Some(content) = content else {
        return Ok((String::new(), Vec::new()));
    };

    match content {
        ChatRequestContent::Text(text) => Ok((text.clone(), Vec::new())),
        ChatRequestContent::Parts(parts) => {
            let mut rendered = String::new();
            let mut images = Vec::new();
            let layout = parts
                .iter()
                .any(|part| part_kind(part) == Some("image_url"))
                .then(|| {
                    runtime.vision_prompt_layout().ok_or((
                        StatusCode::BAD_REQUEST,
                        "image inputs require a loaded mmproj-compatible vision model".to_string(),
                    ))
                });

            let layout = match layout {
                Some(result) => Some(result?),
                None => None,
            };

            for part in parts {
                match part_kind(part) {
                    Some("text") => rendered.push_str(extract_text_part(part)?),
                    Some("image_url") => {
                        let layout = layout.as_ref().expect("image layout checked above");
                        let image_url = extract_image_url(part)?;
                        rendered.push_str(&layout.prompt_fragment());
                        images.push(load_and_preprocess_image(
                            image_url,
                            runtime
                                .vision()
                                .expect("vision availability checked above")
                                .config()
                                .image_size,
                        )?);
                    }
                    Some("input_audio") => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            "audio chat parts are not supported yet".to_string(),
                        ));
                    }
                    Some("video_url") | Some("video") => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            "video chat parts are not supported yet".to_string(),
                        ));
                    }
                    Some(other) => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            format!("unsupported chat content part type: {other}"),
                        ));
                    }
                    None => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            "chat content parts must include a type".to_string(),
                        ));
                    }
                }
            }

            Ok((rendered, images))
        }
    }
}

fn part_kind(part: &serde_json::Value) -> Option<&str> {
    part.get("type").and_then(serde_json::Value::as_str)
}

fn extract_text_part(part: &serde_json::Value) -> Result<&str, (StatusCode, String)> {
    part.get("text").and_then(serde_json::Value::as_str).ok_or((
        StatusCode::BAD_REQUEST,
        "text chat parts must include a string text field".to_string(),
    ))
}

fn extract_image_url(part: &serde_json::Value) -> Result<&str, (StatusCode, String)> {
    match part.get("image_url") {
        Some(serde_json::Value::String(url)) => Ok(url.as_str()),
        Some(serde_json::Value::Object(image_url)) => image_url
            .get("url")
            .and_then(serde_json::Value::as_str)
            .ok_or((
                StatusCode::BAD_REQUEST,
                "image_url chat parts must include image_url.url".to_string(),
            )),
        _ => Err((
            StatusCode::BAD_REQUEST,
            "image_url chat parts must include an image_url object or string".to_string(),
        )),
    }
}

fn load_and_preprocess_image(
    image_ref: &str,
    image_size: usize,
) -> Result<Vec<f32>, (StatusCode, String)> {
    let bytes = load_image_bytes(image_ref)?;
    let image = image::load_from_memory(&bytes).map_err(bad_request)?;
    preprocess_image(image, image_size).map_err(bad_request)
}

fn load_image_bytes(image_ref: &str) -> Result<Vec<u8>, (StatusCode, String)> {
    if let Some((_, payload)) = image_ref.split_once(",") {
        if image_ref.starts_with("data:") {
            return BASE64_STANDARD
                .decode(payload)
                .map_err(|err| bad_request(format!("invalid base64 image payload: {err}")));
        }
    }

    if image_ref.starts_with("http://") || image_ref.starts_with("https://") {
        let response = ureq::get(image_ref)
            .call()
            .map_err(|err| bad_request(format!("failed to fetch image URL: {err}")))?;
        let mut reader = response.into_reader();
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|err| bad_request(format!("failed to read image response: {err}")))?;
        return Ok(bytes);
    }

    let path = image_ref.strip_prefix("file://").unwrap_or(image_ref);
    std::fs::read(path).map_err(|err| bad_request(format!("failed to read image file: {err}")))
}

fn preprocess_image(image: DynamicImage, image_size: usize) -> Result<Vec<f32>, String> {
    let pixels = image_tensor_pixels(image_size)?;
    let output_len = pixels
        .checked_mul(3)
        .ok_or_else(|| "image tensor length overflow".to_string())?;
    let target = image_size as u32;
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();

    let scale = target as f32 / width.min(height) as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(target);
    let resized_height = ((height as f32 * scale).round() as u32).max(target);
    let resized =
        image::imageops::resize(&rgb, resized_width, resized_height, FilterType::CatmullRom);

    let crop_x = (resized_width - target) / 2;
    let crop_y = (resized_height - target) / 2;
    let cropped = image::imageops::crop_imm(&resized, crop_x, crop_y, target, target).to_image();

    let mut output = vec![0.0f32; output_len];
    for y in 0..target {
        for x in 0..target {
            let pixel = cropped.get_pixel(x, y);
            let index = (y * target + x) as usize;
            output[index] = pixel[0] as f32 / 127.5 - 1.0;
            output[pixels + index] = pixel[1] as f32 / 127.5 - 1.0;
            output[2 * pixels + index] = pixel[2] as f32 / 127.5 - 1.0;
        }
    }
    Ok(output)
}

fn image_tensor_pixels(image_size: usize) -> Result<usize, String> {
    let target = u32::try_from(image_size)
        .map_err(|_| format!("image size {image_size} exceeds u32::MAX"))?;
    if target == 0 {
        return Err("image size must be greater than zero".to_string());
    }
    image_size
        .checked_mul(image_size)
        .ok_or_else(|| "image pixel count overflow".to_string())
}

fn request_to_generate_request<T>(
    prompt: String,
    request: &T,
    add_special_tokens: bool,
) -> GenerateRequest
where
    T: RequestConfig,
{
    GenerateRequest {
        prompt,
        add_special_tokens,
        cache_policy: request.cache_policy().map(ToOwned::to_owned),
        recent_window_tokens: request.recent_window_tokens(),
        max_tokens: request.max_tokens().unwrap_or(128),
        temperature: request.temperature().unwrap_or(0.8),
        top_k: request.top_k().unwrap_or(40),
        top_p: request.top_p().unwrap_or(0.95),
        repetition_penalty: request.repetition_penalty().unwrap_or(1.1),
        seed: request.seed(),
        ..Default::default()
    }
}

trait RequestConfig {
    fn cache_policy(&self) -> Option<&str>;
    fn recent_window_tokens(&self) -> Option<usize>;
    fn max_tokens(&self) -> Option<usize>;
    fn temperature(&self) -> Option<f32>;
    fn top_k(&self) -> Option<usize>;
    fn top_p(&self) -> Option<f32>;
    fn repetition_penalty(&self) -> Option<f32>;
    fn seed(&self) -> Option<u64>;
}

impl RequestConfig for CompletionRequest {
    fn cache_policy(&self) -> Option<&str> {
        self.cache_policy.as_deref()
    }
    fn recent_window_tokens(&self) -> Option<usize> {
        self.recent_window_tokens
    }
    fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }
    fn top_k(&self) -> Option<usize> {
        self.top_k
    }
    fn top_p(&self) -> Option<f32> {
        self.top_p
    }
    fn repetition_penalty(&self) -> Option<f32> {
        self.repetition_penalty
    }
    fn seed(&self) -> Option<u64> {
        self.seed
    }
}

impl RequestConfig for ChatCompletionRequest {
    fn cache_policy(&self) -> Option<&str> {
        self.cache_policy.as_deref()
    }
    fn recent_window_tokens(&self) -> Option<usize> {
        self.recent_window_tokens
    }
    fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }
    fn top_k(&self) -> Option<usize> {
        self.top_k
    }
    fn top_p(&self) -> Option<f32> {
        self.top_p
    }
    fn repetition_penalty(&self) -> Option<f32> {
        self.repetition_penalty
    }
    fn seed(&self) -> Option<u64> {
        self.seed
    }
}

fn chat_prompt_with_spans(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    runtime: &Runtime,
) -> (String, Vec<PromptSpan>) {
    let prepared = prepare_template_messages(messages, tools, runtime);
    let prompt = render_prepared_messages(runtime, &prepared, true);
    let tokenizer = runtime.tokenizer();
    let mut spans = Vec::new();
    let mut previous_end = 0usize;

    for end_index in 0..prepared.len() {
        let prefix_prompt = render_prepared_messages(runtime, &prepared[..=end_index], false);
        let prefix_end = tokenizer
            .encode_with_options(&prefix_prompt, false, true)
            .map(|tokens| tokens.len())
            .unwrap_or(previous_end);
        if prefix_end > previous_end {
            spans.push(PromptSpan {
                kind: prepared[end_index].span_kind,
                token_start: previous_end,
                token_end: prefix_end,
            });
            previous_end = prefix_end;
        }
    }

    (prompt, spans)
}

fn render_prepared_messages(
    runtime: &Runtime,
    prepared: &[PreparedTemplateMessage],
    add_generation_prompt: bool,
) -> String {
    let template_messages = prepared
        .iter()
        .map(|message| message.message.clone())
        .collect::<Vec<_>>();

    match format_runtime_chat(runtime, &template_messages, add_generation_prompt) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!("chat template render failed, using fallback: {e}");
            let mut prompt = String::new();
            for message in prepared {
                prompt.push_str(&message.message.role.to_uppercase());
                prompt.push_str(": ");
                prompt.push_str(&message.message.content);
                prompt.push('\n');
            }
            if add_generation_prompt {
                prompt.push_str("ASSISTANT: ");
            }
            prompt
        }
    }
}

fn format_runtime_chat(
    runtime: &Runtime,
    messages: &[TemplateChatMessage],
    add_generation_prompt: bool,
) -> Result<String, String> {
    let has_system = messages.iter().any(|message| message.role == "system");
    if runtime.model_architecture() == "qwen35" && has_system {
        let tokenizer = runtime.tokenizer();
        let special = tokenizer.special_tokens();
        let bos = special
            .bos
            .and_then(|id| tokenizer.token_to_piece(id))
            .unwrap_or("");
        let eos = special
            .eos
            .and_then(|id| tokenizer.token_to_piece(id))
            .unwrap_or("");
        return apply_chat_template(CHATML_TEMPLATE, messages, bos, eos, add_generation_prompt)
            .map_err(|e| e.to_string());
    }

    runtime
        .tokenizer()
        .format_chat(messages, add_generation_prompt)
        .map_err(|e| e.to_string())
}

fn prepare_template_messages(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    runtime: &Runtime,
) -> Vec<PreparedTemplateMessage> {
    if runtime.model_architecture() == "qwen35" {
        return prepare_qwen35_template_messages(messages, tools);
    }

    let tool_block = tools.and_then(build_tool_instruction_block);
    let runtime_block = build_runtime_behavior_block();
    let mut prepared = Vec::with_capacity(messages.len() + usize::from(tool_block.is_some()) + 1);

    prepared.push(PreparedTemplateMessage {
        message: TemplateChatMessage {
            role: "system".to_string(),
            content: runtime_block.to_string(),
        },
        span_kind: PromptSpanKind::Developer,
    });

    if let Some(block) = tool_block {
        prepared.push(PreparedTemplateMessage {
            message: TemplateChatMessage {
                role: "system".to_string(),
                content: block,
            },
            span_kind: PromptSpanKind::ToolSchema,
        });
    }

    for message in messages {
        match message.role.as_str() {
            "system" => prepared.push(PreparedTemplateMessage {
                message: TemplateChatMessage {
                    role: "system".to_string(),
                    content: message.content.clone(),
                },
                span_kind: PromptSpanKind::System,
            }),
            "assistant" => {
                let mut content = message.content.clone();
                if let Some(tool_calls) = &message.tool_calls {
                    let rendered = render_assistant_tool_calls(tool_calls);
                    if !rendered.is_empty() {
                        if !content.trim().is_empty() {
                            content.push_str("\n\n");
                        }
                        content.push_str(&rendered);
                    }
                }
                prepared.push(PreparedTemplateMessage {
                    message: TemplateChatMessage {
                        role: "assistant".to_string(),
                        content,
                    },
                    span_kind: PromptSpanKind::Assistant,
                });
            }
            "tool" => prepared.push(PreparedTemplateMessage {
                message: TemplateChatMessage {
                    role: "user".to_string(),
                    content: render_tool_result_message(message),
                },
                span_kind: PromptSpanKind::ToolResult,
            }),
            role => prepared.push(PreparedTemplateMessage {
                message: TemplateChatMessage {
                    role: role.to_string(),
                    content: message.content.clone(),
                },
                span_kind: PromptSpanKind::User,
            }),
        }
    }

    prepared
}

fn prepare_qwen35_template_messages(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
) -> Vec<PreparedTemplateMessage> {
    let tool_block = tools.and_then(build_tool_instruction_block);
    let mut system_sections = Vec::new();
    if let Some(block) = tool_block {
        system_sections.push(block);
    }
    for message in messages {
        if message.role == "system" && !message.content.trim().is_empty() {
            system_sections.push(message.content.clone());
        }
    }

    let mut prepared = Vec::with_capacity(messages.len() + 1);
    if !system_sections.is_empty() {
        prepared.push(PreparedTemplateMessage {
            message: TemplateChatMessage {
                role: "system".to_string(),
                content: system_sections.join("\n\n"),
            },
            span_kind: PromptSpanKind::Developer,
        });
    }

    for message in messages {
        match message.role.as_str() {
            "system" => {}
            "assistant" => {
                let mut content = message.content.clone();
                if let Some(tool_calls) = &message.tool_calls {
                    let rendered = render_assistant_tool_calls(tool_calls);
                    if !rendered.is_empty() {
                        if !content.trim().is_empty() {
                            content.push_str("\n\n");
                        }
                        content.push_str(&rendered);
                    }
                }
                prepared.push(PreparedTemplateMessage {
                    message: TemplateChatMessage {
                        role: "assistant".to_string(),
                        content,
                    },
                    span_kind: PromptSpanKind::Assistant,
                });
            }
            "tool" => prepared.push(PreparedTemplateMessage {
                message: TemplateChatMessage {
                    role: "user".to_string(),
                    content: render_tool_result_message(message),
                },
                span_kind: PromptSpanKind::ToolResult,
            }),
            role => prepared.push(PreparedTemplateMessage {
                message: TemplateChatMessage {
                    role: role.to_string(),
                    content: message.content.clone(),
                },
                span_kind: PromptSpanKind::User,
            }),
        }
    }

    prepared
}

fn build_runtime_behavior_block() -> &'static str {
    concat!(
        "Respond in plain text only unless the user explicitly asks for structured output.\n",
        "Do not emit hidden reasoning, <think> tags, XML tags, role labels, template markers, or markdown fences unless asked.\n",
        "If you greet the user, keep it brief and natural.\n"
    )
}

fn build_tool_instruction_block(tools: &[serde_json::Value]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }

    let rendered_tools = tools
        .iter()
        .map(|tool| serde_json::to_string_pretty(tool).unwrap_or_else(|_| tool.to_string()))
        .collect::<Vec<_>>()
        .join("\n\n");

    Some(format!(
        concat!(
            "Tool calling is enabled.\n",
            "If you need a tool, respond with only valid JSON and no markdown.\n",
            "Use an exact function name from the available tools list.\n",
            "The arguments object must use the exact property names from the tool schema.\n",
            "Never output placeholder names like TOOL_NAME.\n",
            "Preferred shape:\n",
            "{{\"tool_calls\":[{{\"id\":\"call_1\",\"type\":\"function\",\"function\":{{\"name\":\"Write\",\"arguments\":{{\"file_path\":\"example.txt\",\"content\":\"hello\"}}}}}}]}}\n",
            "Single-call shorthand also allowed, for example:\n",
            "{{\"name\":\"Write\",\"arguments\":{{\"file_path\":\"example.txt\",\"content\":\"hello\"}}}}\n",
            "Do not add any explanatory text before or after the JSON tool call.\n",
            "If no tool is needed, answer normally.\n\n",
            "Available tools:\n{}\n"
        ),
        rendered_tools
    ))
}

fn render_assistant_tool_calls(tool_calls: &[ChatToolCall]) -> String {
    if tool_calls.is_empty() {
        return String::new();
    }

    let payload = serde_json::json!({
        "tool_calls": tool_calls,
    });
    format!(
        "[assistant_tool_calls]\n{}\n[/assistant_tool_calls]",
        payload
    )
}

fn render_tool_result_message(message: &ChatMessage) -> String {
    let mut content = String::from("[tool_result");
    if let Some(tool_call_id) = &message.tool_call_id {
        content.push(':');
        content.push_str(tool_call_id);
    }
    content.push_str("]\n");
    content.push_str(&message.content);
    content
}

fn extract_tool_calls_from_text(text: &str) -> Option<Vec<ChatToolCall>> {
    for candidate in json_tool_call_candidates(text) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) else {
            continue;
        };
        if let Some(tool_calls) = parse_tool_calls_value(&value) {
            return Some(tool_calls);
        }
    }
    None
}

fn json_tool_call_candidates(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut candidates = vec![trimmed.to_string()];

    let mut remaining = trimmed;
    while let Some(start) = remaining.find("```") {
        let after_tick = &remaining[start + 3..];
        let Some(end) = after_tick.find("```") else {
            break;
        };
        let block = after_tick[..end]
            .trim()
            .trim_start_matches("json")
            .trim()
            .to_string();
        if !block.is_empty() {
            candidates.push(block);
        }
        remaining = &after_tick[end + 3..];
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if end > start {
            candidates.push(trimmed[start..=end].to_string());
        }
    }

    candidates
}

fn parse_tool_calls_value(value: &serde_json::Value) -> Option<Vec<ChatToolCall>> {
    if let Some(tool_calls) = value.get("tool_calls").and_then(|v| v.as_array()) {
        let parsed = tool_calls
            .iter()
            .enumerate()
            .filter_map(|(index, item)| value_to_tool_call(item, index))
            .collect::<Vec<_>>();
        return (!parsed.is_empty()).then_some(parsed);
    }

    value_to_tool_call(value, 0).map(|tool_call| vec![tool_call])
}

fn value_to_tool_call(value: &serde_json::Value, index: usize) -> Option<ChatToolCall> {
    let function_value = value.get("function").unwrap_or(value);
    let name = function_value.get("name")?.as_str()?.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("TOOL_NAME") {
        return None;
    }

    let arguments_value = function_value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let arguments = if let Some(arguments) = arguments_value.as_str() {
        arguments.to_string()
    } else {
        serde_json::to_string(&arguments_value).ok()?
    };

    Some(ChatToolCall {
        id: value
            .get("id")
            .and_then(|v| v.as_str())
            .map(|id| id.to_string())
            .or_else(|| Some(format!("call_{}", index + 1))),
        kind: Some("function".to_string()),
        function: Some(ChatToolFunction {
            name: name.to_string(),
            arguments,
        }),
    })
}

fn sanitize_assistant_text(text: &str) -> String {
    let mut cleaned = text.trim().to_string();

    loop {
        let trimmed = cleaned.trim_start();
        let next = if let Some(rest) = trimmed.strip_prefix("</think>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("<think>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("<text>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("</text>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("<|im_start|>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("<|im_end|>") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("Assistantassistant") {
            rest.trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("assistant") {
            rest.trim_start_matches(':').trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("system") {
            rest.trim_start_matches(':').trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("user") {
            rest.trim_start_matches(':').trim_start().to_string()
        } else if let Some(rest) = trimmed.strip_prefix("model") {
            rest.trim_start_matches(':').trim_start().to_string()
        } else {
            break;
        };
        if next == cleaned {
            break;
        }
        cleaned = next;
    }

    if cleaned.starts_with("```") && cleaned.ends_with("```") {
        let inner = cleaned
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim_start_matches("text")
            .trim();
        if !inner.is_empty() {
            cleaned = inner.to_string();
        }
    }

    cleaned = cleaned
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<text>", "")
        .replace("</text>", "");

    cleaned.trim().to_string()
}

fn completion_id(prefix: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}-{millis}")
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn bad_request(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, err.to_string())
}

fn internal_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        extract_image_url, extract_text_part, image_tensor_pixels, load_image_bytes, part_kind,
        preprocess_image,
    };
    use image::{DynamicImage, RgbImage};

    #[test]
    fn multipart_request_parts_parse_expected_fields() {
        let part = serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/cat.png"
            }
        });
        assert_eq!(part_kind(&part), Some("image_url"));
        assert_eq!(
            extract_image_url(&part).expect("image_url should parse"),
            "https://example.com/cat.png"
        );

        let text = serde_json::json!({
            "type": "text",
            "text": "describe this"
        });
        assert_eq!(
            extract_text_part(&text).expect("text part should parse"),
            "describe this"
        );
    }

    #[test]
    fn data_url_images_decode() {
        let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+i1n8AAAAASUVORK5CYII=";
        let bytes = load_image_bytes(png).expect("data url should decode");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn preprocess_image_outputs_chw_normalized_tensor() {
        let image =
            DynamicImage::ImageRgb8(RgbImage::from_fn(2, 4, |_, _| image::Rgb([255, 0, 127])));
        let tensor = preprocess_image(image, 4).expect("image should preprocess");
        assert_eq!(tensor.len(), 3 * 4 * 4);
        assert!(tensor[0] <= 1.0 && tensor[0] >= -1.0);
        assert!(tensor[16] <= 1.0 && tensor[16] >= -1.0);
        assert!(tensor[32] <= 1.0 && tensor[32] >= -1.0);
    }

    #[test]
    fn image_tensor_pixels_rejects_bad_sizes() {
        assert_eq!(image_tensor_pixels(4).unwrap(), 16);
        assert!(image_tensor_pixels(0).is_err());
        assert!(image_tensor_pixels(usize::MAX).is_err());
    }
}
