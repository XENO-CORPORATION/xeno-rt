mod external_openai;
#[cfg(feature = "image-generation")]
mod image_api;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
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
    BackendKind, GenerateRequest, GpuResourceManager, GpuResourceStatus, HybridRuntimeStatus,
    MoeRuntimeConfig, MoeRuntimeStatus, PrefixCacheManager, PrefixCacheStatus, PromptSpan,
    PromptSpanKind, RequestScheduler, Runtime, SchedulerAcquireError, SchedulerConfig,
    SchedulerPermit, SchedulerStatus,
};
use xrt_tokenizer::{apply_chat_template, ChatMessage as TemplateChatMessage, CHATML_TEMPLATE};

use external_openai::{ExternalOpenAiClient, ExternalOpenAiConfig};

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
    /// Base URL for an external OpenAI-compatible runtime, including `/v1`
    #[arg(long, env = "XRT_EXTERNAL_BASE_URL")]
    external_base_url: Option<String>,
    /// Optional bearer token for the external OpenAI-compatible runtime
    #[arg(long, env = "XRT_EXTERNAL_API_KEY")]
    external_api_key: Option<String>,
    /// Default model inserted when a proxied request omits `model`
    #[arg(long, env = "XRT_EXTERNAL_MODEL")]
    external_model: Option<String>,
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
    /// Enable Qwen NextN / MTP speculative decoding
    #[arg(long, env = "XRT_QWEN_MTP", default_value_t = false)]
    enable_mtp: bool,
    /// Path to a companion MTP draft model GGUF file
    #[arg(long, env = "XRT_QWEN_MTP_DRAFT_MODEL")]
    mtp_draft_model: Option<String>,
    /// Maximum recursive draft tokens for MTP speculation (1..15)
    #[arg(long, env = "XRT_QWEN_MTP_MAX_DRAFT_TOKENS")]
    mtp_max_draft_tokens: Option<usize>,
    /// Adaptively skip speculative drafting when acceptance rate drops
    #[arg(long, env = "XRT_QWEN_MTP_ADAPTIVE_FALLBACK")]
    mtp_adaptive_fallback: Option<bool>,
    /// Default KV cache precision mode (f32, f16, q8_0, q4_0, etc.)
    #[arg(long, env = "XRT_KV_CACHE_MODE")]
    kv_cache_mode: Option<String>,
    /// Override model context window size (tokens)
    #[arg(short = 'c', long = "ctx-size", alias = "context-length", env = "XRT_CONTEXT_LENGTH")]
    ctx_size: Option<usize>,
}

#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<Option<Arc<Runtime>>>>,
    external_openai: Arc<RwLock<Option<ExternalOpenAiClient>>>,
    requested_backend: Arc<RwLock<BackendKind>>,
    loaded_model_name: Arc<RwLock<Option<String>>>,
    loaded_model_path: Arc<RwLock<Option<String>>>,
    loaded_mmproj_path: Arc<RwLock<Option<String>>>,
    gpu_resources: Arc<GpuResourceManager>,
    scheduler: Arc<RequestScheduler>,
    stream_buffer_capacity: usize,
    #[cfg(feature = "image-generation")]
    image: image_api::ImageServerState,
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
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
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
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stream: Option<bool>,
    seed: Option<u64>,
    /// Additive Qwen-compatible chat-template control. Omitted preserves the
    /// model template's default behavior.
    enable_thinking: Option<bool>,
    /// vLLM-compatible envelope for chat-template variables.
    chat_template_kwargs: Option<ChatTemplateKwargs>,
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
struct ChatTemplateKwargs {
    enable_thinking: Option<bool>,
}

impl ChatCompletionRequest {
    fn resolved_enable_thinking(&self) -> Option<bool> {
        self.enable_thinking.or_else(|| {
            self.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.enable_thinking)
        })
    }
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
    message: ChatResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

// --- /v1/models response types ---

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Clone, Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Debug, Deserialize)]
struct RuntimeLoadRequest {
    modality: Option<String>,
    model: Option<String>,
    model_path: Option<String>,
    hf_repo: Option<String>,
    hf_file: Option<String>,
    mmproj_path: Option<String>,
    backend: Option<String>,
    external_base_url: Option<String>,
    external_api_key: Option<String>,
    external_model: Option<String>,
    ctx_size: Option<usize>,
    context_length: Option<usize>,
    mtp_adaptive_fallback: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct RuntimeUnloadRequest {
    modality: Option<String>,
    model: Option<String>,
    #[serde(default)]
    force: bool,
}

#[derive(Serialize)]
struct RuntimeStatusResponse {
    object: &'static str,
    ready: bool,
    kv_cache_mode: &'static str,
    requested_backend: String,
    active_backend: Option<String>,
    gpu_resource: GpuResourceStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe: Option<MoeRuntimeStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hybrid_state: Option<HybridRuntimeStatus>,
    prefix_cache: PrefixCacheStatus,
    scheduler: SchedulerStatus,
    loaded_model: Option<String>,
    loaded_model_path: Option<String>,
    loaded_mmproj_path: Option<String>,
    external_base_url: Option<String>,
    external_model: Option<String>,
}

#[derive(Serialize)]
struct RuntimeLoadResponse {
    success: bool,
    loaded_model: String,
    loaded_model_path: Option<String>,
    loaded_mmproj_path: Option<String>,
    external_base_url: Option<String>,
    external_model: Option<String>,
    requested_backend: String,
    active_backend: String,
    gpu_resource: GpuResourceStatus,
    prefix_cache: PrefixCacheStatus,
}

#[derive(Serialize)]
struct RuntimeUnloadResponse {
    success: bool,
}

#[derive(Serialize)]
struct RuntimeCapabilitiesResponse {
    object: &'static str,
    version: &'static str,
    modalities: Vec<&'static str>,
    available_backends: Vec<&'static str>,
    cuda_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_device_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_total_vram_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_free_vram_bytes: Option<u64>,
    supported_architectures: Vec<&'static str>,
    supported_kv_cache_modes: Vec<&'static str>,
    mtp_speculative_supported: bool,
    prefix_caching_supported: bool,
    hybrid_moe_supported: bool,
}

#[derive(Debug, Deserialize)]
struct RuntimePreflightRequest {
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    is_moe: Option<bool>,
}

#[derive(Serialize)]
struct RuntimePreflightResponse {
    object: &'static str,
    fits: bool,
    recommended_backend: String,
    estimated_vram_bytes: u64,
    free_vram_bytes: u64,
    total_vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
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
    if cli.enable_mtp {
        std::env::set_var("XRT_QWEN_MTP", "1");
    }
    if let Some(draft_model) = &cli.mtp_draft_model {
        std::env::set_var("XRT_QWEN_MTP_DRAFT_MODEL", draft_model);
    }
    if let Some(max_draft) = cli.mtp_max_draft_tokens {
        std::env::set_var("XRT_QWEN_MTP_MAX_DRAFT_TOKENS", max_draft.to_string());
    }
    if let Some(adaptive) = cli.mtp_adaptive_fallback {
        std::env::set_var(
            "XRT_QWEN_MTP_ADAPTIVE_FALLBACK",
            if adaptive { "1" } else { "0" },
        );
    }
    if let Some(cache_mode) = &cli.kv_cache_mode {
        std::env::set_var("XRT_KV_CACHE_MODE", cache_mode);
    }
    if let Some(ctx) = cli.ctx_size {
        std::env::set_var("XRT_CONTEXT_LENGTH", ctx.to_string());
    }
    let initial_backend = parse_backend_value(&cli.backend)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
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
    let gpu_resources = Arc::new(GpuResourceManager::from_env());
    #[cfg(feature = "image-generation")]
    let image = image_api::ImageServerState::from_env(Arc::clone(&gpu_resources), &cli.host)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let state = AppState {
        runtime: Arc::new(RwLock::new(None)),
        external_openai: Arc::new(RwLock::new(None)),
        requested_backend: Arc::new(RwLock::new(initial_backend)),
        loaded_model_name: Arc::new(RwLock::new(None)),
        loaded_model_path: Arc::new(RwLock::new(None)),
        loaded_mmproj_path: Arc::new(RwLock::new(None)),
        gpu_resources,
        scheduler: Arc::new(RequestScheduler::new(scheduler_config)),
        stream_buffer_capacity: cli.stream_buffer_capacity,
        #[cfg(feature = "image-generation")]
        image,
    };

    if initial_backend == BackendKind::ExternalOpenAi
        || cli.model.is_some()
        || cli.hf_repo.is_some()
    {
        load_runtime_from_cli(&state, &cli).await?;
    }

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/runtime/status", get(runtime_status))
        .route("/v1/runtime/capabilities", get(runtime_capabilities))
        .route("/v1/runtime/preflight", post(runtime_preflight))
        .route("/v1/runtime/load", post(runtime_load))
        .route("/v1/runtime/unload", post(runtime_unload))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // Image-domain task endpoints, served from `xrt-vision`. These run
        // ONNX inference (BiRefNet et al.) inside `tokio::task::spawn_blocking`
        // so the async executor stays unblocked across the multi-second hits.
        .route("/v1/images/remove-background", post(remove_background));
    #[cfg(feature = "image-generation")]
    let app = app
        .route("/v1/images/generations", post(image_api::image_generations))
        .route(
            "/v1/images/edits",
            post(image_api::image_edits).layer(axum::extract::DefaultBodyLimit::max(
                image_api::MAX_EDIT_REQUEST_BYTES,
            )),
        )
        .route("/v1/runtime/models", get(image_api::runtime_models));
    let app = app.with_state(state);

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
    let requested_backend = parse_backend_value(&cli.backend)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    if requested_backend == BackendKind::ExternalOpenAi {
        if cli.model.is_some()
            || cli.hf_repo.is_some()
            || cli.hf_file.is_some()
            || cli.mmproj.is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "external-openai cannot be combined with local model or mmproj arguments",
            )
            .into());
        }
        let config = ExternalOpenAiConfig::from_env_with_overrides(
            cli.external_base_url.as_deref(),
            cli.external_api_key.as_deref(),
            cli.external_model.as_deref(),
        )
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        activate_external_openai(state, config).await;
        return Ok(());
    }

    let model_path = resolve_model_path(cli)?;
    let runtime = Runtime::load_with_backend_configs_and_resource_manager(
        &model_path,
        requested_backend,
        MoeRuntimeConfig::from_env()?,
        state.gpu_resources.config(),
        Arc::clone(&state.gpu_resources),
    )?;
    let runtime = if let Some(mmproj_path) = cli.mmproj.as_deref() {
        runtime.load_vision(mmproj_path)?
    } else {
        runtime
    };

    activate_local_runtime(state, runtime, model_path, cli.mmproj.clone()).await;
    Ok(())
}

async fn activate_external_openai(state: &AppState, config: ExternalOpenAiConfig) {
    let model_name = config.display_model().to_string();
    let client = ExternalOpenAiClient::new(config);
    *state.runtime.write().await = None;
    state.scheduler.configure_kv_budget(None);
    state.scheduler.configure_external_kv_bytes(0);
    *state.requested_backend.write().await = BackendKind::ExternalOpenAi;
    *state.loaded_model_name.write().await = Some(model_name);
    *state.loaded_model_path.write().await = None;
    *state.loaded_mmproj_path.write().await = None;
    *state.external_openai.write().await = Some(client);
}

async fn activate_local_runtime(
    state: &AppState,
    runtime: Arc<Runtime>,
    model_path: String,
    mmproj_path: Option<String>,
) {
    let model_name = runtime.model_name().to_string();
    let requested_backend = runtime.requested_backend();
    state
        .scheduler
        .configure_kv_budget(runtime.gpu_resource_status().kv_budget_bytes);
    state
        .scheduler
        .configure_external_kv_bytes(runtime.prefix_cache_status().device_resident_bytes);
    *state.external_openai.write().await = None;
    *state.requested_backend.write().await = requested_backend;
    *state.runtime.write().await = Some(runtime);
    *state.loaded_model_name.write().await = Some(model_name);
    *state.loaded_model_path.write().await = Some(model_path);
    *state.loaded_mmproj_path.write().await = mmproj_path;
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

async fn list_models(State(state): State<AppState>) -> Result<Response, (StatusCode, String)> {
    #[cfg(feature = "image-generation")]
    let image_models = state.image.openai_models().await;
    #[cfg(not(feature = "image-generation"))]
    let image_models: Vec<ModelInfo> = Vec::new();

    if let Some(config) = state.external_openai.read().await.clone() {
        if image_models.is_empty() {
            return external_openai::proxy_get(config, "models").await;
        }
        let image_models = image_models
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(internal_error)?;
        return external_openai::proxy_models_merged(config, image_models).await;
    }
    let model_name = state.loaded_model_name.read().await.clone();
    let mut data = model_name
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model",
            created: unix_timestamp(),
            owned_by: "xeno-rt",
        })
        .collect::<Vec<_>>();
    data.extend(image_models);
    Ok(Json(ModelList {
        object: "list",
        data,
    })
    .into_response())
}

async fn runtime_status(State(state): State<AppState>) -> Json<RuntimeStatusResponse> {
    let runtime = state.runtime.read().await.clone();
    let external = state.external_openai.read().await.clone();
    let loaded_model = state.loaded_model_name.read().await.clone();
    let loaded_model_path = state.loaded_model_path.read().await.clone();
    let loaded_mmproj_path = state.loaded_mmproj_path.read().await.clone();
    let requested_backend = state.requested_backend.read().await.as_str().to_string();
    let active_backend = external
        .as_ref()
        .map(|_| "external-openai".to_string())
        .or_else(|| {
            runtime
                .as_ref()
                .map(|runtime| runtime.active_backend().as_str().to_string())
        });
    let gpu_resource = runtime
        .as_ref()
        .map(|runtime| runtime.gpu_resource_status())
        .unwrap_or_else(|| state.gpu_resources.status());
    let prefix_cache = runtime
        .as_ref()
        .map(|runtime| runtime.prefix_cache_status())
        .unwrap_or_else(|| {
            PrefixCacheManager::from_env(if external.is_some() {
                "external-openai"
            } else {
                "unloaded"
            })
            .status()
        });
    let moe = runtime.as_ref().map(|runtime| runtime.moe_status());
    let hybrid_state = runtime
        .as_ref()
        .and_then(|runtime| runtime.hybrid_state_status());
    Json(RuntimeStatusResponse {
        object: "runtime.status",
        ready: runtime.is_some() || external.is_some(),
        kv_cache_mode: xrt_runtime::KvCacheMode::from_env().as_str(),
        requested_backend,
        active_backend,
        gpu_resource,
        moe,
        hybrid_state,
        prefix_cache,
        scheduler: state.scheduler.status(),
        loaded_model,
        loaded_model_path,
        loaded_mmproj_path,
        external_base_url: external
            .as_ref()
            .map(|client| client.config().base_url().to_string()),
        external_model: external
            .as_ref()
            .and_then(|client| client.config().default_model().map(ToOwned::to_owned)),
    })
}

async fn runtime_capabilities(State(state): State<AppState>) -> Json<RuntimeCapabilitiesResponse> {
    let gpu_status = state.gpu_resources.status();
    #[allow(unused_mut)]
    let mut modalities = vec!["text", "vision"];
    #[cfg(feature = "image-generation")]
    modalities.push("image");

    let mut backends = vec!["cpu", "external-openai"];
    if gpu_status.cuda_available {
        backends.push("cuda-resident");
    }

    Json(RuntimeCapabilitiesResponse {
        object: "runtime.capabilities",
        version: env!("CARGO_PKG_VERSION"),
        modalities,
        available_backends: backends,
        cuda_available: gpu_status.cuda_available,
        cuda_device_name: gpu_status.device_name,
        cuda_total_vram_bytes: gpu_status.total_vram_bytes,
        cuda_free_vram_bytes: gpu_status.free_vram_bytes,
        supported_architectures: vec![
            "qwen35",
            "qwen35moe",
            "qwen2",
            "llama",
            "gemma",
            "gemma4",
            "glm",
        ],
        supported_kv_cache_modes: vec![
            "f32",
            "f16",
            "q8_0",
            "q4_0",
            "key_q4_value_q8",
        ],
        mtp_speculative_supported: true,
        prefix_caching_supported: true,
        hybrid_moe_supported: true,
    })
}

async fn runtime_preflight(
    State(state): State<AppState>,
    Json(request): Json<RuntimePreflightRequest>,
) -> Json<RuntimePreflightResponse> {
    let gpu_status = state.gpu_resources.status();
    let free_vram = gpu_status.free_vram_bytes.unwrap_or(0);
    let total_vram = gpu_status.total_vram_bytes.unwrap_or(0);

    let raw_size = if let Some(size) = request.size_bytes {
        size
    } else if let Some(path) = request.model_path.as_deref() {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let is_moe = request.is_moe.unwrap_or(false);
    // MoE hybrid expert placement runs active weights + router on GPU (~50-55% of file size),
    // dense models run entire weight tensor on GPU + scratch (~115% of file size).
    let estimated_vram = if is_moe {
        (raw_size as f64 * 0.55) as u64
    } else {
        (raw_size as f64 * 1.15) as u64
    };

    let explicit_cpu = request
        .backend
        .as_deref()
        .map(|b| b.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);

    let (fits, recommended_backend, reason) = if explicit_cpu {
        (true, "cpu".to_string(), None)
    } else if !gpu_status.cuda_available {
        (
            true,
            "cpu".to_string(),
            Some("CUDA is unavailable; model will run on CPU fallback".to_string()),
        )
    } else if free_vram >= estimated_vram {
        (
            true,
            "cuda-resident".to_string(),
            None,
        )
    } else if is_moe && free_vram >= (raw_size as f64 * 0.35) as u64 {
        (
            true,
            "cuda-resident".to_string(),
            Some("Model will fit using hybrid CPU/GPU expert placement".to_string()),
        )
    } else {
        (
            false,
            "cpu".to_string(),
            Some(format!(
                "Model requires estimated {estimated_vram} bytes VRAM, but only {free_vram} bytes are free; recommend CPU fallback or closing other applications"
            )),
        )
    };

    Json(RuntimePreflightResponse {
        object: "runtime.preflight",
        fits,
        recommended_backend,
        estimated_vram_bytes: estimated_vram,
        free_vram_bytes: free_vram,
        total_vram_bytes: total_vram,
        reason,
    })
}

async fn runtime_load(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RuntimeLoadRequest>,
) -> Result<Response, (StatusCode, String)> {
    #[cfg(feature = "image-generation")]
    state
        .image
        .authorize_admin(&headers)
        .map_err(|message| (StatusCode::UNAUTHORIZED, message))?;
    #[cfg(not(feature = "image-generation"))]
    let _ = headers;

    let modality = parse_runtime_modality(request.modality.as_deref())
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    if modality == "image" {
        #[cfg(feature = "image-generation")]
        {
            if present(&request.model_path)
                || present(&request.hf_repo)
                || present(&request.hf_file)
                || present(&request.mmproj_path)
                || present(&request.external_base_url)
                || present(&request.external_api_key)
                || present(&request.external_model)
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "image runtime load accepts only an installed catalog model ID and backend"
                        .to_string(),
                ));
            }
            let model = request
                .model
                .clone()
                .filter(|model| !model.trim().is_empty())
                .ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "image runtime load requires `model`".to_string(),
                    )
                })?;
            let backend = image_api::parse_image_backend(request.backend.as_deref())
                .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
            let response = state.image.load_installed(model, backend).await?;
            return Ok(Json(response).into_response());
        }
        #[cfg(not(feature = "image-generation"))]
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "image generation support is not enabled in this server build".to_string(),
            ));
        }
    }
    if present(&request.model) {
        return Err((
            StatusCode::BAD_REQUEST,
            "`model` is valid only when modality is `image`; existing text loads use model_path or hf_repo + hf_file"
                .to_string(),
        ));
    }

    if let Some(ctx) = request.ctx_size.or(request.context_length) {
        std::env::set_var("XRT_CONTEXT_LENGTH", ctx.to_string());
    }

    if let Some(adaptive) = request.mtp_adaptive_fallback {
        std::env::set_var(
            "XRT_QWEN_MTP_ADAPTIVE_FALLBACK",
            if adaptive { "1" } else { "0" },
        );
    }

    let requested_backend = request
        .backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(parse_backend_value)
        .transpose()
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let requested_backend = match requested_backend {
        Some(requested_backend) => requested_backend,
        None => *state.requested_backend.read().await,
    };
    if requested_backend == BackendKind::ExternalOpenAi {
        if request
            .model_path
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            || request
                .hf_repo
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            || request
                .hf_file
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            || request
                .mmproj_path
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "external-openai cannot be combined with local model or mmproj fields".to_string(),
            ));
        }
        let config = ExternalOpenAiConfig::from_env_with_overrides(
            request.external_base_url.as_deref(),
            request.external_api_key.as_deref(),
            request.external_model.as_deref(),
        )
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
        let loaded_model = config.display_model().to_string();
        let external_base_url = Some(config.base_url().to_string());
        let external_model = config.default_model().map(ToOwned::to_owned);
        let gpu_resource = state.gpu_resources.status();
        let prefix_cache = PrefixCacheManager::from_env("external-openai").status();
        activate_external_openai(&state, config).await;
        return Ok(Json(RuntimeLoadResponse {
            success: true,
            loaded_model,
            loaded_model_path: None,
            loaded_mmproj_path: None,
            external_base_url,
            external_model,
            requested_backend: BackendKind::ExternalOpenAi.as_str().to_string(),
            active_backend: BackendKind::ExternalOpenAi.as_str().to_string(),
            gpu_resource,
            prefix_cache,
        })
        .into_response());
    }
    if request
        .external_base_url
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        || request
            .external_api_key
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        || request
            .external_model
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "external OpenAI fields require backend `external-openai`".to_string(),
        ));
    }
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
                external_base_url: None,
                external_api_key: None,
                external_model: None,
                max_active_sequences: 1,
                max_queued_sequences: 32,
                stream_buffer_capacity: 32,
                prefill_chunk_tokens: 128,
                max_decode_turns_before_prefill: 8,
                max_decode_batch_size: 4,
                decode_batch_wait_micros: 20_000,
                enable_mtp: false,
                mtp_draft_model: None,
                mtp_max_draft_tokens: None,
                kv_cache_mode: None,
                ctx_size: None,
                mtp_adaptive_fallback: None,
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
        let gpu_resources = Arc::clone(&state.gpu_resources);
        move || {
            let runtime = Runtime::load_with_backend_configs_and_resource_manager(
                &model_path,
                requested_backend,
                MoeRuntimeConfig::from_env()?,
                gpu_resources.config(),
                gpu_resources,
            )?;
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
    let prefix_cache = runtime.prefix_cache_status();
    activate_local_runtime(&state, runtime, model_path.clone(), mmproj_path.clone()).await;

    Ok(Json(RuntimeLoadResponse {
        success: true,
        loaded_model,
        loaded_model_path: Some(model_path),
        loaded_mmproj_path: mmproj_path,
        external_base_url: None,
        external_model: None,
        requested_backend,
        active_backend,
        gpu_resource,
        prefix_cache,
    })
    .into_response())
}

fn parse_backend_value(value: &str) -> Result<BackendKind, String> {
    BackendKind::parse(value).ok_or_else(|| format!("unsupported backend value: {value}"))
}

fn parse_runtime_modality(value: Option<&str>) -> Result<&'static str, String> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("text") => Ok("text"),
        Some("image") => Ok("image"),
        Some(other) => Err(format!("unsupported runtime modality: {other}")),
    }
}

fn present(value: &Option<String>) -> bool {
    value
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
}

async fn runtime_unload(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Option<Json<RuntimeUnloadRequest>>,
) -> Result<Response, (StatusCode, String)> {
    #[cfg(feature = "image-generation")]
    state
        .image
        .authorize_admin(&headers)
        .map_err(|message| (StatusCode::UNAUTHORIZED, message))?;
    #[cfg(not(feature = "image-generation"))]
    let _ = headers;
    let request = request.map(|Json(request)| request).unwrap_or_default();
    let modality = parse_runtime_modality(request.modality.as_deref())
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    if modality == "image" {
        #[cfg(feature = "image-generation")]
        {
            return state
                .image
                .unload(request.model.as_deref(), request.force)
                .await
                .map(|response| Json(response).into_response());
        }
        #[cfg(not(feature = "image-generation"))]
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "image generation support is not enabled in this server build".to_string(),
            ));
        }
    }
    if present(&request.model) || request.force {
        return Err((
            StatusCode::BAD_REQUEST,
            "text unload remains bodyless; model and force apply only to modality `image`"
                .to_string(),
        ));
    }
    *state.runtime.write().await = None;
    *state.external_openai.write().await = None;
    state.scheduler.configure_kv_budget(None);
    state.scheduler.configure_external_kv_bytes(0);
    *state.loaded_model_name.write().await = None;
    *state.loaded_model_path.write().await = None;
    *state.loaded_mmproj_path.write().await = None;
    Ok(Json(RuntimeUnloadResponse { success: true }).into_response())
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
    Json(payload): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, String)> {
    if let Some(config) = state.external_openai.read().await.clone() {
        if payload_requests_streaming(&payload) {
            return external_openai::proxy_sse(
                config,
                "completions",
                payload,
                state.stream_buffer_capacity,
            )
            .await;
        }
        return external_openai::proxy_json(config, "completions", payload).await;
    }
    let request: CompletionRequest = serde_json::from_value(payload).map_err(bad_request)?;
    if request.stream.unwrap_or(false) {
        completion_stream(state, request).await
    } else {
        completion_once(state, request).await
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, String)> {
    if let Some(config) = state.external_openai.read().await.clone() {
        if payload_requests_streaming(&payload) {
            return external_openai::proxy_sse(
                config,
                "chat/completions",
                payload,
                state.stream_buffer_capacity,
            )
            .await;
        }
        return external_openai::proxy_json(config, "chat/completions", payload).await;
    }
    let request: ChatCompletionRequest = serde_json::from_value(payload).map_err(bad_request)?;
    if request.stream.unwrap_or(false) {
        chat_stream(state, request).await
    } else {
        chat_once(state, request).await
    }
}

fn payload_requests_streaming(payload: &serde_json::Value) -> bool {
    payload
        .as_object()
        .and_then(|object| object.get("stream"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn generation_finish_reason(generated_tokens: usize, max_tokens: usize) -> &'static str {
    if generated_tokens >= max_tokens {
        "length"
    } else {
        "stop"
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
    let prompt_token_ids = runtime
        .tokenizer()
        .encode_with_options(&prompt_text, true, true)
        .map_err(internal_error)?;
    let prompt_tokens = prompt_token_ids.len();

    let permit = acquire_inference_permit(&state).await?;
    let generate_runtime = runtime.clone();
    let scheduler = state.scheduler.clone();
    let max_tokens = generate.max_tokens;
    let (text, generated_token_ids) = task::spawn_blocking(move || {
        let _permit = permit;
        let mut session = generate_runtime.new_session();
        session
            .generate_scheduled_with_prompt_tokens(&generate, &prompt_token_ids, &scheduler)
            .map(|text| {
                let generated_token_ids = session
                    .generated_token_ids()
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                (text, generated_token_ids)
            })
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;
    let generated_tokens = generated_token_ids.len();

    // Generated token IDs are authoritative. Re-tokenizing decoded text can
    // undercount stripped special tokens (notably Qwen's hidden </think>
    // boundary) or choose a different equivalent tokenization.
    let completion_tokens = generated_tokens;

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
            finish_reason: generation_finish_reason(generated_tokens, max_tokens),
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
    let parse_reasoning = runtime.model_architecture() == "qwen35"
        && request.resolved_enable_thinking() != Some(false);
    let prepared_chat = prepare_chat_request(&request.messages, &runtime)?;
    let (prompt, prompt_spans) = chat_prompt_with_spans(
        &prepared_chat.messages,
        request.tools.as_deref(),
        &runtime,
        request.resolved_enable_thinking(),
    );
    let mut generate = request_to_generate_request(prompt.clone(), &request, false);
    apply_qwen38_chat_defaults(&runtime, &request, &mut generate);
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

    let prompt_token_ids = runtime
        .tokenizer()
        .encode_with_options(&prompt, false, true)
        .map_err(internal_error)?;
    let prompt_tokens = prompt_token_ids.len();

    let permit = acquire_inference_permit(&state).await?;
    let generate_runtime = runtime.clone();
    let scheduler = state.scheduler.clone();
    let max_tokens = generate.max_tokens;
    let (text, generated_token_ids) = task::spawn_blocking(move || {
        let _permit = permit;
        let mut session = generate_runtime.new_session();
        session
            .generate_scheduled_with_prompt_tokens(&generate, &prompt_token_ids, &scheduler)
            .map(|text| {
                let generated_token_ids = session
                    .generated_token_ids()
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                (text, generated_token_ids)
            })
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;
    let generated_tokens = generated_token_ids.len();

    // Include reasoning and hidden model-control tokens exactly as generated;
    // decoded response text is not a reliable token-accounting source.
    let completion_tokens = generated_tokens;

    let (reasoning_content, sanitized_text) =
        split_assistant_reasoning_tokens(&runtime, &text, &generated_token_ids, parse_reasoning);
    let (response_text, response_tool_calls, finish_reason) =
        if let Some(tool_calls) = extract_tool_calls_from_text(&sanitized_text) {
            (String::new(), Some(tool_calls), "tool_calls")
        } else {
            (
                sanitized_text,
                None,
                generation_finish_reason(generated_tokens, max_tokens),
            )
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
            message: ChatResponseMessage {
                role: "assistant".to_string(),
                content: response_text,
                reasoning_content,
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
        let finish_reason = match result {
            Ok(generated_tokens) => generation_finish_reason(generated_tokens, generate.max_tokens),
            Err(_) => "error",
        };
        let finish = CompletionChunk {
            id,
            object: "text_completion.chunk",
            created,
            model: model_name,
            choices: vec![CompletionChunkChoice {
                text: String::new(),
                index: 0,
                finish_reason: Some(finish_reason),
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
    let parse_reasoning = runtime.model_architecture() == "qwen35"
        && request.resolved_enable_thinking() != Some(false);
    let reasoning_boundary = parse_reasoning
        .then(|| runtime.tokenizer().token_id_for_piece("</think>"))
        .flatten();
    let (tx, rx) =
        mpsc::channel::<Result<Event, Infallible>>(state.scheduler.config().stream_buffer_capacity);
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| runtime.model_name().to_string());
    let prepared_chat = prepare_chat_request(&request.messages, &runtime)?;
    let (prompt, prompt_spans) = chat_prompt_with_spans(
        &prepared_chat.messages,
        request.tools.as_deref(),
        &runtime,
        request.resolved_enable_thinking(),
    );
    let mut generate = request_to_generate_request(prompt, &request, false);
    apply_qwen38_chat_defaults(&runtime, &request, &mut generate);
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
        let mut parser = ThinkingStreamParser::new(parse_reasoning);
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
                    reasoning_content: None,
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
        let result = session.generate_stream_scheduled_with_token_control(
            &generate,
            &scheduler,
            |token, piece| {
                for parsed in parser.push_token(token, piece, reasoning_boundary) {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: parsed.content,
                                reasoning_content: parsed.reasoning_content,
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(data) = serde_json::to_string(&chunk) {
                        if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                            return ControlFlow::Break(());
                        }
                    }
                }
                ControlFlow::Continue(())
            },
        );

        if tx.is_closed() {
            return;
        }
        if let Some(parsed) = parser.finish() {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: parsed.content,
                        reasoning_content: parsed.reasoning_content,
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(data) = serde_json::to_string(&chunk) {
                if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                    return;
                }
            }
        }
        let finish_reason = match result {
            Ok(generated_tokens) => generation_finish_reason(generated_tokens, generate.max_tokens),
            Err(_) => "error",
        };
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
                    reasoning_content: None,
                },
                finish_reason: Some(finish_reason),
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
        presence_penalty: request.presence_penalty().unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty().unwrap_or(0.0),
        seed: request.seed(),
        ..Default::default()
    }
}

fn apply_qwen38_chat_defaults(
    runtime: &Runtime,
    request: &ChatCompletionRequest,
    generation: &mut GenerateRequest,
) {
    let thinking = request.resolved_enable_thinking() != Some(false);
    let Some((temperature, top_k, top_p, repetition, presence, frequency)) =
        qwen38_chat_default_profile(runtime.model_name(), runtime.model_architecture(), thinking)
    else {
        return;
    };
    if request.temperature.is_none() {
        generation.temperature = temperature;
    }
    if request.top_k.is_none() {
        generation.top_k = top_k;
    }
    if request.top_p.is_none() {
        generation.top_p = top_p;
    }
    if request.repetition_penalty.is_none() {
        generation.repetition_penalty = repetition;
    }
    if request.presence_penalty.is_none() {
        generation.presence_penalty = presence;
    }
    if request.frequency_penalty.is_none() {
        generation.frequency_penalty = frequency;
    }
}

fn qwen38_chat_default_profile(
    model_name: &str,
    architecture: &str,
    thinking: bool,
) -> Option<(f32, usize, f32, f32, f32, f32)> {
    if architecture != "qwen35" || !model_name.to_ascii_lowercase().starts_with("qwen3.8") {
        return None;
    }
    Some(if thinking {
        (1.0, 20, 0.95, 1.0, 0.0, 0.0)
    } else {
        (0.7, 20, 0.8, 1.0, 1.5, 0.0)
    })
}

trait RequestConfig {
    fn cache_policy(&self) -> Option<&str>;
    fn recent_window_tokens(&self) -> Option<usize>;
    fn max_tokens(&self) -> Option<usize>;
    fn temperature(&self) -> Option<f32>;
    fn top_k(&self) -> Option<usize>;
    fn top_p(&self) -> Option<f32>;
    fn repetition_penalty(&self) -> Option<f32>;
    fn presence_penalty(&self) -> Option<f32>;
    fn frequency_penalty(&self) -> Option<f32>;
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
    fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }
    fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
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
    fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }
    fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }
    fn seed(&self) -> Option<u64> {
        self.seed
    }
}

fn chat_prompt_with_spans(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    runtime: &Runtime,
    enable_thinking: Option<bool>,
) -> (String, Vec<PromptSpan>) {
    let prepared = prepare_template_messages(messages, tools, runtime);
    let prompt = render_prepared_messages(runtime, &prepared, true, enable_thinking);
    let tokenizer = runtime.tokenizer();
    let mut spans = Vec::new();
    let mut previous_end = 0usize;

    for end_index in 0..prepared.len() {
        let prefix_prompt =
            render_prepared_messages(runtime, &prepared[..=end_index], false, enable_thinking);
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
    enable_thinking: Option<bool>,
) -> String {
    let template_messages = prepared
        .iter()
        .map(|message| message.message.clone())
        .collect::<Vec<_>>();

    match format_runtime_chat(
        runtime,
        &template_messages,
        add_generation_prompt,
        enable_thinking,
    ) {
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
    enable_thinking: Option<bool>,
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
        let mut prompt =
            apply_chat_template(CHATML_TEMPLATE, messages, bos, eos, add_generation_prompt)
                .map_err(|e| e.to_string())?;
        if add_generation_prompt {
            match enable_thinking {
                Some(false) => prompt.push_str("<think>\n\n</think>\n\n"),
                Some(true) => prompt.push_str("<think>\n"),
                None => {}
            }
        }
        return Ok(prompt);
    }

    runtime
        .tokenizer()
        .format_chat_with_thinking(messages, add_generation_prompt, enable_thinking)
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

fn split_assistant_reasoning(text: &str) -> (Option<String>, String) {
    if let Some((reasoning, answer)) = text.split_once("</think>") {
        let reasoning = reasoning
            .trim()
            .strip_prefix("<think>")
            .unwrap_or(reasoning.trim())
            .trim()
            .to_string();
        return (
            (!reasoning.is_empty()).then_some(reasoning),
            sanitize_assistant_text(answer),
        );
    }
    let trimmed = text.trim();
    if let Some(reasoning) = trimmed.strip_prefix("<think>") {
        let reasoning = reasoning.trim().to_string();
        return ((!reasoning.is_empty()).then_some(reasoning), String::new());
    }
    (None, sanitize_assistant_text(text))
}

fn split_assistant_reasoning_tokens(
    runtime: &Runtime,
    text: &str,
    token_ids: &[u32],
    expected: bool,
) -> (Option<String>, String) {
    let tokenizer = runtime.tokenizer();
    let Some(boundary) = tokenizer.token_id_for_piece("</think>") else {
        return split_assistant_reasoning_fallback(text, expected);
    };
    let Some(index) = token_ids.iter().position(|&token| token == boundary) else {
        return split_assistant_reasoning_fallback(text, expected);
    };
    let reasoning = tokenizer
        .decode(&token_ids[..index], true)
        .unwrap_or_else(|_| text.to_string());
    let answer = tokenizer
        .decode(&token_ids[index + 1..], true)
        .unwrap_or_default();
    let reasoning = reasoning.trim().to_string();
    (
        (!reasoning.is_empty()).then_some(reasoning),
        sanitize_assistant_text(&answer),
    )
}

fn split_assistant_reasoning_fallback(text: &str, expected: bool) -> (Option<String>, String) {
    let parsed = split_assistant_reasoning(text);
    if parsed.0.is_some() || !expected {
        return parsed;
    }
    let reasoning = sanitize_assistant_text(text);
    ((!reasoning.is_empty()).then_some(reasoning), String::new())
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedChatChunk {
    content: Option<String>,
    reasoning_content: Option<String>,
}

struct ThinkingStreamParser {
    in_reasoning: bool,
    reasoning_prefix_resolved: bool,
    pending: String,
}

impl ThinkingStreamParser {
    const OPEN: &'static str = "<think>";
    const CLOSE: &'static str = "</think>";

    fn new(in_reasoning: bool) -> Self {
        Self {
            in_reasoning,
            reasoning_prefix_resolved: !in_reasoning,
            pending: String::new(),
        }
    }

    fn push(&mut self, piece: &str) -> Vec<ParsedChatChunk> {
        self.pending.push_str(piece);
        if !self.in_reasoning {
            return self.take_pending_content().into_iter().collect();
        }

        if !self.reasoning_prefix_resolved {
            let trimmed = self.pending.trim_start();
            if Self::OPEN.starts_with(trimmed) && trimmed.len() < Self::OPEN.len() {
                return Vec::new();
            }
            if let Some(rest) = trimmed.strip_prefix(Self::OPEN) {
                self.pending = rest.trim_start().to_string();
            }
            self.reasoning_prefix_resolved = true;
        }

        if let Some(close) = self.pending.find(Self::CLOSE) {
            let reasoning = self.pending[..close].to_string();
            let answer = self.pending[close + Self::CLOSE.len()..].to_string();
            self.pending.clear();
            self.in_reasoning = false;
            let mut chunks = Vec::with_capacity(2);
            if !reasoning.is_empty() {
                chunks.push(ParsedChatChunk {
                    content: None,
                    reasoning_content: Some(reasoning),
                });
            }
            if !answer.is_empty() {
                chunks.push(ParsedChatChunk {
                    content: Some(answer),
                    reasoning_content: None,
                });
            }
            return chunks;
        }

        let keep = (1..Self::CLOSE.len())
            .rev()
            .find(|&length| self.pending.ends_with(&Self::CLOSE[..length]))
            .unwrap_or(0);
        let emit_len = self.pending.len().saturating_sub(keep);
        if emit_len == 0 {
            return Vec::new();
        }
        let reasoning = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        vec![ParsedChatChunk {
            content: None,
            reasoning_content: Some(reasoning),
        }]
    }

    fn push_token(
        &mut self,
        token: u32,
        piece: &str,
        reasoning_boundary: Option<u32>,
    ) -> Vec<ParsedChatChunk> {
        if self.in_reasoning && reasoning_boundary == Some(token) {
            let mut chunks = Vec::new();
            if let Some(chunk) = self.finish() {
                chunks.push(chunk);
            }
            self.in_reasoning = false;
            self.reasoning_prefix_resolved = true;
            return chunks;
        }
        self.push(piece)
    }

    fn finish(&mut self) -> Option<ParsedChatChunk> {
        if self.pending.is_empty() {
            return None;
        }
        let pending = std::mem::take(&mut self.pending);
        Some(if self.in_reasoning {
            ParsedChatChunk {
                content: None,
                reasoning_content: Some(pending),
            }
        } else {
            ParsedChatChunk {
                content: Some(pending),
                reasoning_content: None,
            }
        })
    }

    fn take_pending_content(&mut self) -> Option<ParsedChatChunk> {
        if self.pending.is_empty() {
            return None;
        }
        Some(ParsedChatChunk {
            content: Some(std::mem::take(&mut self.pending)),
            reasoning_content: None,
        })
    }
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
        activate_external_openai, extract_image_url, extract_text_part, generation_finish_reason,
        image_tensor_pixels, load_image_bytes, parse_runtime_modality, part_kind,
        payload_requests_streaming, preprocess_image, qwen38_chat_default_profile,
        request_to_generate_request, runtime_capabilities, runtime_preflight, runtime_status,
        runtime_unload, split_assistant_reasoning, split_assistant_reasoning_fallback, AppState,
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatResponseMessage,
        CompletionChoice, CompletionResponse, ModelInfo, ModelList, RuntimePreflightRequest,
        ThinkingStreamParser, UsageInfo,
    };
    use crate::external_openai::ExternalOpenAiConfig;
    use axum::{extract::State, http::HeaderMap, Json};
    use image::{DynamicImage, RgbImage};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use xrt_runtime::{BackendKind, RequestScheduler, SchedulerConfig};

    fn empty_state() -> AppState {
        let gpu_resources = Arc::new(xrt_runtime::GpuResourceManager::from_env());
        #[cfg(feature = "image-generation")]
        let image = crate::image_api::ImageServerState::for_tests(Arc::clone(&gpu_resources));
        AppState {
            runtime: Arc::new(RwLock::new(None)),
            external_openai: Arc::new(RwLock::new(None)),
            requested_backend: Arc::new(RwLock::new(BackendKind::Auto)),
            loaded_model_name: Arc::new(RwLock::new(None)),
            loaded_model_path: Arc::new(RwLock::new(None)),
            loaded_mmproj_path: Arc::new(RwLock::new(None)),
            gpu_resources,
            scheduler: Arc::new(RequestScheduler::new(
                SchedulerConfig::new(1, 1, 2).unwrap(),
            )),
            stream_buffer_capacity: 2,
            #[cfg(feature = "image-generation")]
            image,
        }
    }

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

    #[test]
    fn payload_stream_flag_requires_a_json_boolean() {
        assert!(payload_requests_streaming(
            &serde_json::json!({"stream": true})
        ));
        assert!(!payload_requests_streaming(
            &serde_json::json!({"stream": "true"})
        ));
        assert!(!payload_requests_streaming(&serde_json::json!([])));
    }

    #[test]
    fn generation_finish_reason_distinguishes_eos_from_token_limit() {
        assert_eq!(generation_finish_reason(7, 8), "stop");
        assert_eq!(generation_finish_reason(8, 8), "length");
        assert_eq!(generation_finish_reason(9, 8), "length");
    }

    #[test]
    fn openai_presence_and_frequency_penalties_reach_generation() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "presence_penalty": 0.75,
            "frequency_penalty": -0.25
        }))
        .unwrap();
        let generation = request_to_generate_request("prompt".to_string(), &request, false);
        assert_eq!(generation.presence_penalty, 0.75);
        assert_eq!(generation.frequency_penalty, -0.25);
    }

    #[test]
    fn chat_thinking_control_accepts_direct_and_vllm_compatible_forms() {
        let direct: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": false
        }))
        .unwrap();
        assert_eq!(direct.resolved_enable_thinking(), Some(false));

        let nested: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": true}
        }))
        .unwrap();
        assert_eq!(nested.resolved_enable_thinking(), Some(true));

        let default: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert_eq!(default.resolved_enable_thinking(), None);

        let direct_wins: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": false,
            "chat_template_kwargs": {"enable_thinking": true}
        }))
        .unwrap();
        assert_eq!(direct_wins.resolved_enable_thinking(), Some(false));
    }

    #[test]
    fn qwen38_chat_defaults_match_thinking_and_non_thinking_profiles() {
        assert_eq!(
            qwen38_chat_default_profile("Qwen3.8-27B", "qwen35", true),
            Some((1.0, 20, 0.95, 1.0, 0.0, 0.0))
        );
        assert_eq!(
            qwen38_chat_default_profile("Qwen3.8-27B", "qwen35", false),
            Some((0.7, 20, 0.8, 1.0, 1.5, 0.0))
        );
        assert_eq!(
            qwen38_chat_default_profile("Qwen3.6-27B", "qwen35", true),
            None
        );
    }

    #[test]
    fn qwen_reasoning_is_separated_from_openai_answer_content() {
        let (reasoning, answer) =
            split_assistant_reasoning("Multiply 240 by 1.35, then 0.8.\n</think>\n\n259.2");
        assert_eq!(
            reasoning.as_deref(),
            Some("Multiply 240 by 1.35, then 0.8.")
        );
        assert_eq!(answer, "259.2");

        let (reasoning, answer) = split_assistant_reasoning("plain answer");
        assert_eq!(reasoning, None);
        assert_eq!(answer, "plain answer");
    }

    #[test]
    fn truncated_qwen_reasoning_never_leaks_into_answer_content() {
        let (reasoning, answer) =
            split_assistant_reasoning_fallback("Still working through the calculation", true);
        assert_eq!(
            reasoning.as_deref(),
            Some("Still working through the calculation")
        );
        assert!(answer.is_empty());

        let (reasoning, answer) = split_assistant_reasoning_fallback("ordinary answer", false);
        assert_eq!(reasoning, None);
        assert_eq!(answer, "ordinary answer");
    }

    #[test]
    fn qwen_streaming_reasoning_parser_handles_split_close_tag() {
        let mut parser = ThinkingStreamParser::new(true);
        let mut chunks = parser.push("<think>reasoning</th");
        chunks.extend(parser.push("ink>final"));
        if let Some(chunk) = parser.finish() {
            chunks.push(chunk);
        }
        let reasoning = chunks
            .iter()
            .filter_map(|chunk| chunk.reasoning_content.as_deref())
            .collect::<String>();
        let content = chunks
            .iter()
            .filter_map(|chunk| chunk.content.as_deref())
            .collect::<String>();
        assert_eq!(reasoning, "reasoning");
        assert_eq!(content, "final");
        assert!(chunks.iter().all(|chunk| {
            chunk.content.as_deref() != Some("</think>")
                && chunk.reasoning_content.as_deref() != Some("</think>")
        }));
    }

    #[test]
    fn qwen_streaming_reasoning_parser_honors_hidden_boundary_token() {
        let mut parser = ThinkingStreamParser::new(true);
        let mut chunks = parser.push_token(1, "reasoning", Some(248_069));
        chunks.extend(parser.push_token(248_069, "", Some(248_069)));
        chunks.extend(parser.push_token(2, "final", Some(248_069)));
        let reasoning = chunks
            .iter()
            .filter_map(|chunk| chunk.reasoning_content.as_deref())
            .collect::<String>();
        let content = chunks
            .iter()
            .filter_map(|chunk| chunk.content.as_deref())
            .collect::<String>();
        assert_eq!(reasoning, "reasoning");
        assert_eq!(content, "final");
    }

    #[test]
    fn omitted_runtime_modality_remains_text() {
        assert_eq!(parse_runtime_modality(None).unwrap(), "text");
        assert_eq!(parse_runtime_modality(Some("text")).unwrap(), "text");
        assert_eq!(parse_runtime_modality(Some("image")).unwrap(), "image");
        assert!(parse_runtime_modality(Some("video")).is_err());
    }

    #[tokio::test]
    async fn bodyless_runtime_unload_preserves_the_text_contract() {
        let state = empty_state();
        let response = runtime_unload(State(state), HeaderMap::new(), None)
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn openai_response_schema_snapshots_exclude_runtime_acceleration_metadata() {
        let usage = || UsageInfo {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        };
        let completion = serde_json::to_value(CompletionResponse {
            id: "cmpl-test".to_string(),
            object: "text_completion",
            created: 123,
            model: "fixture".to_string(),
            choices: vec![CompletionChoice {
                text: "hello".to_string(),
                index: 0,
                finish_reason: "stop",
            }],
            usage: usage(),
        })
        .unwrap();
        assert_eq!(
            completion,
            serde_json::json!({
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 123,
                "model": "fixture",
                "choices": [{
                    "text": "hello",
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }
            })
        );

        let chat = serde_json::to_value(ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 124,
            model: "fixture".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                finish_reason: "stop",
            }],
            usage: usage(),
        })
        .unwrap();
        assert_eq!(
            chat,
            serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 124,
                "model": "fixture",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }
            })
        );

        let reasoning_chat = serde_json::to_value(ChatCompletionResponse {
            id: "chatcmpl-thinking".to_string(),
            object: "chat.completion",
            created: 125,
            model: "fixture".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: "259.2".to_string(),
                    reasoning_content: Some("240 times 1.35 times 0.8".to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                },
                finish_reason: "stop",
            }],
            usage: usage(),
        })
        .unwrap();
        assert_eq!(reasoning_chat["choices"][0]["message"]["content"], "259.2");
        assert_eq!(
            reasoning_chat["choices"][0]["message"]["reasoning_content"],
            "240 times 1.35 times 0.8"
        );

        let models = serde_json::to_value(ModelList {
            object: "list",
            data: vec![ModelInfo {
                id: "fixture".to_string(),
                object: "model",
                created: 125,
                owned_by: "xeno",
            }],
        })
        .unwrap();
        assert_eq!(
            models,
            serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "fixture",
                    "object": "model",
                    "created": 125,
                    "owned_by": "xeno"
                }]
            })
        );

        for response in [&completion, &chat, &models] {
            let serialized = response.to_string();
            for forbidden in [
                "moe",
                "placement",
                "manifest",
                "gpu_expert_budget",
                "hybrid_state",
            ] {
                assert!(
                    !serialized.contains(forbidden),
                    "OpenAI schema snapshot leaked internal field {forbidden}"
                );
            }
        }
    }

    #[tokio::test]
    async fn external_runtime_status_is_explicit_and_redacts_credentials() {
        let state = empty_state();
        let config = ExternalOpenAiConfig::new(
            "http://127.0.0.1:8000/v1",
            Some("top-secret".to_string()),
            Some("external-model".to_string()),
            false,
            30,
        )
        .unwrap();
        activate_external_openai(&state, config).await;

        let status = runtime_status(State(state)).await.0;
        assert!(status.ready);
        assert_eq!(status.requested_backend, "external-openai");
        assert_eq!(status.active_backend.as_deref(), Some("external-openai"));
        assert_eq!(status.loaded_model.as_deref(), Some("external-model"));
        assert_eq!(
            status.external_base_url.as_deref(),
            Some("http://127.0.0.1:8000/v1")
        );
        assert_eq!(status.external_model.as_deref(), Some("external-model"));
        let serialized = serde_json::to_string(&status).unwrap();
        assert!(!serialized.contains("top-secret"));
    }

    #[tokio::test]
    async fn runtime_capabilities_reports_modalities_and_architectures() {
        let state = empty_state();
        let capabilities = runtime_capabilities(State(state)).await.0;
        assert_eq!(capabilities.object, "runtime.capabilities");
        assert!(capabilities.modalities.contains(&"text"));
        assert!(capabilities.supported_architectures.contains(&"qwen35"));
        assert!(capabilities.supported_architectures.contains(&"qwen35moe"));
        assert!(capabilities.supported_kv_cache_modes.contains(&"f32"));
        assert!(capabilities.supported_kv_cache_modes.contains(&"q4_0"));
        assert!(capabilities.mtp_speculative_supported);
        assert!(capabilities.prefix_caching_supported);
        assert!(capabilities.hybrid_moe_supported);
    }

    #[tokio::test]
    async fn runtime_preflight_reports_valid_fit_estimates() {
        let state = empty_state();
        let preflight = runtime_preflight(
            State(state),
            Json(RuntimePreflightRequest {
                model_path: None,
                size_bytes: Some(2_783_446_304),
                backend: None,
                is_moe: Some(false),
            }),
        )
        .await
        .0;
        assert_eq!(preflight.object, "runtime.preflight");
        assert!(preflight.estimated_vram_bytes > 2_783_446_304);
    }
}
