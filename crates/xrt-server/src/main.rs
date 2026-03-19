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
use clap::{ArgGroup, Parser};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    io::{self, Write as _},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{signal, sync::mpsc, task};
use tokio_stream::wrappers::UnboundedReceiverStream;
use xrt_hub::{DownloadProgress, ModelHub};
use xrt_runtime::{GenerateRequest, Runtime};
use xrt_tokenizer::ChatMessage as TemplateChatMessage;

#[derive(Parser)]
#[command(name = "xrt-server", about = "xeno-rt OpenAI-compatible server")]
#[command(group(
    ArgGroup::new("model_source")
        .args(["model", "hf_repo"])
        .required(true)
))]
struct Cli {
    /// Path to a local GGUF model file
    #[arg(long, conflicts_with_all = ["hf_repo", "hf_file"])]
    model: Option<String>,
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
}

#[derive(Clone)]
struct AppState {
    runtime: Arc<Runtime>,
}

// --- OpenAI-compatible request/response types ---

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
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
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    stream: Option<bool>,
    seed: Option<u64>,
    /// Tool definitions for function calling (accepted but not yet executed).
    #[serde(default)]
    #[allow(dead_code)]
    tools: Option<Vec<serde_json::Value>>,
    /// Tool choice strategy (accepted but not yet executed).
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let model_path = resolve_model_path(&cli)?;
    let runtime = Runtime::load(&model_path)?;
    let state = AppState { runtime };

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
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

/// Resolve model path from --model or --hf-repo/--hf-file flags,
/// downloading from HuggingFace if needed.
fn resolve_model_path(cli: &Cli) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(model) = &cli.model {
        return Ok(model.clone());
    }
    let repo = cli.hf_repo.as_deref().ok_or("missing --hf-repo")?;
    let file = cli.hf_file.as_deref().ok_or("missing --hf-file")?;
    let hub = ModelHub::new()?;
    let mut reporter = progress_reporter(repo, file);
    let model = hub.download_with_progress(repo, file, &mut reporter)?;
    finish_download(&model.repo_id, &model.filename, &model.path, model.size, model.was_cached)?;
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
    let model_name = state.runtime.model_name().to_string();
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: model_name,
            object: "model",
            created: unix_timestamp(),
            owned_by: "xeno-rt",
        }],
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
    let runtime = state.runtime.clone();
    let prompt_text = request.prompt.clone();
    let generate = request_to_generate_request(request.prompt.clone(), &request);

    // Count prompt tokens for usage info
    let prompt_tokens = runtime
        .tokenizer()
        .encode(&prompt_text)
        .map(|t| t.len())
        .unwrap_or(0);

    let text = task::spawn_blocking(move || {
        let mut session = runtime.new_session();
        session.generate(&generate)
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;

    let completion_tokens = state
        .runtime
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
            .unwrap_or_else(|| state.runtime.model_name().to_string()),
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
    let runtime = state.runtime.clone();
    let prompt = chat_prompt(&request.messages, &runtime);
    let generate = request_to_generate_request(prompt.clone(), &request);

    let prompt_tokens = runtime
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(0);

    let text = task::spawn_blocking(move || {
        let mut session = runtime.new_session();
        session.generate(&generate)
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;

    let completion_tokens = state
        .runtime
        .tokenizer()
        .encode(&text)
        .map(|t| t.len())
        .unwrap_or(0);

    let created = unix_timestamp();
    let response = ChatCompletionResponse {
        id: completion_id("chatcmpl"),
        object: "chat.completion",
        created,
        model: request
            .model
            .unwrap_or_else(|| state.runtime.model_name().to_string()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
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

async fn completion_stream(
    state: AppState,
    request: CompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let runtime = state.runtime.clone();
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| state.runtime.model_name().to_string());
    let generate = request_to_generate_request(request.prompt.clone(), &request);
    let id = completion_id("cmpl");
    let created = unix_timestamp();

    task::spawn_blocking(move || {
        let mut session = runtime.new_session();
        let result = session.generate_stream(&generate, |piece| {
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
                let _ = tx.send(Ok(Event::default().data(data)));
            }
        });

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
            let _ = tx.send(Ok(Event::default().data(data)));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });

    Ok(Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn chat_stream(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let runtime = state.runtime.clone();
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| state.runtime.model_name().to_string());
    let prompt = chat_prompt(&request.messages, &runtime);
    let generate = request_to_generate_request(prompt, &request);
    let id = completion_id("chatcmpl");
    let created = unix_timestamp();

    task::spawn_blocking(move || {
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
            let _ = tx.send(Ok(Event::default().data(data)));
        }

        let mut session = runtime.new_session();
        let result = session.generate_stream(&generate, |piece| {
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
                let _ = tx.send(Ok(Event::default().data(data)));
            }
        });

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
            let _ = tx.send(Ok(Event::default().data(data)));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });

    Ok(Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn request_to_generate_request<T>(prompt: String, request: &T) -> GenerateRequest
where
    T: RequestConfig,
{
    GenerateRequest {
        prompt,
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
    fn max_tokens(&self) -> Option<usize>;
    fn temperature(&self) -> Option<f32>;
    fn top_k(&self) -> Option<usize>;
    fn top_p(&self) -> Option<f32>;
    fn repetition_penalty(&self) -> Option<f32>;
    fn seed(&self) -> Option<u64>;
}

impl RequestConfig for CompletionRequest {
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

fn chat_prompt(messages: &[ChatMessage], runtime: &Runtime) -> String {
    let template_messages: Vec<TemplateChatMessage> = messages
        .iter()
        .map(|m| TemplateChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    match runtime.tokenizer().format_chat(&template_messages, true) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!("chat template render failed, using fallback: {e}");
            let mut prompt = String::new();
            for message in messages {
                prompt.push_str(&message.role.to_uppercase());
                prompt.push_str(": ");
                prompt.push_str(&message.content);
                prompt.push('\n');
            }
            prompt.push_str("ASSISTANT: ");
            prompt
        }
    }
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

fn internal_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
