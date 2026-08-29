mod bundle_commands;
#[cfg(feature = "image-generation")]
mod image_commands;
mod process_memory;

use clap::{ArgGroup, Args, Parser, Subcommand};
use serde::Serialize;
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    sync::{Arc, Barrier, OnceLock},
    thread,
    time::{Duration, Instant},
};
use xrt_hub::{resolve_model_alias_or_path, DownloadProgress, ModelHub};
use xrt_openai::{ExternalOpenAiClient, ExternalOpenAiConfig};
use xrt_runtime::{
    BackendKind, GenerateRequest, GpuAllocationDelta, GpuAllocationStats, GpuResourceStatus,
    GpuTransferStats, KvCacheMode, PrefixCacheStatus, PromptSpan, PromptSpanKind, RequestScheduler,
    Runtime, SchedulerConfig, SchedulerStatus,
};
use xrt_tokenizer::ChatMessage;

use process_memory::{process_memory_status, ProcessMemoryStatus};

#[derive(Parser)]
#[command(name = "xrt", about = "xeno-rt CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Generate(GenerateArgs),
    Chat(ChatArgs),
    Bench(BenchArgs),
    Download(DownloadArgs),
    /// Install or import a complete immutable model bundle.
    Bundle(bundle_commands::BundleArgs),
    #[cfg(feature = "image-generation")]
    Image(image_commands::ImageArgs),
}

#[derive(Args)]
#[command(group(
    ArgGroup::new("model_source")
        .args(["model", "hf_repo"])
        .required(true)
))]
struct GenerateArgs {
    #[arg(long, conflicts_with_all = ["hf_repo", "hf_file"])]
    model: Option<String>,
    #[arg(long, requires = "hf_file", conflicts_with = "model")]
    hf_repo: Option<String>,
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    cache_policy: Option<String>,
    #[arg(long)]
    recent_window_tokens: Option<usize>,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, env = "XRT_BACKEND", default_value = "auto")]
    backend: String,
}

#[derive(Args)]
#[command(group(
    ArgGroup::new("chat_model_source")
        .args(["model", "hf_repo"])
        .required(true)
))]
struct ChatArgs {
    #[arg(long, conflicts_with_all = ["hf_repo", "hf_file"])]
    model: Option<String>,
    #[arg(long, requires = "hf_file", conflicts_with = "model")]
    hf_repo: Option<String>,
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    cache_policy: Option<String>,
    #[arg(long)]
    recent_window_tokens: Option<usize>,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, env = "XRT_BACKEND", default_value = "auto")]
    backend: String,
}

#[derive(Args)]
#[command(group(
    ArgGroup::new("bench_model_source")
        .args(["model", "hf_repo"])
))]
struct BenchArgs {
    #[arg(long, conflicts_with_all = ["hf_repo", "hf_file"])]
    model: Option<String>,
    #[arg(long, requires = "hf_file", conflicts_with = "model")]
    hf_repo: Option<String>,
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    system: Option<String>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        String::from("f32"),
        String::from("q8"),
        String::from("agent_adaptive"),
    ])]
    cache_modes: Vec<String>,
    #[arg(long, env = "XRT_BACKEND", value_delimiter = ',', default_values_t = vec![
        String::from("auto"),
    ])]
    backends: Vec<String>,
    #[arg(long, env = "XRT_EXTERNAL_BASE_URL")]
    external_base_url: Option<String>,
    #[arg(long, env = "XRT_EXTERNAL_API_KEY")]
    external_api_key: Option<String>,
    #[arg(long, env = "XRT_EXTERNAL_MODEL")]
    external_model: Option<String>,
    #[arg(long, default_value = "agent_adaptive")]
    cache_policy: String,
    #[arg(long)]
    recent_window_tokens: Option<usize>,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long, default_value_t = 128)]
    prefill_chunk_tokens: usize,
    #[arg(long, default_value_t = 8)]
    max_decode_turns_before_prefill: usize,
    #[arg(long, default_value_t = 4)]
    max_decode_batch_size: usize,
    #[arg(long, default_value_t = 20_000)]
    decode_batch_wait_micros: u64,
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    object: &'static str,
    model_path: String,
    quantization: Option<String>,
    prompt_tokens: Option<usize>,
    environment: BenchEnvironment,
    results: Vec<BenchResult>,
}

#[derive(Debug, Serialize)]
struct BenchEnvironment {
    os: &'static str,
    arch: &'static str,
    commit: Option<String>,
    cuda_feature_enabled: bool,
}

#[derive(Debug, Serialize)]
struct BenchResult {
    requested_backend: String,
    active_backend: Option<String>,
    model_name: Option<String>,
    model_architecture: Option<String>,
    cache_mode: Option<String>,
    cache_policy: Option<String>,
    repetition: Option<usize>,
    concurrency: Option<usize>,
    prompt_tokens: Option<usize>,
    output_tokens: Option<usize>,
    /// Backward-compatible name for time to first token.
    prefill_ms: Option<f64>,
    ttft_ms: Option<f64>,
    decode_tokens: Option<usize>,
    decode_ms: Option<f64>,
    /// Aggregate across all requests when `concurrency > 1`.
    decode_tok_s: Option<f64>,
    total_ms: Option<f64>,
    /// Total output throughput including time to first token.
    tok_s: Option<f64>,
    mean_request_ms: Option<f64>,
    max_request_ms: Option<f64>,
    preview: Option<String>,
    load_ms: f64,
    gpu_resource: Option<GpuResourceStatus>,
    prefix_cache: Option<PrefixCacheStatus>,
    scheduler: Option<SchedulerStatus>,
    host_memory: Option<ProcessMemoryStatus>,
    tracked_resident_vram_bytes: Option<u64>,
    transfer_delta: Option<GpuTransferStats>,
    allocation_delta: Option<GpuAllocationDelta>,
    error: Option<String>,
}

struct BenchMeasurement {
    prompt_tokens: Option<usize>,
    output_tokens: usize,
    /// Backward-compatible name for time to first token.
    prefill_ms: f64,
    decode_tokens: usize,
    decode_ms: f64,
    decode_tok_s: f64,
    total_ms: f64,
    tok_s: f64,
    mean_request_ms: f64,
    max_request_ms: f64,
    preview: String,
    gpu_resource: Option<GpuResourceStatus>,
    prefix_cache: Option<PrefixCacheStatus>,
    scheduler: Option<SchedulerStatus>,
    host_memory: Option<ProcessMemoryStatus>,
    tracked_resident_vram_bytes: Option<u64>,
    transfer_delta: Option<GpuTransferStats>,
    allocation_delta: Option<GpuAllocationDelta>,
    error: Option<String>,
}

struct SequenceMeasurement {
    output_tokens: usize,
    first_token: Option<Duration>,
    elapsed: Duration,
    output: String,
    gpu_resource: GpuResourceStatus,
    error: Option<String>,
}

struct ExternalSequenceMeasurement {
    prompt_tokens: Option<usize>,
    output_tokens: usize,
    first_token: Option<Duration>,
    elapsed: Duration,
    output: String,
    error: Option<String>,
}

const MAX_EXTERNAL_ERROR_BYTES: u64 = 1024 * 1024;
const MAX_EXTERNAL_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_EXTERNAL_SSE_EVENT_BYTES: usize = 2 * 1024 * 1024;

#[derive(Args)]
#[command(group(
    ArgGroup::new("download_target")
        .args(["file", "quantization", "bundle"])
        .required(true)
))]
struct DownloadArgs {
    #[arg(long, required_unless_present = "bundle", conflicts_with = "bundle")]
    repo: Option<String>,
    #[arg(long, conflicts_with = "quantization")]
    file: Option<String>,
    #[arg(long, conflicts_with = "file")]
    quantization: Option<String>,
    /// Install a complete immutable image bundle by catalog ID.
    #[arg(long, conflicts_with_all = ["repo", "file", "quantization"])]
    bundle: Option<String>,
    /// Override the model cache used for files or complete bundles.
    #[arg(long, env = "XRT_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    /// Override the directory containing audited bundle catalog manifests.
    #[arg(long, env = "XRT_BUNDLE_CATALOG_DIR", requires = "bundle")]
    catalog_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Generate(args) => run_generate(args)?,
        Command::Chat(args) => run_chat(args)?,
        Command::Bench(args) => run_bench(args)?,
        Command::Download(args) => run_download(args)?,
        Command::Bundle(args) => bundle_commands::run(args)?,
        #[cfg(feature = "image-generation")]
        Command::Image(args) => image_commands::run(args)?,
    }

    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = resolve_model_path(&args)?;
    let runtime = Runtime::load_with_backend(&model_path, parse_backend_arg(&args.backend)?)?;
    let mut session = runtime.new_session();
    let request = GenerateRequest {
        prompt: args.prompt,
        add_special_tokens: true,
        cache_policy: args.cache_policy,
        recent_window_tokens: args.recent_window_tokens,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        ..Default::default()
    };

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut first_token_time: Option<std::time::Duration> = None;
    let start = std::time::Instant::now();
    let token_count = session.generate_stream(&request, |piece| {
        if first_token_time.is_none() {
            first_token_time = Some(start.elapsed());
        }
        let _ = handle.write_all(piece.as_bytes());
        let _ = handle.flush();
    })?;
    let elapsed = start.elapsed();
    writeln!(handle)?;
    let prefill_ms = first_token_time
        .map(|t| t.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let decode_time = elapsed.as_secs_f64() - first_token_time.unwrap_or_default().as_secs_f64();
    let decode_tok_s = if decode_time > 0.0 && token_count > 1 {
        (token_count - 1) as f64 / decode_time
    } else {
        0.0
    };
    let total_tok_s = token_count as f64 / elapsed.as_secs_f64();
    eprintln!(
        "\n--- {token_count} tokens in {:.2}s | prefill {:.0}ms | decode {:.2} tok/s | total {:.2} tok/s ---",
        elapsed.as_secs_f64(),
        prefill_ms,
        decode_tok_s,
        total_tok_s,
    );
    Ok(())
}

fn run_chat(args: ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = resolve_chat_model_path(&args)?;
    let runtime = Runtime::load_with_backend(&model_path, parse_backend_arg(&args.backend)?)?;

    let has_template = runtime.tokenizer().chat_template().is_some();
    if has_template {
        eprintln!("Using model's chat template");
    } else {
        eprintln!("No chat template in model, using ChatML fallback");
    }

    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = &args.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    loop {
        eprint!("\n> ");
        io::stderr().flush()?;
        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });

        let prompt = runtime.tokenizer().format_chat(&messages, true)?;
        let request = GenerateRequest {
            prompt,
            add_special_tokens: false,
            cache_policy: args.cache_policy.clone(),
            recent_window_tokens: args.recent_window_tokens,
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
            ..Default::default()
        };

        let mut session = runtime.new_session();
        let mut handle = stdout.lock();
        let mut response = String::new();
        let start = std::time::Instant::now();
        let token_count = session.generate_stream(&request, |piece| {
            response.push_str(piece);
            let _ = handle.write_all(piece.as_bytes());
            let _ = handle.flush();
        })?;
        let elapsed = start.elapsed();
        writeln!(handle)?;
        eprintln!(
            "--- {token_count} tokens in {:.2}s ({:.1} tok/s) ---",
            elapsed.as_secs_f64(),
            token_count as f64 / elapsed.as_secs_f64(),
        );

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: response,
        });
    }

    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.repetitions == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--repetitions must be at least 1",
        )
        .into());
    }
    if args.concurrency == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--concurrency must be at least 1",
        )
        .into());
    }
    if args.concurrency > 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--concurrency above 8 is intentionally blocked until aggregate GPU KV budgeting is available",
        )
        .into());
    }
    let backends = args
        .backends
        .iter()
        .map(|backend| parse_backend_arg(backend))
        .collect::<Result<Vec<_>, _>>()?;
    let has_external_backend = backends.contains(&BackendKind::ExternalOpenAi);
    let has_local_backend = backends
        .iter()
        .any(|backend| *backend != BackendKind::ExternalOpenAi);
    if has_local_backend && args.model.is_none() && args.hf_repo.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local benchmark backends require --model or --hf-repo + --hf-file",
        )
        .into());
    }
    let scheduler_config = if has_local_backend {
        Some(
            SchedulerConfig::new(args.concurrency, 0, 32)?
                .with_execution_policy(
                    args.prefill_chunk_tokens,
                    args.max_decode_turns_before_prefill,
                )?
                .with_decode_batching(
                    args.max_decode_batch_size.min(args.concurrency.max(1)),
                    args.decode_batch_wait_micros,
                )?,
        )
    } else {
        None
    };
    let model_path = if has_local_backend {
        Some(resolve_bench_model_path(&args)?)
    } else {
        None
    };
    let external_config = if has_external_backend {
        let config = ExternalOpenAiConfig::from_env_with_overrides(
            args.external_base_url.as_deref(),
            args.external_api_key.as_deref(),
            args.external_model.as_deref(),
        )
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        if config.default_model().is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "external benchmarks require --external-model or XRT_EXTERNAL_MODEL",
            )
            .into());
        }
        Some(config)
    } else {
        None
    };
    let cache_modes = if has_local_backend {
        args.cache_modes
            .iter()
            .map(|mode_name| {
                KvCacheMode::parse(mode_name).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unsupported --cache-modes value: {mode_name}"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };

    let mut report = BenchReport {
        object: "xrt.bench",
        model_path: model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .or_else(|| {
                external_config
                    .as_ref()
                    .map(|config| config.base_url().to_string())
            })
            .unwrap_or_default(),
        quantization: model_path.as_deref().and_then(infer_quantization),
        prompt_tokens: None,
        environment: BenchEnvironment {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            commit: git_commit_hash(),
            cuda_feature_enabled: cfg!(feature = "cuda"),
        },
        results: Vec::new(),
    };

    if !args.json {
        println!("backend\trun\tconcurrency\tmode\tpolicy\toutput_tokens\tttft_ms\tdecode_tokens\tdecode_ms\tdecode_tok_s\ttotal_ms\ttotal_tok_s\tpreview");
    }

    for backend in backends {
        if backend == BackendKind::ExternalOpenAi {
            run_external_bench(
                &args,
                external_config
                    .as_ref()
                    .expect("external config should be validated before the backend loop"),
                &mut report,
            )?;
            continue;
        }
        let model_path = model_path
            .as_ref()
            .expect("local model path should be resolved before the backend loop");
        let load_start = std::time::Instant::now();
        let runtime = match Runtime::load_with_backend(model_path, backend) {
            Ok(runtime) => runtime,
            Err(err) => {
                let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                let error = err.to_string();
                if !args.json {
                    println!(
                        "{}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\tERROR: {}",
                        backend.as_str(),
                        error.replace(['\r', '\n', '\t'], " ")
                    );
                }
                report.results.push(BenchResult {
                    requested_backend: backend.as_str().to_string(),
                    active_backend: None,
                    model_name: None,
                    model_architecture: None,
                    cache_mode: None,
                    cache_policy: None,
                    repetition: None,
                    concurrency: Some(args.concurrency),
                    prompt_tokens: None,
                    output_tokens: None,
                    prefill_ms: None,
                    ttft_ms: None,
                    decode_tokens: None,
                    decode_ms: None,
                    decode_tok_s: None,
                    total_ms: None,
                    tok_s: None,
                    mean_request_ms: None,
                    max_request_ms: None,
                    preview: None,
                    load_ms,
                    gpu_resource: None,
                    prefix_cache: None,
                    scheduler: None,
                    host_memory: process_memory_status(),
                    tracked_resident_vram_bytes: None,
                    transfer_delta: None,
                    allocation_delta: None,
                    error: Some(error),
                });
                continue;
            }
        };
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        let mut messages = Vec::new();
        if let Some(system) = &args.system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: args.prompt.clone(),
        });
        let (prompt, prompt_spans) = build_chat_prompt_with_spans(&runtime, &messages)?;
        let prompt_tokens = runtime
            .tokenizer()
            .encode_with_options(&prompt, false, true)?
            .len();
        report.prompt_tokens.get_or_insert(prompt_tokens);

        if !args.json {
            eprintln!(
                "model={} backend_requested={} backend_active={} prompt_tokens={} cache_modes={} concurrency={}",
                runtime.model_name(),
                runtime.requested_backend().as_str(),
                runtime.active_backend().as_str(),
                prompt_tokens,
                args.cache_modes.join(","),
                args.concurrency
            );
        }

        for mode in &cache_modes {
            for repetition in 1..=args.repetitions {
                let request = GenerateRequest {
                    prompt: prompt.clone(),
                    add_special_tokens: false,
                    cache_policy: if *mode == KvCacheMode::AgentAdaptive {
                        Some(args.cache_policy.clone())
                    } else {
                        None
                    },
                    recent_window_tokens: args.recent_window_tokens,
                    prompt_spans: prompt_spans.clone(),
                    max_tokens: args.max_tokens,
                    temperature: args.temperature,
                    top_k: args.top_k,
                    top_p: args.top_p,
                    repetition_penalty: args.repetition_penalty,
                    seed: args.seed,
                    ..Default::default()
                };
                let policy_label = request.cache_policy.as_deref().unwrap_or("default_chat");
                let measurement = run_bench_measurement(
                    &runtime,
                    *mode,
                    &request,
                    args.concurrency,
                    scheduler_config
                        .expect("scheduler config should exist for local benchmark backends"),
                );
                if !args.json {
                    if let Some(error) = &measurement.error {
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{:.1}\t{:.2}\t{:.1}\t{:.2}\tERROR: {}",
                            runtime.active_backend().as_str(),
                            repetition,
                            args.concurrency,
                            mode.as_str(),
                            policy_label,
                            measurement.output_tokens,
                            measurement.prefill_ms,
                            measurement.decode_tokens,
                            measurement.decode_ms,
                            measurement.decode_tok_s,
                            measurement.total_ms,
                            measurement.tok_s,
                            error.replace(['\r', '\n', '\t'], " ")
                        );
                    } else {
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{:.1}\t{:.2}\t{:.1}\t{:.2}\t{}",
                            runtime.active_backend().as_str(),
                            repetition,
                            args.concurrency,
                            mode.as_str(),
                            policy_label,
                            measurement.output_tokens,
                            measurement.prefill_ms,
                            measurement.decode_tokens,
                            measurement.decode_ms,
                            measurement.decode_tok_s,
                            measurement.total_ms,
                            measurement.tok_s,
                            measurement.preview
                        );
                    }
                }
                report.results.push(BenchResult {
                    requested_backend: runtime.requested_backend().as_str().to_string(),
                    active_backend: Some(runtime.active_backend().as_str().to_string()),
                    model_name: Some(runtime.model_name().to_string()),
                    model_architecture: Some(runtime.model_architecture().to_string()),
                    cache_mode: Some(mode.as_str().to_string()),
                    cache_policy: Some(policy_label.to_string()),
                    repetition: Some(repetition),
                    concurrency: Some(args.concurrency),
                    prompt_tokens: Some(prompt_tokens),
                    output_tokens: Some(measurement.output_tokens),
                    prefill_ms: Some(measurement.prefill_ms),
                    ttft_ms: Some(measurement.prefill_ms),
                    decode_tokens: Some(measurement.decode_tokens),
                    decode_ms: Some(measurement.decode_ms),
                    decode_tok_s: Some(measurement.decode_tok_s),
                    total_ms: Some(measurement.total_ms),
                    tok_s: Some(measurement.tok_s),
                    mean_request_ms: Some(measurement.mean_request_ms),
                    max_request_ms: Some(measurement.max_request_ms),
                    preview: Some(measurement.preview),
                    load_ms,
                    gpu_resource: measurement.gpu_resource,
                    prefix_cache: measurement.prefix_cache,
                    scheduler: measurement.scheduler,
                    host_memory: measurement.host_memory,
                    tracked_resident_vram_bytes: measurement.tracked_resident_vram_bytes,
                    transfer_delta: measurement.transfer_delta,
                    allocation_delta: measurement.allocation_delta,
                    error: measurement.error,
                });
            }
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
}

fn run_external_bench(
    args: &BenchArgs,
    config: &ExternalOpenAiConfig,
    report: &mut BenchReport,
) -> Result<(), Box<dyn std::error::Error>> {
    let load_started = Instant::now();
    let client = ExternalOpenAiClient::new(config.clone());
    let load_ms = duration_ms(load_started.elapsed());
    let payload = external_chat_payload(args);
    let model_name = config.display_model().to_string();

    if !args.json {
        eprintln!(
            "model={} backend_requested=external-openai backend_active=external-openai endpoint={} concurrency={}",
            model_name,
            config.base_url(),
            args.concurrency
        );
    }

    for repetition in 1..=args.repetitions {
        let measurement = run_external_bench_measurement(&client, &payload, args.concurrency);
        if let Some(prompt_tokens) = measurement.prompt_tokens {
            report.prompt_tokens.get_or_insert(prompt_tokens);
        }
        if !args.json {
            if let Some(error) = &measurement.error {
                println!(
                    "external-openai\t{}\t{}\texternal\tupstream\t{}\t{:.1}\t{}\t{:.1}\t{:.2}\t{:.1}\t{:.2}\tERROR: {}",
                    repetition,
                    args.concurrency,
                    measurement.output_tokens,
                    measurement.prefill_ms,
                    measurement.decode_tokens,
                    measurement.decode_ms,
                    measurement.decode_tok_s,
                    measurement.total_ms,
                    measurement.tok_s,
                    error.replace(['\r', '\n', '\t'], " ")
                );
            } else {
                println!(
                    "external-openai\t{}\t{}\texternal\tupstream\t{}\t{:.1}\t{}\t{:.1}\t{:.2}\t{:.1}\t{:.2}\t{}",
                    repetition,
                    args.concurrency,
                    measurement.output_tokens,
                    measurement.prefill_ms,
                    measurement.decode_tokens,
                    measurement.decode_ms,
                    measurement.decode_tok_s,
                    measurement.total_ms,
                    measurement.tok_s,
                    measurement.preview
                );
            }
        }
        report.results.push(BenchResult {
            requested_backend: BackendKind::ExternalOpenAi.as_str().to_string(),
            active_backend: Some(BackendKind::ExternalOpenAi.as_str().to_string()),
            model_name: Some(model_name.clone()),
            model_architecture: None,
            cache_mode: None,
            cache_policy: None,
            repetition: Some(repetition),
            concurrency: Some(args.concurrency),
            prompt_tokens: measurement.prompt_tokens,
            output_tokens: Some(measurement.output_tokens),
            prefill_ms: Some(measurement.prefill_ms),
            ttft_ms: Some(measurement.prefill_ms),
            decode_tokens: Some(measurement.decode_tokens),
            decode_ms: Some(measurement.decode_ms),
            decode_tok_s: Some(measurement.decode_tok_s),
            total_ms: Some(measurement.total_ms),
            tok_s: Some(measurement.tok_s),
            mean_request_ms: Some(measurement.mean_request_ms),
            max_request_ms: Some(measurement.max_request_ms),
            preview: Some(measurement.preview),
            load_ms,
            gpu_resource: None,
            prefix_cache: None,
            scheduler: None,
            host_memory: measurement.host_memory,
            tracked_resident_vram_bytes: measurement.tracked_resident_vram_bytes,
            transfer_delta: measurement.transfer_delta,
            allocation_delta: measurement.allocation_delta,
            error: measurement.error,
        });
    }
    Ok(())
}

fn external_chat_payload(args: &BenchArgs) -> serde_json::Value {
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": args.prompt,
    }));
    let mut payload = serde_json::json!({
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(seed) = args.seed {
        payload
            .as_object_mut()
            .expect("benchmark payload should be an object")
            .insert("seed".to_string(), serde_json::json!(seed));
    }
    payload
}

fn run_external_bench_measurement(
    client: &ExternalOpenAiClient,
    payload: &serde_json::Value,
    concurrency: usize,
) -> BenchMeasurement {
    if concurrency == 1 {
        let measurement = run_external_sequence(client, payload.clone());
        let elapsed = measurement.elapsed;
        return aggregate_external_measurements(vec![(0, measurement)], elapsed);
    }

    let ready_barrier = Arc::new(Barrier::new(concurrency + 1));
    let start_barrier = Arc::new(Barrier::new(concurrency + 1));
    let start_epoch = Arc::new(OnceLock::<Instant>::new());
    let mut workers = Vec::with_capacity(concurrency);
    for sequence_index in 0..concurrency {
        let client = client.clone();
        let payload = payload.clone();
        let ready_barrier = ready_barrier.clone();
        let start_barrier = start_barrier.clone();
        let start_epoch = start_epoch.clone();
        workers.push(thread::spawn(move || {
            ready_barrier.wait();
            start_barrier.wait();
            let started = *start_epoch
                .get()
                .expect("external benchmark start epoch must be published before release");
            (
                sequence_index,
                run_external_sequence_from(&client, payload, started),
            )
        }));
    }

    ready_barrier.wait();
    let wall_started = Instant::now();
    start_epoch
        .set(wall_started)
        .expect("external benchmark start epoch is written once");
    start_barrier.wait();
    let mut measurements = Vec::with_capacity(concurrency);
    for worker in workers {
        match worker.join() {
            Ok(measurement) => measurements.push(measurement),
            Err(_) => measurements.push((
                measurements.len(),
                ExternalSequenceMeasurement {
                    prompt_tokens: None,
                    output_tokens: 0,
                    first_token: None,
                    elapsed: Duration::ZERO,
                    output: String::new(),
                    error: Some("external benchmark worker panicked".to_string()),
                },
            )),
        }
    }
    aggregate_external_measurements(measurements, wall_started.elapsed())
}

fn run_external_sequence(
    client: &ExternalOpenAiClient,
    payload: serde_json::Value,
) -> ExternalSequenceMeasurement {
    run_external_sequence_from(client, payload, Instant::now())
}

fn run_external_sequence_from(
    client: &ExternalOpenAiClient,
    payload: serde_json::Value,
    started: Instant,
) -> ExternalSequenceMeasurement {
    let response = match client.post_json("chat/completions", payload, "text/event-stream") {
        Ok(response) => response,
        Err(error) => {
            return ExternalSequenceMeasurement {
                prompt_tokens: None,
                output_tokens: 0,
                first_token: None,
                elapsed: started.elapsed(),
                output: String::new(),
                error: Some(error.to_string()),
            }
        }
    };
    if !(200..300).contains(&response.status()) {
        let status = response.status();
        let body = read_external_error_body(response.into_reader());
        return ExternalSequenceMeasurement {
            prompt_tokens: None,
            output_tokens: 0,
            first_token: None,
            elapsed: started.elapsed(),
            output: String::new(),
            error: Some(format!(
                "external OpenAI returned HTTP {status}: {}",
                output_preview(&body)
            )),
        };
    }
    if !response
        .content_type()
        .to_ascii_lowercase()
        .starts_with("text/event-stream")
    {
        let content_type = response.content_type().to_string();
        return ExternalSequenceMeasurement {
            prompt_tokens: None,
            output_tokens: 0,
            first_token: None,
            elapsed: started.elapsed(),
            output: String::new(),
            error: Some(format!(
                "external OpenAI returned content type `{content_type}` instead of text/event-stream"
            )),
        };
    }

    let mut reader = BufReader::new(response.into_reader());
    let mut line = Vec::new();
    let mut data_lines = Vec::new();
    let mut data_bytes = 0usize;
    let mut prompt_tokens = None;
    let mut completion_tokens = None;
    let mut content_chunks = 0usize;
    let mut first_token = None;
    let mut output = String::new();
    let mut saw_done = false;
    let mut error = None;

    loop {
        match read_external_sse_line(&mut reader, &mut line) {
            Ok(0) => {
                if !data_lines.is_empty() {
                    match apply_external_sse_event(
                        &data_lines.join("\n"),
                        started,
                        &mut first_token,
                        &mut prompt_tokens,
                        &mut completion_tokens,
                        &mut content_chunks,
                        &mut output,
                    ) {
                        Ok(done) => saw_done = done,
                        Err(message) => error = Some(message),
                    }
                }
                break;
            }
            Ok(_) => {}
            Err(read_error) => {
                error = Some(format!("failed to read external OpenAI SSE: {read_error}"));
                break;
            }
        }
        let line = match std::str::from_utf8(&line) {
            Ok(line) => line.trim_end_matches(['\r', '\n']),
            Err(utf8_error) => {
                error = Some(format!(
                    "external OpenAI returned non-UTF8 SSE data: {utf8_error}"
                ));
                break;
            }
        };
        if line.is_empty() {
            if data_lines.is_empty() {
                continue;
            }
            match apply_external_sse_event(
                &data_lines.join("\n"),
                started,
                &mut first_token,
                &mut prompt_tokens,
                &mut completion_tokens,
                &mut content_chunks,
                &mut output,
            ) {
                Ok(done) => {
                    data_lines.clear();
                    data_bytes = 0;
                    if done {
                        saw_done = true;
                        break;
                    }
                }
                Err(message) => {
                    error = Some(message);
                    break;
                }
            }
        } else if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim_start();
            data_bytes = data_bytes.saturating_add(data.len());
            if data_bytes > MAX_EXTERNAL_SSE_EVENT_BYTES {
                error = Some(format!(
                    "external OpenAI SSE event exceeds {MAX_EXTERNAL_SSE_EVENT_BYTES} bytes"
                ));
                break;
            }
            data_lines.push(data.to_string());
        }
    }

    if error.is_none() && !saw_done {
        error = Some("external OpenAI SSE ended before `[DONE]`".to_string());
    }
    ExternalSequenceMeasurement {
        prompt_tokens,
        output_tokens: completion_tokens.unwrap_or(content_chunks),
        first_token,
        elapsed: started.elapsed(),
        output,
        error,
    }
}

fn read_external_sse_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<usize> {
    line.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(line.len());
        }
        let read = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(read) > MAX_EXTERNAL_SSE_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("external OpenAI SSE line exceeds {MAX_EXTERNAL_SSE_LINE_BYTES} bytes"),
            ));
        }
        line.extend_from_slice(&available[..read]);
        reader.consume(read);
        if line.last() == Some(&b'\n') {
            return Ok(line.len());
        }
    }
}

fn apply_external_sse_event(
    data: &str,
    started: Instant,
    first_token: &mut Option<Duration>,
    prompt_tokens: &mut Option<usize>,
    completion_tokens: &mut Option<usize>,
    content_chunks: &mut usize,
    output: &mut String,
) -> Result<bool, String> {
    if data.trim() == "[DONE]" {
        return Ok(true);
    }
    let event: serde_json::Value = serde_json::from_str(data)
        .map_err(|error| format!("external OpenAI returned invalid SSE JSON: {error}"))?;
    if event
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
        && first_token.is_none()
    {
        *first_token = Some(started.elapsed());
    }
    if let Some(usage) = event.get("usage") {
        *prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .or(*prompt_tokens);
        *completion_tokens = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .or(*completion_tokens);
    }
    let content = event
        .pointer("/choices/0/delta/content")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            event
                .pointer("/choices/0/text")
                .and_then(serde_json::Value::as_str)
        });
    if let Some(content) = content.filter(|content| !content.is_empty()) {
        *content_chunks = content_chunks.saturating_add(1);
        output.push_str(content);
    }
    Ok(false)
}

fn aggregate_external_measurements(
    mut sequences: Vec<(usize, ExternalSequenceMeasurement)>,
    wall_elapsed: Duration,
) -> BenchMeasurement {
    sequences.sort_by_key(|(sequence_index, _)| *sequence_index);
    let output_tokens = sequences
        .iter()
        .map(|(_, measurement)| measurement.output_tokens)
        .sum();
    let first_token_samples = sequences
        .iter()
        .filter_map(|(_, measurement)| measurement.first_token)
        .collect::<Vec<_>>();
    let request_samples = sequences
        .iter()
        .map(|(_, measurement)| measurement.elapsed)
        .collect::<Vec<_>>();
    let prompt_token_values = sequences
        .iter()
        .filter_map(|(_, measurement)| measurement.prompt_tokens)
        .collect::<Vec<_>>();
    let prompt_tokens = prompt_token_values.first().copied();
    let decode = aggregate_decode_metrics(sequences.iter().map(|(_, measurement)| {
        (
            measurement.output_tokens,
            measurement.first_token,
            measurement.elapsed,
        )
    }));
    let mut errors = sequences
        .iter()
        .filter_map(|(sequence_index, measurement)| {
            measurement
                .error
                .as_ref()
                .map(|error| format!("sequence {sequence_index}: {error}"))
        })
        .collect::<Vec<_>>();
    if prompt_token_values
        .iter()
        .any(|value| Some(*value) != prompt_tokens)
    {
        errors.push("external sequences reported inconsistent prompt token counts".to_string());
    }
    let preview = sequences
        .first()
        .map(|(_, measurement)| output_preview(&measurement.output))
        .unwrap_or_default();

    BenchMeasurement {
        prompt_tokens,
        output_tokens,
        prefill_ms: mean_duration_ms(&first_token_samples),
        decode_tokens: decode.tokens,
        decode_ms: duration_ms(decode.elapsed),
        decode_tok_s: decode.tok_s,
        total_ms: duration_ms(wall_elapsed),
        tok_s: tokens_per_second(output_tokens, wall_elapsed),
        mean_request_ms: mean_duration_ms(&request_samples),
        max_request_ms: request_samples
            .iter()
            .copied()
            .max()
            .map(duration_ms)
            .unwrap_or(0.0),
        preview,
        gpu_resource: None,
        prefix_cache: None,
        scheduler: None,
        host_memory: process_memory_status(),
        tracked_resident_vram_bytes: None,
        transfer_delta: None,
        allocation_delta: None,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn read_external_error_body(reader: impl Read) -> String {
    let mut reader = reader.take(MAX_EXTERNAL_ERROR_BYTES + 1);
    let mut body = String::new();
    match reader.read_to_string(&mut body) {
        Ok(_) if body.len() as u64 <= MAX_EXTERNAL_ERROR_BYTES => body,
        Ok(_) => format!("response exceeded {MAX_EXTERNAL_ERROR_BYTES} bytes"),
        Err(error) => format!("failed to read response body: {error}"),
    }
}

fn run_bench_measurement(
    runtime: &Arc<Runtime>,
    cache_mode: KvCacheMode,
    request: &GenerateRequest,
    concurrency: usize,
    scheduler_config: SchedulerConfig,
) -> BenchMeasurement {
    if concurrency == 1 {
        return run_single_bench_measurement(runtime, cache_mode, request);
    }
    run_concurrent_bench_measurement(runtime, cache_mode, request, concurrency, scheduler_config)
}

fn run_single_bench_measurement(
    runtime: &Arc<Runtime>,
    cache_mode: KvCacheMode,
    request: &GenerateRequest,
) -> BenchMeasurement {
    let transfer_before = runtime.gpu_transfer_stats();
    let allocation_before = runtime.gpu_allocation_stats();
    runtime.reset_gpu_allocation_peak();
    let mut session = runtime.new_session_with_cache_mode(cache_mode);
    let mut emitted_pieces = 0usize;
    let mut first_token = None;
    let mut output = String::new();
    let started = Instant::now();
    let result = session.generate_stream(request, |piece| {
        if first_token.is_none() {
            first_token = Some(started.elapsed());
        }
        emitted_pieces += 1;
        output.push_str(piece);
    });
    let elapsed = started.elapsed();
    let output_tokens = result.as_ref().copied().unwrap_or(emitted_pieces);
    let total_ms = duration_ms(elapsed);
    let decode = aggregate_decode_metrics([(output_tokens, first_token, elapsed)]);
    let gpu_resource = session.gpu_resource_status();
    let tracked_resident_vram_bytes = tracked_resident_vram_bytes(&gpu_resource);
    drop(session);
    let transfer_delta = gpu_transfer_delta(runtime, transfer_before);
    let allocation_delta = gpu_allocation_delta(runtime, allocation_before);

    BenchMeasurement {
        prompt_tokens: None,
        output_tokens,
        prefill_ms: first_token.map(duration_ms).unwrap_or(0.0),
        decode_tokens: decode.tokens,
        decode_ms: duration_ms(decode.elapsed),
        decode_tok_s: decode.tok_s,
        total_ms,
        tok_s: tokens_per_second(output_tokens, elapsed),
        mean_request_ms: total_ms,
        max_request_ms: total_ms,
        preview: output_preview(&output),
        gpu_resource: Some(gpu_resource),
        prefix_cache: Some(runtime.prefix_cache_status()),
        scheduler: None,
        host_memory: process_memory_status(),
        tracked_resident_vram_bytes,
        transfer_delta,
        allocation_delta,
        error: result.err().map(|err| err.to_string()),
    }
}

fn run_concurrent_bench_measurement(
    runtime: &Arc<Runtime>,
    cache_mode: KvCacheMode,
    request: &GenerateRequest,
    concurrency: usize,
    scheduler_config: SchedulerConfig,
) -> BenchMeasurement {
    let transfer_before = runtime.gpu_transfer_stats();
    let allocation_before = runtime.gpu_allocation_stats();
    runtime.reset_gpu_allocation_peak();
    let scheduler = Arc::new(RequestScheduler::new(scheduler_config));
    scheduler.configure_kv_budget(runtime.gpu_resource_status().kv_budget_bytes);
    let ready_barrier = Arc::new(Barrier::new(concurrency + 1));
    let start_barrier = Arc::new(Barrier::new(concurrency + 1));
    let start_epoch = Arc::new(OnceLock::<Instant>::new());
    let mut workers = Vec::with_capacity(concurrency);

    for sequence_index in 0..concurrency {
        let runtime = runtime.clone();
        let request = request.clone();
        let scheduler = scheduler.clone();
        let ready_barrier = ready_barrier.clone();
        let start_barrier = start_barrier.clone();
        let start_epoch = start_epoch.clone();
        workers.push(thread::spawn(move || {
            let mut session = runtime.new_session_with_cache_mode(cache_mode);
            let mut emitted_pieces = 0usize;
            let mut first_token = None;
            let mut output = String::new();
            ready_barrier.wait();
            start_barrier.wait();
            let started = *start_epoch
                .get()
                .expect("local benchmark start epoch must be published before release");
            let result = session.generate_stream_scheduled(&request, &scheduler, |piece| {
                if first_token.is_none() {
                    first_token = Some(started.elapsed());
                }
                emitted_pieces += 1;
                output.push_str(piece);
            });
            let elapsed = started.elapsed();
            let output_tokens = result.as_ref().copied().unwrap_or(emitted_pieces);
            (
                sequence_index,
                SequenceMeasurement {
                    output_tokens,
                    first_token,
                    elapsed,
                    output,
                    gpu_resource: session.gpu_resource_status(),
                    error: result.err().map(|err| err.to_string()),
                },
            )
        }));
    }

    ready_barrier.wait();
    let wall_started = Instant::now();
    start_epoch
        .set(wall_started)
        .expect("local benchmark start epoch is written once");
    start_barrier.wait();
    let mut sequences = Vec::with_capacity(concurrency);
    let mut errors = Vec::new();
    for worker in workers {
        match worker.join() {
            Ok((sequence_index, measurement)) => {
                if let Some(error) = &measurement.error {
                    errors.push(format!("sequence {sequence_index}: {error}"));
                }
                sequences.push((sequence_index, measurement));
            }
            Err(_) => errors.push("concurrent benchmark worker panicked".to_string()),
        }
    }
    let wall_elapsed = wall_started.elapsed();
    sequences.sort_by_key(|(sequence_index, _)| *sequence_index);

    let output_tokens = sequences
        .iter()
        .map(|(_, measurement)| measurement.output_tokens)
        .sum();
    let first_token_samples = sequences
        .iter()
        .filter_map(|(_, measurement)| measurement.first_token)
        .collect::<Vec<_>>();
    let request_samples = sequences
        .iter()
        .map(|(_, measurement)| measurement.elapsed)
        .collect::<Vec<_>>();
    let decode = aggregate_decode_metrics(sequences.iter().map(|(_, measurement)| {
        (
            measurement.output_tokens,
            measurement.first_token,
            measurement.elapsed,
        )
    }));
    let prefill_ms = mean_duration_ms(&first_token_samples);
    let mean_request_ms = mean_duration_ms(&request_samples);
    let max_request_ms = request_samples
        .iter()
        .copied()
        .max()
        .map(duration_ms)
        .unwrap_or(0.0);
    let preview = sequences
        .first()
        .map(|(_, measurement)| output_preview(&measurement.output))
        .unwrap_or_default();
    let mut gpu_resource = aggregate_gpu_resource_status(&sequences);
    if let Some(status) = gpu_resource.as_mut() {
        status.transfer_totals = runtime.gpu_transfer_stats();
    }
    let tracked_resident_vram_bytes = gpu_resource.as_ref().and_then(tracked_resident_vram_bytes);
    let transfer_delta = gpu_transfer_delta(runtime, transfer_before);
    let allocation_delta = gpu_allocation_delta(runtime, allocation_before);

    BenchMeasurement {
        prompt_tokens: None,
        output_tokens,
        prefill_ms,
        decode_tokens: decode.tokens,
        decode_ms: duration_ms(decode.elapsed),
        decode_tok_s: decode.tok_s,
        total_ms: duration_ms(wall_elapsed),
        tok_s: tokens_per_second(output_tokens, wall_elapsed),
        mean_request_ms,
        max_request_ms,
        preview,
        gpu_resource,
        prefix_cache: Some(runtime.prefix_cache_status()),
        scheduler: Some(scheduler.status()),
        host_memory: process_memory_status(),
        tracked_resident_vram_bytes,
        transfer_delta,
        allocation_delta,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn aggregate_gpu_resource_status(
    sequences: &[(usize, SequenceMeasurement)],
) -> Option<GpuResourceStatus> {
    let mut status = sequences.first()?.1.gpu_resource.clone();
    status.kv_allocated_bytes = sequences
        .iter()
        .map(|(_, measurement)| measurement.gpu_resource.kv_allocated_bytes)
        .fold(0u64, u64::saturating_add);
    status.scratch_allocated_bytes = sequences
        .iter()
        .map(|(_, measurement)| measurement.gpu_resource.scratch_allocated_bytes)
        .fold(0u64, u64::saturating_add);
    status.tracked_allocated_bytes = status
        .model_weight_bytes
        .saturating_add(status.kv_allocated_bytes)
        .saturating_add(status.scratch_allocated_bytes);
    status.device_used_vram_bytes = sequences
        .iter()
        .filter_map(|(_, measurement)| measurement.gpu_resource.device_used_vram_bytes)
        .max();
    status.free_vram_bytes = sequences
        .iter()
        .filter_map(|(_, measurement)| measurement.gpu_resource.free_vram_bytes)
        .min();
    status.active_sessions = sequences
        .iter()
        .map(|(_, measurement)| measurement.gpu_resource.active_sessions)
        .max()
        .unwrap_or(0);
    Some(status)
}

fn tracked_resident_vram_bytes(status: &GpuResourceStatus) -> Option<u64> {
    status
        .cuda_available
        .then_some(status.tracked_allocated_bytes)
}

fn gpu_transfer_delta(
    runtime: &Runtime,
    before: Option<GpuTransferStats>,
) -> Option<GpuTransferStats> {
    before
        .zip(runtime.gpu_transfer_stats())
        .map(|(before, after)| after.saturating_sub(&before))
}

fn gpu_allocation_delta(
    runtime: &Runtime,
    before: Option<GpuAllocationStats>,
) -> Option<GpuAllocationDelta> {
    before
        .zip(runtime.gpu_allocation_stats())
        .map(|(before, after)| GpuAllocationDelta::between(&before, &after))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn mean_duration_ms(samples: &[Duration]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().copied().map(duration_ms).sum::<f64>() / samples.len() as f64
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DecodeMetrics {
    tokens: usize,
    elapsed: Duration,
    tok_s: f64,
}

/// Measure the shared decode window after the first emitted token.
///
/// Concurrent benchmark workers are released by the same barrier. The window
/// starts at the earliest worker's first token, ends at the latest completing
/// worker, and counts every token after each request's first token. This keeps
/// aggregate decode throughput separate from TTFT and total-output throughput.
fn aggregate_decode_metrics<I>(sequences: I) -> DecodeMetrics
where
    I: IntoIterator<Item = (usize, Option<Duration>, Duration)>,
{
    let mut tokens = 0usize;
    let mut earliest_first = None::<Duration>;
    let mut latest_finish = None::<Duration>;
    for (output_tokens, first_token, elapsed) in sequences {
        let decode_tokens = output_tokens.saturating_sub(1);
        if decode_tokens == 0 {
            continue;
        }
        let Some(first_token) = first_token else {
            continue;
        };
        tokens = tokens.saturating_add(decode_tokens);
        earliest_first = Some(
            earliest_first
                .map(|earliest| earliest.min(first_token))
                .unwrap_or(first_token),
        );
        latest_finish = Some(
            latest_finish
                .map(|latest| latest.max(elapsed))
                .unwrap_or(elapsed),
        );
    }

    let elapsed = earliest_first
        .zip(latest_finish)
        .map(|(first, finish)| finish.saturating_sub(first))
        .unwrap_or(Duration::ZERO);
    DecodeMetrics {
        tokens,
        elapsed,
        tok_s: tokens_per_second(tokens, elapsed),
    }
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

fn output_preview(output: &str) -> String {
    output
        .replace(['\r', '\n', '\t'], " ")
        .chars()
        .take(80)
        .collect()
}

fn parse_backend_arg(value: &str) -> Result<BackendKind, Box<dyn std::error::Error>> {
    BackendKind::parse(value)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported backend value: {value}"),
            )
        })
        .map_err(Into::into)
}

fn git_commit_hash() -> Option<String> {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn infer_quantization(model_path: &Path) -> Option<String> {
    let name = model_path
        .file_name()?
        .to_string_lossy()
        .to_ascii_lowercase();
    for quant in [
        "q2_k", "q3_k", "q4_k_m", "q4_k_s", "q4_k", "q5_k_m", "q5_k_s", "q5_k", "q6_k", "q8_0",
        "f16", "bf16", "f32",
    ] {
        if name.contains(quant) {
            return Some(quant.to_ascii_uppercase());
        }
    }
    None
}

fn resolve_chat_model_path(args: &ChatArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(model) = &args.model {
        return Ok(resolve_model_alias_or_path(model));
    }
    let repo = required_value(args.hf_repo.as_deref(), "--hf-repo")?;
    let file = required_value(args.hf_file.as_deref(), "--hf-file")?;
    let hub = ModelHub::new()?;
    let mut reporter = progress_reporter(repo, file);
    let model = hub.download_with_progress(repo, file, &mut reporter)?;
    finish_download(
        &model.repo_id,
        &model.filename,
        model.path.as_path(),
        model.size,
        model.was_cached,
    )?;
    Ok(model.path)
}

fn resolve_bench_model_path(args: &BenchArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(model) = &args.model {
        return Ok(resolve_model_alias_or_path(model));
    }
    let repo = required_value(args.hf_repo.as_deref(), "--hf-repo")?;
    let file = required_value(args.hf_file.as_deref(), "--hf-file")?;
    let hub = ModelHub::new()?;
    let mut reporter = progress_reporter(repo, file);
    let model = hub.download_with_progress(repo, file, &mut reporter)?;
    finish_download(
        &model.repo_id,
        &model.filename,
        model.path.as_path(),
        model.size,
        model.was_cached,
    )?;
    Ok(model.path)
}

fn run_download(args: DownloadArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(bundle) = args.bundle.as_deref() {
        #[cfg(feature = "image-generation")]
        return image_commands::download_bundle(
            bundle,
            args.cache_dir.as_deref(),
            args.catalog_dir.as_deref(),
        );
        #[cfg(not(feature = "image-generation"))]
        {
            let _ = bundle;
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "complete bundle downloads require the image-generation feature",
            )
            .into());
        }
    }
    let repo = required_value(args.repo.as_deref(), "--repo")?;
    let hub = match args.cache_dir {
        Some(cache_dir) => ModelHub::with_cache_dir(cache_dir)?,
        None => ModelHub::new()?,
    };
    let model = if let Some(file) = args.file {
        let mut reporter = progress_reporter(repo, &file);
        hub.download_with_progress(repo, &file, &mut reporter)?
    } else {
        let quantization = required_value(args.quantization.as_deref(), "--quantization")?;
        let mut reporter = progress_reporter(repo, quantization);
        hub.download_by_quantization(repo, quantization, &mut reporter)?
    };

    finish_download(
        &model.repo_id,
        &model.filename,
        model.path.as_path(),
        model.size,
        model.was_cached,
    )?;
    println!("{}", model.path.display());
    Ok(())
}

fn build_chat_prompt_with_spans(
    runtime: &Runtime,
    messages: &[ChatMessage],
) -> Result<(String, Vec<PromptSpan>), Box<dyn std::error::Error>> {
    let prompt = runtime.tokenizer().format_chat(messages, true)?;
    let mut spans = Vec::new();
    let mut previous_end = 0usize;
    for end_index in 0..messages.len() {
        let prefix = runtime
            .tokenizer()
            .format_chat(&messages[..=end_index], false)?;
        let prefix_end = runtime
            .tokenizer()
            .encode_with_options(&prefix, false, true)?
            .len();
        if prefix_end > previous_end {
            spans.push(PromptSpan {
                kind: match messages[end_index].role.as_str() {
                    "system" => PromptSpanKind::System,
                    "assistant" => PromptSpanKind::Assistant,
                    _ => PromptSpanKind::User,
                },
                token_start: previous_end,
                token_end: prefix_end,
            });
            previous_end = prefix_end;
        }
    }
    Ok((prompt, spans))
}

fn resolve_model_path(args: &GenerateArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(model) = &args.model {
        return Ok(resolve_model_alias_or_path(model));
    }

    let repo = required_value(args.hf_repo.as_deref(), "--hf-repo")?;
    let file = required_value(args.hf_file.as_deref(), "--hf-file")?;
    let hub = ModelHub::new()?;
    let mut reporter = progress_reporter(repo, file);
    let model = hub.download_with_progress(repo, file, &mut reporter)?;
    finish_download(
        &model.repo_id,
        &model.filename,
        model.path.as_path(),
        model.size,
        model.was_cached,
    )?;
    Ok(model.path)
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
    path: &Path,
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

fn required_value<'a>(
    value: Option<&'a str>,
    flag: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    value.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("missing {flag}")).into()
    })
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

#[cfg(test)]
mod tests {
    use super::{
        aggregate_decode_metrics, mean_duration_ms, output_preview, read_external_sse_line,
        run_external_sequence, tokens_per_second, Cli, Command, MAX_EXTERNAL_SSE_LINE_BYTES,
    };
    use clap::Parser;
    use serde_json::json;
    use std::{
        io::{Cursor, Read, Write},
        net::{TcpListener, TcpStream},
        sync::mpsc,
        thread,
        time::Duration,
    };
    use xrt_openai::{ExternalOpenAiClient, ExternalOpenAiConfig};

    #[test]
    fn concurrent_bench_helpers_report_aggregate_metrics() {
        let samples = [Duration::from_millis(10), Duration::from_millis(30)];
        assert_eq!(mean_duration_ms(&samples), 20.0);
        assert_eq!(tokens_per_second(20, Duration::from_millis(500)), 40.0);
        assert_eq!(tokens_per_second(20, Duration::ZERO), 0.0);
        assert_eq!(output_preview("one\ntwo\tthree"), "one two three");

        let decode = aggregate_decode_metrics([
            (
                8,
                Some(Duration::from_millis(1000)),
                Duration::from_millis(3000),
            ),
            (
                8,
                Some(Duration::from_millis(1200)),
                Duration::from_millis(3500),
            ),
        ]);
        assert_eq!(decode.tokens, 14);
        assert_eq!(decode.elapsed, Duration::from_millis(2500));
        assert!((decode.tok_s - 5.6).abs() < f64::EPSILON);

        let no_decode = aggregate_decode_metrics([
            (
                1,
                Some(Duration::from_millis(10)),
                Duration::from_millis(20),
            ),
            (8, None, Duration::from_millis(30)),
        ]);
        assert_eq!(no_decode.tokens, 0);
        assert_eq!(no_decode.elapsed, Duration::ZERO);
        assert_eq!(no_decode.tok_s, 0.0);
    }

    #[test]
    fn external_sse_line_reader_rejects_unbounded_lines() {
        let mut reader = Cursor::new(vec![b'a'; MAX_EXTERNAL_SSE_LINE_BYTES + 1]);
        let mut line = Vec::new();
        let error = read_external_sse_line(&mut reader, &mut line).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn external_bench_cli_accepts_proxy_without_local_model() {
        let cli = Cli::try_parse_from([
            "xrt",
            "bench",
            "--prompt",
            "Hello",
            "--backends",
            "external-openai",
            "--external-base-url",
            "http://127.0.0.1:8000/v1",
            "--external-model",
            "model",
        ])
        .unwrap();
        let Command::Bench(args) = cli.command else {
            panic!("expected bench command");
        };
        assert!(args.model.is_none());
        assert!(args.hf_repo.is_none());
        assert_eq!(args.backends, vec!["external-openai".to_string()]);
    }

    #[test]
    fn download_cli_accepts_complete_bundle_without_a_legacy_repo() {
        let cli =
            Cli::try_parse_from(["xrt", "download", "--bundle", "qwen-image-2512-q4_k_m"]).unwrap();
        let Command::Download(args) = cli.command else {
            panic!("expected download command");
        };
        assert_eq!(args.bundle.as_deref(), Some("qwen-image-2512-q4_k_m"));
        assert!(args.repo.is_none());
        assert!(Cli::try_parse_from([
            "xrt",
            "download",
            "--bundle",
            "qwen-image-2512-q4_k_m",
            "--repo",
            "Qwen/Qwen-Image-2512",
        ])
        .is_err());
    }

    #[cfg(feature = "image-generation")]
    #[test]
    fn image_bench_cli_accepts_a_retained_quality_output() {
        assert!(Cli::try_parse_from([
            "xrt",
            "image",
            "bench",
            "--model-path",
            "bundle",
            "--prompt",
            "fixture",
            "--size",
            "512x512",
            "--steps",
            "4",
            "--retain-first-output",
            "candidate.png",
            "--json",
        ])
        .is_ok());
    }

    #[test]
    fn external_bench_sequence_records_sse_usage_and_output() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
            "data: [DONE]\n\n"
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (request_tx, request_rx) = mpsc::channel();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            request_tx.send(read_http_request(&mut stream)).unwrap();
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        let config = ExternalOpenAiConfig::new(
            format!("http://{address}/v1"),
            None,
            Some("bench-model".to_string()),
            false,
            30,
        )
        .unwrap();
        let measurement = run_external_sequence(
            &ExternalOpenAiClient::new(config),
            json!({"messages": [], "stream": true}),
        );

        assert_eq!(measurement.prompt_tokens, Some(5));
        assert_eq!(measurement.output_tokens, 2);
        assert_eq!(measurement.output, "Hello world");
        assert!(measurement.first_token.is_some());
        assert!(measurement.error.is_none(), "{:?}", measurement.error);

        let request = request_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        let header_end = find_bytes(&request, b"\r\n\r\n").unwrap();
        let payload: serde_json::Value =
            serde_json::from_slice(&request[header_end + 4..]).unwrap();
        assert_eq!(payload["model"], "bench-model");
    }

    fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = find_bytes(&request, b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        request
    }

    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }
}
