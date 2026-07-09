use clap::{ArgGroup, Args, Parser, Subcommand};
use serde::Serialize;
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
};
use xrt_hub::{resolve_model_alias_or_path, DownloadProgress, ModelHub};
use xrt_runtime::{BackendKind, GenerateRequest, KvCacheMode, PromptSpan, PromptSpanKind, Runtime};
use xrt_tokenizer::ChatMessage;

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
        .required(true)
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
    #[arg(long, default_value = "agent_adaptive")]
    cache_policy: String,
    #[arg(long)]
    recent_window_tokens: Option<usize>,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
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
    output_tokens: Option<usize>,
    prefill_ms: Option<f64>,
    total_ms: Option<f64>,
    tok_s: Option<f64>,
    preview: Option<String>,
    load_ms: f64,
    gpu_resource: Option<xrt_runtime::GpuResourceStatus>,
    error: Option<String>,
}

#[derive(Args)]
#[command(group(
    ArgGroup::new("download_target")
        .args(["file", "quantization"])
        .required(true)
))]
struct DownloadArgs {
    #[arg(long)]
    repo: String,
    #[arg(long, conflicts_with = "quantization")]
    file: Option<String>,
    #[arg(long, conflicts_with = "file")]
    quantization: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
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
    let model_path = resolve_bench_model_path(&args)?;
    let backends = args
        .backends
        .iter()
        .map(|backend| parse_backend_arg(backend))
        .collect::<Result<Vec<_>, _>>()?;
    let cache_modes = args
        .cache_modes
        .iter()
        .map(|mode_name| {
            KvCacheMode::parse(mode_name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported --cache-modes value: {mode_name}"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut report = BenchReport {
        object: "xrt.bench",
        model_path: model_path.display().to_string(),
        quantization: infer_quantization(&model_path),
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
        println!("backend\tmode\tpolicy\toutput_tokens\tprefill_ms\ttotal_ms\ttok_s\tpreview");
    }

    for backend in backends {
        let load_start = std::time::Instant::now();
        let runtime = match Runtime::load_with_backend(&model_path, backend) {
            Ok(runtime) => runtime,
            Err(err) => {
                let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                let error = err.to_string();
                if !args.json {
                    println!(
                        "{}\t-\t-\t-\t-\t-\t-\tERROR: {}",
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
                    output_tokens: None,
                    prefill_ms: None,
                    total_ms: None,
                    tok_s: None,
                    preview: None,
                    load_ms,
                    gpu_resource: None,
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
                "model={} backend_requested={} backend_active={} prompt_tokens={} cache_modes={}",
                runtime.model_name(),
                runtime.requested_backend().as_str(),
                runtime.active_backend().as_str(),
                prompt_tokens,
                args.cache_modes.join(",")
            );
        }

        for mode in &cache_modes {
            let mut session = runtime.new_session_with_cache_mode(*mode);
            let mut emitted_pieces = 0usize;
            let mut first_token_time: Option<std::time::Duration> = None;
            let mut output = String::new();
            let start = std::time::Instant::now();
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

            let generation_result = session.generate_stream(&request, |piece| {
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed());
                }
                emitted_pieces += 1;
                output.push_str(piece);
            });

            let elapsed = start.elapsed();
            let prefill_ms = first_token_time
                .map(|t| t.as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let token_count = generation_result
                .as_ref()
                .copied()
                .unwrap_or(emitted_pieces);
            let tok_s = if elapsed.as_secs_f64() > 0.0 {
                token_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            let preview = output.replace(['\r', '\n', '\t'], " ");
            let preview = preview.chars().take(80).collect::<String>();
            let policy_label = request.cache_policy.as_deref().unwrap_or("default_chat");
            let error = generation_result.err().map(|err| err.to_string());
            if !args.json {
                if let Some(error) = &error {
                    println!(
                        "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.2}\tERROR: {}",
                        runtime.active_backend().as_str(),
                        mode.as_str(),
                        policy_label,
                        token_count,
                        prefill_ms,
                        total_ms,
                        tok_s,
                        error.replace(['\r', '\n', '\t'], " ")
                    );
                } else {
                    println!(
                        "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.2}\t{}",
                        runtime.active_backend().as_str(),
                        mode.as_str(),
                        policy_label,
                        token_count,
                        prefill_ms,
                        total_ms,
                        tok_s,
                        preview
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
                output_tokens: Some(token_count),
                prefill_ms: Some(prefill_ms),
                total_ms: Some(total_ms),
                tok_s: Some(tok_s),
                preview: Some(preview),
                load_ms,
                gpu_resource: Some(session.gpu_resource_status()),
                error,
            });
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
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
    let hub = ModelHub::new()?;
    let model = if let Some(file) = args.file {
        let mut reporter = progress_reporter(&args.repo, &file);
        hub.download_with_progress(&args.repo, &file, &mut reporter)?
    } else {
        let quantization = required_value(args.quantization.as_deref(), "--quantization")?;
        let mut reporter = progress_reporter(&args.repo, quantization);
        hub.download_by_quantization(&args.repo, quantization, &mut reporter)?
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
