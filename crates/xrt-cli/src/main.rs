use clap::{ArgGroup, Args, Parser, Subcommand};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
};
use xrt_hub::{DownloadProgress, ModelHub};
use xrt_runtime::{GenerateRequest, Runtime};

#[derive(Parser)]
#[command(name = "xrt", about = "xeno-rt CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Generate(GenerateArgs),
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
        Command::Download(args) => run_download(args)?,
    }

    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = resolve_model_path(&args)?;
    let runtime = Runtime::load(&model_path)?;
    let mut session = runtime.new_session();
    let request = GenerateRequest {
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut token_count = 0usize;
    let mut first_token_time: Option<std::time::Duration> = None;
    let start = std::time::Instant::now();
    session.generate_stream(&request, |piece| {
        if first_token_time.is_none() {
            first_token_time = Some(start.elapsed());
        }
        token_count += 1;
        let _ = handle.write_all(piece.as_bytes());
        let _ = handle.flush();
    })?;
    let elapsed = start.elapsed();
    writeln!(handle)?;
    let prefill_ms = first_token_time.map(|t| t.as_secs_f64() * 1000.0).unwrap_or(0.0);
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

fn resolve_model_path(args: &GenerateArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(model) = &args.model {
        return Ok(PathBuf::from(model));
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
