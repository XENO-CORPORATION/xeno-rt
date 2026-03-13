use clap::{ArgGroup, Args, Parser, Subcommand};
use std::io::{self, Write};
use xrt_hub::{DownloadProgress, ModelHub};

#[derive(Parser)]
#[command(name = "xtask", about = "xrt development tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Download(DownloadArgs),
    ListCached,
    CleanCache,
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
    let cli = Cli::parse();
    let hub = ModelHub::new()?;

    match cli.command {
        Command::Download(args) => run_download(&hub, args)?,
        Command::ListCached => run_list_cached(&hub)?,
        Command::CleanCache => run_clean_cache(&hub)?,
    }

    Ok(())
}

fn run_download(hub: &ModelHub, args: DownloadArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = if let Some(file) = args.file {
        let mut reporter = progress_reporter(&args.repo, &file);
        hub.download_with_progress(&args.repo, &file, &mut reporter)?
    } else {
        let quantization = args
            .quantization
            .as_deref()
            .ok_or("missing --quantization")?;
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

fn run_list_cached(hub: &ModelHub) -> Result<(), Box<dyn std::error::Error>> {
    let cached = hub.list_cached()?;
    if cached.is_empty() {
        println!("cache is empty");
        return Ok(());
    }

    for entry in cached {
        println!(
            "{}\t{}",
            format_bytes(entry.size),
            entry.relative_path.display()
        );
    }
    Ok(())
}

fn run_clean_cache(hub: &ModelHub) -> Result<(), Box<dyn std::error::Error>> {
    hub.clean_cache()?;
    println!("{}", hub.cache_dir().display());
    Ok(())
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
