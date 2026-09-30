//! Native model/native-dependency bundle installation; one shared installer.
use clap::{Args, Subcommand};
use std::{fs, io::Read, path::PathBuf};
use xrt_hub::{ArtifactManifest, ModelHub};

#[derive(Args)]
pub struct BundleArgs {
    #[arg(long, global = true)]
    cache: Option<PathBuf>,
    #[command(subcommand)]
    command: BundleCommand,
}

#[derive(Subcommand)]
enum BundleCommand {
    /// Plan a verified install; --confirm downloads, otherwise dry-run.
    Install {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        digest: String,
        /// Explicit reviewed source/redirect hosts, never inferred from a model.
        #[arg(long, required = true)]
        allowed_host: Vec<String>,
        #[arg(long, default_value_t = 16_000_000_000u64)]
        max_bytes: u64,
        #[arg(long)]
        confirm: bool,
        /// Host-owned cancellation marker, checked between network reads.
        #[arg(long)]
        cancel_file: Option<PathBuf>,
    },
    /// Import an already-downloaded directory through the identical hash gate.
    Import {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        digest: String,
        #[arg(long)]
        directory: PathBuf,
        #[arg(long, default_value_t = 16_000_000_000u64)]
        max_bytes: u64,
        #[arg(long)]
        confirm: bool,
    },
    /// Rehash every artifact of an installed bundle, without networking.
    Verify {
        id: String,
        #[arg(long)]
        digest: Option<String>,
    },
    /// Remove verified files only; refuses corruption/links and needs --confirm.
    Remove {
        id: String,
        #[arg(long)]
        digest: String,
        #[arg(long)]
        confirm: bool,
    },
    /// Resolve the current installed bundle; does not install or download.
    Path {
        id: String,
        #[arg(long)]
        digest: Option<String>,
    },
}

fn read_manifest(
    path: &PathBuf,
    expected: &str,
) -> Result<ArtifactManifest, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    let manifest = ArtifactManifest::parse(&bytes)?;
    if manifest.digest()? != expected {
        return Err("manifest digest differs from pinned expected digest".into());
    }
    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    manifest.check_runtime(env!("CARGO_PKG_VERSION"), &platform)?;
    Ok(manifest)
}

pub fn run(args: BundleArgs) -> Result<(), Box<dyn std::error::Error>> {
    let hub = match args.cache {
        Some(path) => ModelHub::with_cache_dir(path)?,
        None => ModelHub::new()?,
    };
    match args.command {
        BundleCommand::Install {
            manifest,
            digest,
            allowed_host,
            max_bytes,
            confirm,
            cancel_file,
        } => {
            let manifest = read_manifest(&manifest, &digest)?;
            let plan = manifest.install_plan(allowed_host, max_bytes)?;
            if !confirm {
                println!(
                    "{}",
                    serde_json::json!({"action":"plan","id":manifest.id,"digest":digest,"files":manifest.files.len(),"bytes":manifest.files.iter().map(|f|f.size_bytes).sum::<u64>()})
                );
                return Ok(());
            }
            let mut last = std::time::Instant::now() - std::time::Duration::from_secs(1);
            let installed=hub.install_bundle_resumable(&plan,||cancel_file.as_ref().is_some_and(|p|p.exists()),|p| {
                if p.artifact_downloaded != p.artifact_total && last.elapsed() < std::time::Duration::from_millis(250) { return; }
                last = std::time::Instant::now();
                eprintln!("{}",serde_json::json!({"event":"download","file":p.artifact_path,"file_bytes":p.artifact_downloaded,"downloaded":p.bundle_downloaded,"total":p.bundle_total}));
            })?;
            println!(
                "{}",
                serde_json::json!({"id":installed.id,"digest":installed.digest,"path":installed.path,"cached":installed.was_cached})
            );
        }
        BundleCommand::Import {
            manifest,
            digest,
            directory,
            max_bytes,
            confirm,
        } => {
            let manifest = read_manifest(&manifest, &digest)?;
            let plan = manifest.import_plan(max_bytes)?;
            if !confirm {
                println!(
                    "{}",
                    serde_json::json!({"action":"plan-import","id":manifest.id,"digest":digest,"source":directory})
                );
                return Ok(());
            }
            let installed = hub.import_bundle(directory, &plan)?;
            println!(
                "{}",
                serde_json::json!({"id":installed.id,"digest":installed.digest,"path":installed.path,"cached":installed.was_cached})
            );
        }
        BundleCommand::Verify { id, digest } => {
            let path = hub.resolve_installed_bundle(&id, digest.as_deref())?;
            let bytes = fs::read(path.join("xrt.bundle.json"))?;
            let manifest = ArtifactManifest::parse(&bytes)?;
            let actual = manifest.digest()?;
            if manifest.id != id
                || path.file_name().and_then(|s| s.to_str()) != Some(actual.as_str())
            {
                return Err("installed manifest identity mismatch".into());
            }
            manifest.verify_directory(&path)?;
            println!(
                "{}",
                serde_json::json!({"id":id,"digest":actual,"path":path,"verified":true})
            );
        }
        BundleCommand::Remove {
            id,
            digest,
            confirm,
        } => {
            let path = hub.resolve_installed_bundle(&id, Some(&digest))?;
            if confirm {
                hub.remove_artifact_bundle(&id, &digest)?;
            }
            println!(
                "{}",
                serde_json::json!({"id":id,"path":path,"removed":confirm})
            );
        }
        BundleCommand::Path { id, digest } => {
            let path = hub.resolve_installed_bundle(&id, digest.as_deref())?;
            println!("{}", serde_json::json!({"id":id,"path":path}));
        }
    }
    Ok(())
}
