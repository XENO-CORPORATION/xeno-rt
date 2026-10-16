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
#[derive(Args)]
struct PublisherProof {
    /// Detached Ed25519 authorization over the pinned manifest digest.
    #[arg(long, requires = "trusted_key")]
    signature: Option<PathBuf>,
    /// Public key provisioned through a trusted operator channel; never from a download.
    #[arg(long, requires = "signature")]
    trusted_key: Option<PathBuf>,
}

#[derive(Subcommand)]
enum BundleCommand {
    /// Validate a manifest and print its canonical content digest without installing.
    Digest {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Plan a verified install; --confirm downloads, otherwise dry-run.
    Install {
        #[command(flatten)]
        publisher: PublisherProof,
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
        #[command(flatten)]
        publisher: PublisherProof,
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
    /// Discard a corrupt/cancelled partial download so install can retry cleanly.
    DiscardPartial {
        id: String,
        #[arg(long)]
        digest: String,
        #[arg(long)]
        confirm: bool,
    },
}

fn read_manifest(
    path: &PathBuf,
    expected: &str,
    publisher: &PublisherProof,
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
    if let (Some(signature), Some(key)) = (&publisher.signature, &publisher.trusted_key) {
        let read = |path: &PathBuf| -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::new();
            fs::File::open(path)?.take(4097).read_to_end(&mut bytes)?;
            Ok(bytes)
        };
        let signer = xrt_hub::artifact_signature::verify(
            &manifest,
            expected,
            &read(signature)?,
            &read(key)?,
        )?;
        eprintln!(
            "{}",
            serde_json::json!({"event":"publisher_verified","key_id":signer,"manifest_digest":expected})
        );
    } else if publisher.signature.is_some() || publisher.trusted_key.is_some() {
        return Err("signature and trusted-key must be supplied together".into());
    }
    Ok(manifest)
}

pub fn run(args: BundleArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let BundleCommand::Digest { ref manifest } = args.command {
        let mut bytes = Vec::new();
        fs::File::open(manifest)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        let manifest = ArtifactManifest::parse(&bytes)?;
        println!(
            "{}",
            serde_json::json!({"id":manifest.id,"digest":manifest.digest()?})
        );
        return Ok(());
    }
    // Validate publisher authority before creating cache or staging directories.
    let verified_manifest = match &args.command {
        BundleCommand::Install {
            manifest,
            digest,
            publisher,
            ..
        }
        | BundleCommand::Import {
            manifest,
            digest,
            publisher,
            ..
        } => Some(read_manifest(manifest, digest, publisher)?),
        _ => None,
    };
    let hub = match args.cache {
        Some(path) => ModelHub::with_cache_dir(path)?,
        None => ModelHub::new()?,
    };
    match args.command {
        BundleCommand::Digest { .. } => {}
        BundleCommand::DiscardPartial {
            id,
            digest,
            confirm,
        } => {
            if confirm {
                hub.discard_partial_bundle(&id, &digest)?;
            }
            println!(
                "{}",
                serde_json::json!({"id":id,"digest":digest,"partial_discarded":confirm})
            );
        }
        BundleCommand::Install {
            publisher: _,
            manifest,
            digest,
            allowed_host,
            max_bytes,
            confirm,
            cancel_file,
        } => {
            let _ = manifest;
            let manifest = verified_manifest.ok_or("bundle authorization was not validated")?;
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
            publisher: _,
            manifest,
            digest,
            directory,
            max_bytes,
            confirm,
        } => {
            let _ = manifest;
            let manifest = verified_manifest.ok_or("bundle authorization was not validated")?;
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
