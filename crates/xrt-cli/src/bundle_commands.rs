use clap::{Args, Subcommand};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io, path::PathBuf};
use xrt_hub::{
    BundleArtifact, BundleImportArtifact, BundleImportPlan, BundleInstallPlan, ModelHub,
};

const MAX_MANAGED_BUNDLE_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Args)]
pub(crate) struct BundleArgs {
    #[command(subcommand)]
    command: BundleCommand,
}
#[derive(Debug, Subcommand)]
enum BundleCommand {
    /// Download, verify, and atomically install every artifact in a manifest.
    Install(BundleInstallArgs),
    /// Verify and atomically import a pre-provisioned offline bundle.
    Import(BundleImportArgs),
}

#[derive(Debug, Args)]
struct BundleInstallArgs {
    /// Audited immutable bundle manifest.
    #[arg(long, value_name = "FILE")]
    manifest: PathBuf,
    /// Override the managed XRT model cache.
    #[arg(long, env = "XRT_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct BundleImportArgs {
    /// Audited immutable bundle manifest.
    #[arg(long, value_name = "FILE")]
    manifest: PathBuf,
    /// Directory containing all artifacts declared by the manifest.
    #[arg(long, value_name = "DIRECTORY")]
    source_dir: PathBuf,
    /// Override the managed XRT model cache.
    #[arg(long, env = "XRT_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifest {
    schema_version: u32,
    id: String,
    domain: String,
    model_id: String,
    revision: String,
    license: CatalogLicense,
    contract: serde_json::Value,
    artifacts: Vec<BundleArtifact>,
    allowed_hosts: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogLicense {
    spdx: String,
    notice: String,
    source: String,
}

pub(crate) fn run(args: BundleArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        BundleCommand::Install(args) => install(args),
        BundleCommand::Import(args) => import(args),
    }
}

fn install(args: BundleInstallArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (manifest, manifest_bytes, digest) = read_manifest(&args.manifest)?;
    let max_total = declared_total(&manifest)?;
    let plan = BundleInstallPlan::new(
        &manifest.id,
        &digest,
        manifest_bytes,
        manifest.artifacts,
        manifest.allowed_hosts,
        max_total,
    );
    let hub = hub(args.cache_dir)?;
    let installed = hub.install_bundle(&plan)?;
    println!(
        "{}\t{}\t{}",
        installed.path.display(),
        installed.digest,
        if installed.was_cached {
            "cached"
        } else {
            "installed"
        }
    );
    Ok(())
}

fn import(args: BundleImportArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (manifest, manifest_bytes, digest) = read_manifest(&args.manifest)?;
    let max_total = declared_total(&manifest)?;
    let artifacts = manifest
        .artifacts
        .iter()
        .map(|artifact| BundleImportArtifact {
            path: artifact.path.clone(),
            size_bytes: artifact.size_bytes,
            sha256: artifact.sha256.clone(),
        })
        .collect();
    let plan = BundleImportPlan::new(&manifest.id, &digest, manifest_bytes, artifacts, max_total);
    let hub = hub(args.cache_dir)?;
    let installed = hub.import_bundle(&args.source_dir, &plan)?;
    println!(
        "{}\t{}\t{}",
        installed.path.display(),
        installed.digest,
        if installed.was_cached {
            "cached"
        } else {
            "imported"
        }
    );
    Ok(())
}

fn read_manifest(
    path: &PathBuf,
) -> Result<(CatalogManifest, Vec<u8>, String), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let manifest: CatalogManifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != 1 {
        return Err(invalid("bundle schema_version must be 1"));
    }
    if manifest.domain.trim().is_empty()
        || manifest.model_id.trim().is_empty()
        || manifest.revision.len() != 40
        || !manifest
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || manifest.license.spdx.trim().is_empty()
        || manifest.license.notice.trim().is_empty()
        || !manifest.license.source.starts_with("https://")
        || !manifest.contract.is_object()
    {
        return Err(invalid(
            "bundle provenance, license, or contract is incomplete",
        ));
    }
    if manifest.artifacts.is_empty() {
        return Err(invalid("bundle artifacts must be non-empty"));
    }
    let reviewed_hosts = manifest
        .allowed_hosts
        .iter()
        .map(|host| host.trim().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    if reviewed_hosts.len() != manifest.allowed_hosts.len() {
        return Err(invalid("bundle allowed_hosts must be unique"));
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((manifest, bytes, digest))
}

fn declared_total(manifest: &CatalogManifest) -> Result<u64, Box<dyn std::error::Error>> {
    let total = manifest.artifacts.iter().try_fold(0u64, |total, artifact| {
        total.checked_add(artifact.size_bytes)
    });
    match total {
        Some(total) if total > 0 && total <= MAX_MANAGED_BUNDLE_BYTES => Ok(total),
        _ => Err(invalid(
            "bundle byte total is zero, overflows, or exceeds 1 GiB",
        )),
    }
}

fn hub(cache_dir: Option<PathBuf>) -> Result<ModelHub, Box<dyn std::error::Error>> {
    Ok(match cache_dir {
        Some(path) => ModelHub::with_cache_dir(path)?,
        None => ModelHub::new()?,
    })
}

fn invalid(message: &str) -> Box<dyn std::error::Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_embedding_manifest_is_complete_and_bounded() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../reference/embedding/nomic-embed-text-v1.5-a15734e.json");
        let (manifest, _, digest) = read_manifest(&path).unwrap();
        assert_eq!(manifest.id, "nomic-embed-text-v1.5-a15734e");
        assert_eq!(manifest.domain, "xrt-embedding");
        assert_eq!(manifest.artifacts.len(), 2);
        assert_eq!(digest.len(), 64);
        assert_eq!(declared_total(&manifest).unwrap(), 138_007_688);
    }
}
