use clap::{ArgGroup, Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use xrt_hub::{
    BundleArtifact, BundleImportArtifact, BundleImportPlan, BundleInstallPlan,
    BundleInstallProgress, ModelHub,
};
use xrt_image::{
    decode_image, BundleManifest, ImageBackendKind, ImageCancellation, ImageEditRequest,
    ImageGenerationRequest, ImageIoLimits, ImageModelBundle, ImageOffloadPolicy, ImageOutputFormat,
    ImageProgressEvent, ImageProgressPhase, ImageProgressSink, ImageQuality, ImageResizePolicy,
    ImageRuntime, ManifestMode, IMAGE_RNG_SCHEMA_V1,
};
use xrt_runtime::{GpuResourceManager, GpuResourceStatus};

use crate::format_bytes;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
const MAX_CATALOG_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CATALOG_BUNDLE_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const MAX_RAW_DIFFUSERS_INDEX_BYTES: u64 = 1024 * 1024;
const QWEN_IMAGE_2512_BF16_MANIFEST: &[u8] =
    include_bytes!("../../../reference/image/qwen/manifests/qwen-image-2512-bf16.json");
const QWEN_IMAGE_EDIT_2511_BF16_MANIFEST: &[u8] =
    include_bytes!("../../../reference/image/qwen/manifests/qwen-image-edit-2511-bf16.json");
const IMAGE_QUALITY_SUITE: &[u8] = include_bytes!("../../../tests/common/image-quality-suite.json");

#[derive(Args)]
pub(crate) struct ImageArgs {
    #[command(subcommand)]
    command: ImageCommand,
}

#[derive(Subcommand)]
enum ImageCommand {
    /// Generate one or more images with a local generative bundle.
    Generate(ImageGenerateArgs),
    /// Edit one or more local images with an edit-capable bundle.
    Edit(ImageEditArgs),
    /// Run repeatable local image-generation measurements and emit JSON.
    Bench(ImageBenchArgs),
    /// Produce one resumable shard of the frozen image quality corpus.
    QualityCorpus(ImageQualityCorpusArgs),
    /// Validate or install a local bundle or exact audited raw Qwen Diffusers directory.
    Import(ImageImportArgs),
}

#[derive(Debug, Clone, Args)]
struct ImageImportArgs {
    /// Local xrt.bundle.json directory or exact audited raw Qwen Diffusers directory.
    #[arg(long, value_name = "DIRECTORY")]
    path: PathBuf,
    /// Install the verified bundle into the managed XRT cache.
    #[arg(long)]
    install: bool,
    /// Override the XRT model cache used for installation.
    #[arg(long, env = "XRT_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
#[command(group(
    ArgGroup::new("image_model_source")
        .args(["model", "model_path"])
        .required(true)
))]
struct ImageModelArgs {
    /// Installed bundle catalog ID.
    #[arg(long, conflicts_with = "model_path")]
    model: Option<String>,
    /// Explicit local bundle directory containing xrt.bundle.json.
    #[arg(long, value_name = "DIRECTORY", conflicts_with = "model")]
    model_path: Option<PathBuf>,
    /// Override the XRT model cache used to resolve an installed bundle ID.
    #[arg(long, env = "XRT_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageBackendArg {
    Auto,
    Cpu,
    Cuda,
}

impl From<ImageBackendArg> for ImageBackendKind {
    fn from(value: ImageBackendArg) -> Self {
        match value {
            ImageBackendArg::Auto => Self::Auto,
            ImageBackendArg::Cpu => Self::Cpu,
            ImageBackendArg::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageOffloadArg {
    None,
    Sequential,
    Balanced,
    Cpu,
}

impl From<ImageOffloadArg> for ImageOffloadPolicy {
    fn from(value: ImageOffloadArg) -> Self {
        match value {
            ImageOffloadArg::None => Self::None,
            ImageOffloadArg::Sequential => Self::Sequential,
            ImageOffloadArg::Balanced => Self::Balanced,
            ImageOffloadArg::Cpu => Self::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageFormatArg {
    Png,
    Jpeg,
    Webp,
}

impl From<ImageFormatArg> for ImageOutputFormat {
    fn from(value: ImageFormatArg) -> Self {
        match value {
            ImageFormatArg::Png => Self::Png,
            ImageFormatArg::Jpeg => Self::Jpeg,
            ImageFormatArg::Webp => Self::Webp,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageQualityArg {
    Standard,
    Hd,
}

impl From<ImageQualityArg> for ImageQuality {
    fn from(value: ImageQualityArg) -> Self {
        match value {
            ImageQualityArg::Standard => Self::Standard,
            ImageQualityArg::Hd => Self::Hd,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageResizeArg {
    Reject,
    RoundDown,
}

impl From<ImageResizeArg> for ImageResizePolicy {
    fn from(value: ImageResizeArg) -> Self {
        match value {
            ImageResizeArg::Reject => Self::Reject,
            ImageResizeArg::RoundDown => Self::RoundDown,
        }
    }
}

#[derive(Debug, Clone, Args)]
struct ImageRequestArgs {
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    negative_prompt: Option<String>,
    /// Output size. Generation defaults to 1024x1024; editing follows the
    /// selected adapter's source-aware default when omitted.
    #[arg(long, value_parser = parse_size)]
    size: Option<(u32, u32)>,
    #[arg(long, default_value_t = 50)]
    steps: usize,
    #[arg(long, default_value_t = 4.0)]
    true_cfg_scale: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1)]
    n: usize,
    #[arg(long, value_enum, default_value_t = ImageBackendArg::Auto)]
    backend: ImageBackendArg,
    #[arg(long, value_enum, default_value_t = ImageOffloadArg::Sequential)]
    offload: ImageOffloadArg,
    #[arg(long, value_enum, default_value_t = ImageQualityArg::Standard)]
    quality: ImageQualityArg,
    #[arg(long, value_enum, default_value_t = ImageResizeArg::Reject)]
    resize_policy: ImageResizeArg,
    #[arg(long)]
    preview_interval_steps: Option<usize>,
}

#[derive(Args)]
struct ImageGenerateArgs {
    #[command(flatten)]
    model: ImageModelArgs,
    #[command(flatten)]
    request: ImageRequestArgs,
    #[arg(long, value_enum)]
    format: Option<ImageFormatArg>,
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
    #[arg(long)]
    overwrite: bool,
    #[arg(long, value_name = "FILE")]
    metadata: Option<PathBuf>,
    #[arg(long)]
    quiet: bool,
}

#[derive(Args)]
struct ImageEditArgs {
    #[command(flatten)]
    model: ImageModelArgs,
    #[command(flatten)]
    request: ImageRequestArgs,
    /// Ordered source images. Repeat the flag up to three times.
    #[arg(long = "image", required = true, value_name = "FILE")]
    images: Vec<PathBuf>,
    #[arg(long, value_name = "FILE")]
    mask: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    strength: f32,
    #[arg(long, value_enum)]
    format: Option<ImageFormatArg>,
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    quiet: bool,
}

#[derive(Args)]
struct ImageBenchArgs {
    #[command(flatten)]
    model: ImageModelArgs,
    #[command(flatten)]
    request: ImageRequestArgs,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    /// Retain the first PNG from the first measured repetition for quality review.
    #[arg(long, value_name = "FILE")]
    retain_first_output: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageQualityCorpusRole {
    Generation,
    Edit,
}

impl ImageQualityCorpusRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Generation => "generation",
            Self::Edit => "edit",
        }
    }

    fn accepts(self, category: &str) -> bool {
        match self {
            Self::Generation => category.starts_with("generation_"),
            Self::Edit => {
                category.starts_with("edit_")
                    || category == "identity_preservation"
                    || category == "conditional_inpaint"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageQualityCorpusSide {
    Bf16,
    Candidate,
}

impl ImageQualityCorpusSide {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Args)]
struct ImageQualityCorpusArgs {
    #[command(flatten)]
    model: ImageModelArgs,
    /// Plan emitted by reference/image/qwen/evaluate_quality_suite.py plan.
    #[arg(long, value_name = "FILE")]
    plan: PathBuf,
    /// Root under which the plan's immutable relative artifact paths are written.
    #[arg(long, value_name = "DIRECTORY")]
    artifact_root: PathBuf,
    /// Root containing avatar-NN.png and mask-NN.png suite fixtures.
    #[arg(long, value_name = "DIRECTORY")]
    fixture_root: Option<PathBuf>,
    #[arg(long, value_enum)]
    role: ImageQualityCorpusRole,
    #[arg(long, value_enum)]
    side: ImageQualityCorpusSide,
    #[arg(long, default_value_t = 0)]
    shard_index: usize,
    #[arg(long, default_value_t = 1)]
    shard_count: usize,
    #[arg(long, value_enum, default_value_t = ImageBackendArg::Cuda)]
    backend: ImageBackendArg,
    #[arg(long, value_enum, default_value_t = ImageOffloadArg::Sequential)]
    offload: ImageOffloadArg,
    /// Reuse only outputs whose checkpoint and PNG hashes still match this plan.
    #[arg(long)]
    resume: bool,
    /// Atomic JSON report for this shard.
    #[arg(long, value_name = "FILE")]
    report: PathBuf,
}

#[derive(Debug, Deserialize)]
struct QualityCorpusPlan {
    schema_version: u32,
    object: String,
    suite: QualityCorpusSuite,
    tier: String,
    execution: QualityCorpusExecution,
    cases: Vec<QualityCorpusCase>,
    identity_preservation_pairs: Vec<QualityCorpusCase>,
}

#[derive(Debug, Deserialize)]
struct QualityCorpusSuite {
    version: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct QualityCorpusExecution {
    paired_bf16_reference: bool,
    identical_xeno_initial_latent: bool,
    size: String,
    steps: usize,
    true_cfg_scale: f32,
    rng_schema: String,
    conditional_inpaint_admitted: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct QualityCorpusCase {
    id: String,
    category: String,
    request: QualityCorpusRequest,
    artifacts: QualityCorpusArtifacts,
}

#[derive(Debug, Clone, Deserialize)]
struct QualityCorpusRequest {
    prompt: String,
    seed: u64,
    #[serde(default)]
    source_fixture: Option<String>,
    #[serde(default)]
    source_fixtures: Vec<String>,
    #[serde(default)]
    mask_fixture: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct QualityCorpusArtifacts {
    bf16: String,
    candidate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageQualityCorpusRecord {
    schema_version: u32,
    plan_sha256: String,
    id: String,
    category: String,
    role: String,
    side: String,
    model_id: String,
    bundle_digest: String,
    artifact_revision: String,
    source_revisions: std::collections::BTreeMap<String, String>,
    quantization: String,
    artifact_path: String,
    output_sha256: String,
    width: u32,
    height: u32,
    seed: u64,
    steps: usize,
    true_cfg_scale: f32,
    wall_ms: f64,
}

#[derive(Serialize)]
struct ImageQualityCorpusReport {
    schema_version: u32,
    object: &'static str,
    plan_sha256: String,
    tier: String,
    role: String,
    side: String,
    shard_index: usize,
    shard_count: usize,
    completed: usize,
    resumed: usize,
    records: Vec<ImageQualityCorpusRecord>,
}

#[derive(Serialize)]
struct ImageMetadataSidecar {
    schema_version: u32,
    model: String,
    bundle_digest: String,
    backend: ImageBackendKind,
    quantization: String,
    width: u32,
    height: u32,
    seed: u64,
    steps: usize,
    true_cfg_scale: f32,
    output_sha256: String,
    timings: xrt_image::ImageTimings,
}

#[derive(Serialize)]
struct ImageBenchReport {
    schema_version: u32,
    object: &'static str,
    model: String,
    bundle_digest: String,
    requested_backend: ImageBackendKind,
    load_ms: f64,
    prompt_bytes: usize,
    quality_suite: ImageQualitySuiteIdentity,
    plan: xrt_image::ImageExecutionPlan,
    repetitions: Vec<ImageBenchRepetition>,
    gpu_resource: GpuResourceStatus,
}

#[derive(Debug, Serialize)]
struct ImageQualitySuiteIdentity {
    version: String,
    sha256: String,
}

#[derive(Serialize)]
struct ImageBenchRepetition {
    repetition: usize,
    wall_ms: f64,
    output_count: usize,
    output_bytes: usize,
    first_output_sha256: Option<String>,
    timings: xrt_image::ImageBatchTimings,
    images: Vec<xrt_image::ImageTimings>,
}

pub(crate) fn run(args: ImageArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ImageCommand::Generate(args) => run_generate(args),
        ImageCommand::Edit(args) => run_edit(args),
        ImageCommand::Bench(args) => run_bench(args),
        ImageCommand::QualityCorpus(args) => run_quality_corpus(args),
        ImageCommand::Import(args) => run_import(args),
    }
}

pub(crate) fn download_bundle(
    id: &str,
    cache_dir: Option<&Path>,
    catalog_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = resolve_catalog_manifest(id, cache_dir, catalog_dir)?;
    let metadata = fs::symlink_metadata(&manifest_path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_CATALOG_MANIFEST_BYTES
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bundle catalog manifest must be a non-empty regular file within the size limit",
        )
        .into());
    }
    let manifest_bytes = fs::read(&manifest_path)?;
    let plan = catalog_install_plan(id, manifest_bytes)?;
    let hub = match cache_dir {
        Some(cache_dir) => ModelHub::with_cache_dir(cache_dir)?,
        None => ModelHub::new()?,
    };
    let installed = hub.install_bundle_with_control(
        &plan,
        || false,
        |progress| report_bundle_progress(id, &progress),
    )?;
    if installed.was_cached {
        eprintln!(
            "Using verified bundle {}@{} at {}",
            installed.id,
            installed.digest,
            installed.path.display()
        );
    } else {
        eprintln!(
            "\rInstalled verified bundle {}@{} at {}",
            installed.id,
            installed.digest,
            installed.path.display()
        );
    }
    println!("{}", installed.path.display());
    Ok(())
}

fn catalog_install_plan(
    requested_id: &str,
    manifest_bytes: Vec<u8>,
) -> Result<BundleInstallPlan, Box<dyn std::error::Error>> {
    let manifest = BundleManifest::from_json_bytes(&manifest_bytes, ManifestMode::Catalog)?;
    if manifest.id != requested_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "catalog manifest ID `{}` does not match requested bundle `{requested_id}`",
                manifest.id
            ),
        )
        .into());
    }
    let digest = manifest.digest()?;
    let mut artifacts = Vec::new();
    let mut allowed_hosts = BTreeSet::new();
    let mut total_bytes = 0u64;
    for file in manifest
        .components
        .iter()
        .flat_map(|component| component.files.iter())
        .chain(manifest.license.files.iter())
    {
        total_bytes = total_bytes.checked_add(file.size_bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "bundle catalog byte count overflowed",
            )
        })?;
        let source = file.source.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("catalog artifact `{}` has no source URL", file.path),
            )
        })?;
        let host = https_host(&source).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "catalog artifact `{}` has an invalid source host",
                    file.path
                ),
            )
        })?;
        allowed_hosts.insert(host.to_string());
        artifacts.push(BundleArtifact {
            path: file.path.clone(),
            size_bytes: file.size_bytes,
            sha256: file.sha256.clone(),
            source,
        });
    }
    if total_bytes > MAX_CATALOG_BUNDLE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bundle declares {total_bytes} bytes, above the {MAX_CATALOG_BUNDLE_BYTES}-byte catalog limit"
            ),
        )
        .into());
    }
    if allowed_hosts.contains("huggingface.co") {
        for host in [
            "cdn-lfs.huggingface.co",
            "cdn-lfs-us-1.huggingface.co",
            "cdn-lfs-us-1.hf.co",
            "cas-bridge.xethub.hf.co",
        ] {
            allowed_hosts.insert(host.to_string());
        }
    }
    Ok(BundleInstallPlan::new(
        manifest.id,
        digest,
        manifest_bytes,
        artifacts,
        allowed_hosts.into_iter().collect(),
        MAX_CATALOG_BUNDLE_BYTES,
    ))
}

fn resolve_catalog_manifest(
    id: &str,
    cache_dir: Option<&Path>,
    explicit_catalog_dir: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if id.is_empty()
        || id.len() > 128
        || !id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"-_.".contains(&byte)
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "bundle ID must be a bounded lowercase-safe identifier",
        )
        .into());
    }
    let mut candidates = Vec::new();
    if let Some(directory) = explicit_catalog_dir {
        candidates.push(directory.to_path_buf());
    } else {
        if let Some(cache_dir) = cache_dir {
            candidates.push(cache_dir.join("catalog").join("image"));
        }
        if let Ok(executable) = std::env::current_exe() {
            if let Some(parent) = executable.parent() {
                candidates.push(parent.join("catalog").join("image"));
            }
        }
        candidates.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("reference/image/qwen/manifests"),
        );
    }
    for directory in &candidates {
        if !directory.is_dir() {
            continue;
        }
        let root = fs::canonicalize(directory)?;
        let candidate = root.join(format!("{id}.json"));
        let Ok(path) = fs::canonicalize(&candidate) else {
            continue;
        };
        if path.starts_with(&root) && path.is_file() {
            return Ok(path);
        }
    }
    let searched = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("bundle `{id}` is not present in the audited catalog; searched: {searched}"),
    )
    .into())
}

fn https_host(source: &str) -> Option<&str> {
    source
        .strip_prefix("https://")?
        .split('/')
        .next()
        .filter(|host| !host.is_empty() && !host.contains('@'))
}

fn report_bundle_progress(id: &str, progress: &BundleInstallProgress) {
    let percent = if progress.bundle_total == 0 {
        0.0
    } else {
        progress.bundle_downloaded as f64 / progress.bundle_total as f64 * 100.0
    };
    eprint!(
        "\rInstalling {id} {:>6.2}% ({}/{}) [{}]",
        percent,
        format_bytes(progress.bundle_downloaded),
        format_bytes(progress.bundle_total),
        progress.artifact_path
    );
    let _ = io::stderr().flush();
}

fn synthesize_audited_raw_diffusers_manifest(
    root: &Path,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let root = fs::canonicalize(root)?;
    let index_path = root.join("model_index.json");
    let metadata = fs::symlink_metadata(&index_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "local image import requires xrt.bundle.json or an audited Qwen Diffusers model_index.json: {error}"
            ),
        )
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_RAW_DIFFUSERS_INDEX_BYTES
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw Diffusers model_index.json must be a non-empty regular file within the size limit",
        )
        .into());
    }
    let canonical_index = fs::canonicalize(&index_path)?;
    if !canonical_index.starts_with(&root) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw Diffusers model_index.json escapes the selected import root",
        )
        .into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(&canonical_index)?).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid raw Diffusers model_index.json: {error}"),
            )
        })?;
    let object = value.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "raw Diffusers model_index.json must be a JSON object",
        )
    })?;
    for forbidden in ["_module", "custom_pipeline", "trust_remote_code"] {
        if object.contains_key(forbidden) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw Diffusers field `{forbidden}` is unsupported; XENO RT never executes remote model code"
                ),
            )
            .into());
        }
    }
    let class_name = object
        .get("_class_name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "raw Diffusers model_index.json is missing string `_class_name`",
            )
        })?;
    let catalog = match class_name {
        "QwenImagePipeline" => QWEN_IMAGE_2512_BF16_MANIFEST,
        "QwenImageEditPlusPipeline" => QWEN_IMAGE_EDIT_2511_BF16_MANIFEST,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported raw Diffusers pipeline `{other}`; only exact audited QwenImagePipeline and QwenImageEditPlusPipeline bundles are importable"
                ),
            )
            .into())
        }
    };
    localize_catalog_manifest(catalog)
}

fn localize_catalog_manifest(catalog_bytes: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut manifest = BundleManifest::from_json_bytes(catalog_bytes, ManifestMode::Catalog)?;
    for file in manifest
        .components
        .iter_mut()
        .flat_map(|component| component.files.iter_mut())
        .chain(manifest.license.files.iter_mut())
    {
        file.source = None;
        file.source_kind = Some("local".to_string());
    }
    manifest.validate(ManifestMode::LocalImport)?;
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    if bytes.len() as u64 > MAX_CATALOG_MANIFEST_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "synthesized local image manifest exceeds the manifest size limit",
        )
        .into());
    }
    Ok(bytes)
}

fn run_import(args: ImageImportArgs) -> Result<(), Box<dyn std::error::Error>> {
    let root_metadata = fs::symlink_metadata(&args.path)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "image import path must be a real bundle directory, not a symlink",
        )
        .into());
    }
    let root = fs::canonicalize(&args.path)?;
    let manifest_path = root.join("xrt.bundle.json");
    let (manifest_bytes, synthesized) = match fs::symlink_metadata(&manifest_path) {
        Ok(manifest_metadata) => {
            if manifest_metadata.file_type().is_symlink()
                || !manifest_metadata.is_file()
                || manifest_metadata.len() == 0
                || manifest_metadata.len() > MAX_CATALOG_MANIFEST_BYTES
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "local xrt.bundle.json must be a non-empty regular file within the size limit",
                )
                .into());
            }
            (fs::read(&manifest_path)?, false)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            (synthesize_audited_raw_diffusers_manifest(&root)?, true)
        }
        Err(error) => return Err(error.into()),
    };
    let manifest = BundleManifest::from_json_bytes(&manifest_bytes, ManifestMode::LocalImport)?;
    let bundle = ImageModelBundle::open_local_import(&root, &manifest_bytes).map_err(|error| {
        if synthesized {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw Diffusers directory does not exactly match the audited {} artifact set: {error}",
                    manifest.id
                ),
            )
        } else {
            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
        }
    })?;
    if bundle.manifest() != &manifest {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "validated local manifest changed while the bundle was being opened",
        )
        .into());
    }
    if !args.install {
        if synthesized {
            eprintln!(
                "validated exact audited raw Diffusers bundle {}@{}; review or redirect the candidate manifest below",
                manifest.id,
                bundle.digest()
            );
            println!("{}", String::from_utf8(manifest_bytes)?);
            return Ok(());
        }
        println!(
            "validated local image bundle {}@{} ({})",
            manifest.id,
            bundle.digest(),
            root.display()
        );
        return Ok(());
    }
    if synthesized {
        eprintln!(
            "Recognized and validated exact audited raw Diffusers bundle {}@{}; source directory remains unchanged",
            manifest.id,
            bundle.digest()
        );
    }
    let plan = local_import_plan(&manifest, bundle.digest(), manifest_bytes)?;
    let hub = match args.cache_dir {
        Some(cache_dir) => ModelHub::with_cache_dir(cache_dir)?,
        None => ModelHub::new()?,
    };
    let installed = hub.import_bundle_with_control(
        &root,
        &plan,
        || false,
        |progress| report_bundle_progress(&manifest.id, &progress),
    )?;
    if installed.was_cached {
        eprintln!(
            "Using verified imported bundle {}@{} at {}",
            installed.id,
            installed.digest,
            installed.path.display()
        );
    } else {
        eprintln!(
            "\rImported verified bundle {}@{} at {}",
            installed.id,
            installed.digest,
            installed.path.display()
        );
    }
    println!("{}", installed.path.display());
    Ok(())
}

fn local_import_plan(
    manifest: &BundleManifest,
    digest: &str,
    manifest_bytes: Vec<u8>,
) -> Result<BundleImportPlan, Box<dyn std::error::Error>> {
    let mut artifacts = Vec::new();
    let mut total_bytes = 0u64;
    for file in manifest
        .components
        .iter()
        .flat_map(|component| component.files.iter())
        .chain(manifest.license.files.iter())
    {
        total_bytes = total_bytes.checked_add(file.size_bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "local bundle byte count overflowed",
            )
        })?;
        artifacts.push(BundleImportArtifact {
            path: file.path.clone(),
            size_bytes: file.size_bytes,
            sha256: file.sha256.clone(),
        });
    }
    if total_bytes > MAX_CATALOG_BUNDLE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "local bundle declares {total_bytes} bytes, above the {MAX_CATALOG_BUNDLE_BYTES}-byte import limit"
            ),
        )
        .into());
    }
    Ok(BundleImportPlan::new(
        manifest.id.clone(),
        digest.to_string(),
        manifest_bytes,
        artifacts,
        MAX_CATALOG_BUNDLE_BYTES,
    ))
}

fn run_generate(args: ImageGenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = args
        .format
        .map(ImageOutputFormat::from)
        .unwrap_or_else(|| infer_output_format(&args.output));
    let (runtime, model, bundle_digest, resources, load_ms) =
        load_runtime(&args.model, args.request.backend.into())?;
    let request = generation_request(&args.request, model.clone(), format, (1024, 1024));
    let progress = (!args.quiet).then(progress_sink);
    let started = Instant::now();
    let result = runtime.generate(request.clone(), ImageCancellation::new(), progress)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    if result.images.is_empty() {
        return Err(io::Error::other("image runtime returned no outputs").into());
    }

    let mut written = Vec::with_capacity(result.images.len());
    for (index, image) in result.images.iter().enumerate() {
        let path = output_path(&args.output, index, result.images.len());
        atomic_write(&path, &image.bytes, args.overwrite)?;
        written.push(path);
    }
    if let Some(path) = &args.metadata {
        let first = &result.images[0];
        let sidecar = ImageMetadataSidecar {
            schema_version: 1,
            model,
            bundle_digest,
            backend: first.backend,
            quantization: first.quantization.clone(),
            width: first.width,
            height: first.height,
            seed: first.seed,
            steps: request.steps,
            true_cfg_scale: request.true_cfg_scale,
            output_sha256: sha256(&first.bytes),
            timings: first.timings.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&sidecar)?;
        atomic_write(path, &bytes, args.overwrite)?;
    }

    for path in written {
        println!("{}", path.display());
    }
    if !args.quiet {
        eprintln!(
            "generated {} image(s) in {:.3}s; load {:.3}s; backend={}; tracked_gpu_bytes={}",
            result.images.len(),
            wall_ms / 1000.0,
            load_ms / 1000.0,
            result.images[0].backend.as_str(),
            resources.allocation_arena().snapshot().peak_allocated_bytes,
        );
    }
    Ok(())
}

fn run_edit(args: ImageEditArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = args
        .format
        .map(ImageOutputFormat::from)
        .unwrap_or_else(|| infer_output_format(&args.output));
    let (runtime, model, _, _, _) = load_runtime(&args.model, args.request.backend.into())?;
    let limits = ImageIoLimits::default();
    if args.images.len() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at most three ordered source images are supported",
        )
        .into());
    }
    let images = args
        .images
        .iter()
        .map(|path| read_and_decode(path, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let mask = args
        .mask
        .as_deref()
        .map(|path| read_and_decode(path, limits))
        .transpose()?;
    let default_size = runtime.default_edit_dimensions(&images)?;
    let request = ImageEditRequest {
        generation: generation_request(&args.request, model, format, default_size),
        images,
        mask,
        strength: args.strength,
    };
    let progress = (!args.quiet).then(progress_sink);
    let result = runtime.edit(request, ImageCancellation::new(), progress)?;
    for (index, image) in result.images.iter().enumerate() {
        let path = output_path(&args.output, index, result.images.len());
        atomic_write(&path, &image.bytes, args.overwrite)?;
        println!("{}", path.display());
    }
    Ok(())
}

fn run_bench(args: ImageBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.repetitions == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--repetitions must be at least one",
        )
        .into());
    }
    if let Some(path) = args.retain_first_output.as_deref() {
        ensure_create_new_output(path)?;
    }
    let (runtime, model, bundle_digest, resources, load_ms) =
        load_runtime(&args.model, args.request.backend.into())?;
    let request = generation_request(
        &args.request,
        model.clone(),
        ImageOutputFormat::Png,
        (1024, 1024),
    );
    let plan = runtime.plan_generation(&request)?;
    let mut repetitions = Vec::with_capacity(args.repetitions);
    for repetition in 1..=args.repetitions {
        let started = Instant::now();
        let result = runtime.generate(request.clone(), ImageCancellation::new(), None)?;
        if repetition == 1 {
            if let (Some(path), Some(image)) =
                (args.retain_first_output.as_deref(), result.images.first())
            {
                atomic_write(path, &image.bytes, false)?;
            }
        }
        repetitions.push(ImageBenchRepetition {
            repetition,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            output_count: result.images.len(),
            output_bytes: result.images.iter().map(|image| image.bytes.len()).sum(),
            first_output_sha256: result.images.first().map(|image| sha256(&image.bytes)),
            timings: result.timings,
            images: result
                .images
                .into_iter()
                .map(|image| image.timings)
                .collect(),
        });
    }
    let report = ImageBenchReport {
        schema_version: 1,
        object: "xrt.image.benchmark",
        model,
        bundle_digest,
        requested_backend: args.request.backend.into(),
        load_ms,
        prompt_bytes: args.request.prompt.len(),
        quality_suite: image_quality_suite_identity()?,
        plan,
        repetitions,
        gpu_resource: image_gpu_resource_status(&resources, runtime.backend()),
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        for sample in &report.repetitions {
            println!(
                "repetition={} wall_ms={:.3} outputs={} bytes={} sha256={}",
                sample.repetition,
                sample.wall_ms,
                sample.output_count,
                sample.output_bytes,
                sample.first_output_sha256.as_deref().unwrap_or("none")
            );
        }
    }
    Ok(())
}

fn run_quality_corpus(args: ImageQualityCorpusArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.shard_count == 0 || args.shard_index >= args.shard_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--shard-count must be positive and --shard-index must be smaller",
        )
        .into());
    }
    let plan_bytes = fs::read(&args.plan)?;
    if plan_bytes.is_empty() || plan_bytes.len() as u64 > MAX_CATALOG_MANIFEST_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality plan must be a non-empty JSON file within the manifest size limit",
        )
        .into());
    }
    let plan_sha256 = sha256(&plan_bytes);
    let plan: QualityCorpusPlan = serde_json::from_slice(&plan_bytes)?;
    validate_quality_corpus_plan(&plan)?;
    let size = parse_size(&plan.execution.size)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    fs::create_dir_all(&args.artifact_root)?;
    let artifact_root = args.artifact_root.canonicalize()?;
    let fixture_root = args
        .fixture_root
        .as_deref()
        .map(Path::canonicalize)
        .transpose()?;
    if matches!(args.role, ImageQualityCorpusRole::Edit) && fixture_root.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--fixture-root is required for the edit corpus",
        )
        .into());
    }

    let mut cases = plan
        .cases
        .iter()
        .chain(plan.identity_preservation_pairs.iter())
        .filter(|case| args.role.accepts(&case.category))
        .cloned()
        .collect::<Vec<_>>();
    cases.sort_by(|left, right| left.id.cmp(&right.id));
    let cases = cases
        .into_iter()
        .enumerate()
        .filter_map(|(index, case)| (index % args.shard_count == args.shard_index).then_some(case))
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the selected role and shard contain no quality cases",
        )
        .into());
    }

    let bundle_path = resolve_bundle_path(&args.model)?;
    let bundle = ImageModelBundle::open(bundle_path)?;
    let manifest = bundle.manifest().clone();
    let bundle_digest = bundle.digest().to_string();
    let expected_quantization = match args.side {
        ImageQualityCorpusSide::Bf16 => "BF16",
        ImageQualityCorpusSide::Candidate => plan.tier.as_str(),
    };
    if manifest.quantization != expected_quantization {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} side requires quantization {expected_quantization}, bundle declares {}",
                args.side.as_str(),
                manifest.quantization
            ),
        )
        .into());
    }
    let resources = Arc::new(GpuResourceManager::from_env());
    let runtime = ImageRuntime::load(bundle, args.backend.into(), resources)?;
    let limits = ImageIoLimits::default();
    let mut records = Vec::with_capacity(cases.len());
    let mut completed = 0usize;
    let mut resumed = 0usize;

    for case in cases {
        let relative = quality_artifact_path(&case, args.side)?;
        let output_path = contained_output_path(&artifact_root, &relative)?;
        let checkpoint_path = output_path.with_extension("png.xrt.json");
        if args.resume && output_path.is_file() && checkpoint_path.is_file() {
            let record: ImageQualityCorpusRecord =
                serde_json::from_slice(&fs::read(&checkpoint_path)?)?;
            validate_resumed_quality_record(
                &record,
                &case,
                &relative,
                &output_path,
                &plan_sha256,
                &manifest,
                &bundle_digest,
                args.role,
                args.side,
                size,
            )?;
            records.push(record);
            resumed += 1;
            continue;
        }
        if output_path.exists() || checkpoint_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "quality output/checkpoint is incomplete or resume was not requested: {}",
                    output_path.display()
                ),
            )
            .into());
        }

        let generation = ImageGenerationRequest {
            model: manifest.id.clone(),
            prompt: case.request.prompt.clone(),
            negative_prompt: None,
            width: size.0,
            height: size.1,
            n: 1,
            steps: plan.execution.steps,
            true_cfg_scale: plan.execution.true_cfg_scale,
            seed: case.request.seed,
            output_format: ImageOutputFormat::Png,
            quality: ImageQuality::Standard,
            backend: args.backend.into(),
            offload: args.offload.into(),
            resize_policy: ImageResizePolicy::Reject,
            preview_interval: None,
        };
        let started = Instant::now();
        let result = match args.role {
            ImageQualityCorpusRole::Generation => {
                runtime.generate(generation, ImageCancellation::new(), Some(progress_sink()))?
            }
            ImageQualityCorpusRole::Edit => {
                let fixture_root = fixture_root.as_deref().expect("validated fixture root");
                let images = quality_source_fixture_ids(&case.request)?
                    .into_iter()
                    .map(|fixture| read_quality_fixture(fixture_root, fixture, limits))
                    .collect::<Result<Vec<_>, _>>()?;
                let mask = case
                    .request
                    .mask_fixture
                    .as_deref()
                    .map(|fixture| read_quality_fixture(fixture_root, fixture, limits))
                    .transpose()?;
                runtime.edit(
                    ImageEditRequest {
                        generation,
                        images,
                        mask,
                        strength: 1.0,
                    },
                    ImageCancellation::new(),
                    Some(progress_sink()),
                )?
            }
        };
        if result.images.len() != 1 {
            return Err(
                io::Error::other("quality corpus request returned other than one image").into(),
            );
        }
        let image = &result.images[0];
        if image.quantization != expected_quantization
            || image.width != size.0
            || image.height != size.1
            || image.seed != case.request.seed
        {
            return Err(
                io::Error::other(format!("quality output identity drift for {}", case.id)).into(),
            );
        }
        atomic_write(&output_path, &image.bytes, false)?;
        let record = ImageQualityCorpusRecord {
            schema_version: 1,
            plan_sha256: plan_sha256.clone(),
            id: case.id.clone(),
            category: case.category.clone(),
            role: args.role.as_str().to_string(),
            side: args.side.as_str().to_string(),
            model_id: manifest.id.clone(),
            bundle_digest: bundle_digest.clone(),
            artifact_revision: manifest.revision.clone(),
            source_revisions: manifest.source_revisions.clone(),
            quantization: image.quantization.clone(),
            artifact_path: relative.clone(),
            output_sha256: sha256(&image.bytes),
            width: image.width,
            height: image.height,
            seed: image.seed,
            steps: plan.execution.steps,
            true_cfg_scale: plan.execution.true_cfg_scale,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        atomic_write(
            &checkpoint_path,
            &serde_json::to_vec_pretty(&record)?,
            false,
        )?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "status": "completed",
                "id": case.id,
                "side": args.side.as_str(),
                "artifact": relative,
                "sha256": record.output_sha256,
            }))?
        );
        records.push(record);
        completed += 1;
    }

    let report = ImageQualityCorpusReport {
        schema_version: 1,
        object: "xeno.image.quality_corpus_shard",
        plan_sha256,
        tier: plan.tier,
        role: args.role.as_str().to_string(),
        side: args.side.as_str().to_string(),
        shard_index: args.shard_index,
        shard_count: args.shard_count,
        completed,
        resumed,
        records,
    };
    atomic_write(&args.report, &serde_json::to_vec_pretty(&report)?, true)?;
    Ok(())
}

fn validate_quality_corpus_plan(plan: &QualityCorpusPlan) -> io::Result<()> {
    const SUITE_VERSION: &str = "qwen-image-release-v1";
    const SUITE_SHA256: &str = "eab7ceca3f39705c3f4e8829376c23f554f85fec99de08160414839b79544c88";
    if plan.schema_version != 1
        || plan.object != "xeno.image.quality_plan"
        || plan.suite.version != SUITE_VERSION
        || plan.suite.sha256 != SUITE_SHA256
        || !plan.execution.paired_bf16_reference
        || !plan.execution.identical_xeno_initial_latent
        || plan.execution.rng_schema != IMAGE_RNG_SCHEMA_V1
        || plan.execution.conditional_inpaint_admitted
        || plan.execution.size != "1024x1024"
        || plan.execution.steps != 50
        || plan.execution.true_cfg_scale != 4.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality corpus plan does not match the frozen non-inpaint release protocol",
        ));
    }
    if plan.tier.is_empty() || plan.tier == "BF16" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality corpus plan tier must name one quantized candidate",
        ));
    }
    Ok(())
}

fn quality_artifact_path(
    case: &QualityCorpusCase,
    side: ImageQualityCorpusSide,
) -> io::Result<String> {
    if !valid_quality_token(&case.id) || !valid_quality_token(&case.category) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality case ID or category is not a bounded safe token",
        ));
    }
    let actual = match side {
        ImageQualityCorpusSide::Bf16 => &case.artifacts.bf16,
        ImageQualityCorpusSide::Candidate => &case.artifacts.candidate,
    };
    let expected = format!("{}/{}/{}.png", side.as_str(), case.category, case.id);
    if actual != &expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("quality artifact path drift for {}", case.id),
        ));
    }
    Ok(expected)
}

fn valid_quality_token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn contained_output_path(root: &Path, relative: &str) -> io::Result<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality artifact path must be a normal relative path",
        ));
    }
    let path = root.join(relative);
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "quality artifact has no parent")
    })?;
    fs::create_dir_all(parent)?;
    let canonical_parent = parent.canonicalize()?;
    if !canonical_parent.starts_with(root) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "quality artifact parent escapes the artifact root",
        ));
    }
    Ok(canonical_parent.join(path.file_name().expect("normal relative filename")))
}

fn quality_source_fixture_ids(request: &QualityCorpusRequest) -> io::Result<Vec<&str>> {
    let fixtures: Vec<&str> = if request.source_fixtures.is_empty() {
        request.source_fixture.iter().map(String::as_str).collect()
    } else if request.source_fixture.is_none() {
        request.source_fixtures.iter().map(String::as_str).collect()
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality edit request cannot contain both singular and plural source fixtures",
        ));
    };
    if fixtures.is_empty() || fixtures.len() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality edit request requires between one and three source fixtures",
        ));
    }
    Ok(fixtures)
}

fn read_quality_fixture(
    root: &Path,
    fixture: &str,
    limits: ImageIoLimits,
) -> Result<xrt_image::DecodedImage, Box<dyn std::error::Error>> {
    if !valid_quality_token(fixture) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "quality fixture ID is not a bounded safe token",
        )
        .into());
    }
    let path = root.join(format!("{fixture}.png")).canonicalize()?;
    if !path.starts_with(root) || !path.is_file() || path.is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "quality fixture must be a regular PNG contained by the fixture root",
        )
        .into());
    }
    read_and_decode(&path, limits)
}

#[allow(clippy::too_many_arguments)]
fn validate_resumed_quality_record(
    record: &ImageQualityCorpusRecord,
    case: &QualityCorpusCase,
    relative: &str,
    output_path: &Path,
    plan_sha256: &str,
    manifest: &BundleManifest,
    bundle_digest: &str,
    role: ImageQualityCorpusRole,
    side: ImageQualityCorpusSide,
    size: (u32, u32),
) -> io::Result<()> {
    if record.schema_version != 1
        || record.plan_sha256 != plan_sha256
        || record.id != case.id
        || record.category != case.category
        || record.role != role.as_str()
        || record.side != side.as_str()
        || record.model_id != manifest.id
        || record.bundle_digest != bundle_digest
        || record.artifact_revision != manifest.revision
        || record.source_revisions != manifest.source_revisions
        || record.quantization != manifest.quantization
        || record.artifact_path != relative
        || record.width != size.0
        || record.height != size.1
        || record.seed != case.request.seed
        || record.steps != 50
        || record.true_cfg_scale != 4.0
        || sha256_path(output_path)? != record.output_sha256
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("quality resume checkpoint drift for {}", case.id),
        ));
    }
    Ok(())
}

fn sha256_path(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn image_quality_suite_identity() -> Result<ImageQualitySuiteIdentity, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_slice(IMAGE_QUALITY_SUITE)?;
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
        || value.get("status").and_then(serde_json::Value::as_str) != Some("frozen")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "embedded image quality suite must be frozen schema version 1",
        )
        .into());
    }
    let version = value
        .get("suite_version")
        .and_then(serde_json::Value::as_str)
        .filter(|version| !version.is_empty())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "embedded image quality suite is missing suite_version",
            )
        })?;
    Ok(ImageQualitySuiteIdentity {
        version: version.to_string(),
        sha256: sha256(IMAGE_QUALITY_SUITE),
    })
}

fn image_gpu_resource_status(
    resources: &GpuResourceManager,
    active_backend: ImageBackendKind,
) -> GpuResourceStatus {
    resources.status_with_allocations(0, 0, 0, 0, matches!(active_backend, ImageBackendKind::Cuda))
}

fn load_runtime(
    source: &ImageModelArgs,
    backend: ImageBackendKind,
) -> Result<(ImageRuntime, String, String, Arc<GpuResourceManager>, f64), Box<dyn std::error::Error>>
{
    let path = resolve_bundle_path(source)?;
    let started = Instant::now();
    let bundle = ImageModelBundle::open(path)?;
    let model = bundle.manifest().id.clone();
    let bundle_digest = bundle.digest().to_string();
    let resources = Arc::new(GpuResourceManager::from_env());
    let runtime = ImageRuntime::load(bundle, backend, Arc::clone(&resources))?;
    Ok((
        runtime,
        model,
        bundle_digest,
        resources,
        started.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn resolve_bundle_path(source: &ImageModelArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = &source.model_path {
        return Ok(path.clone());
    }
    let id = source.model.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--model or --model-path is required",
        )
    })?;
    let hub = match &source.cache_dir {
        Some(path) => ModelHub::with_cache_dir(path)?,
        None => ModelHub::new()?,
    };
    Ok(hub.resolve_installed_bundle(id, None)?)
}

fn generation_request(
    args: &ImageRequestArgs,
    model: String,
    format: ImageOutputFormat,
    default_size: (u32, u32),
) -> ImageGenerationRequest {
    let (width, height) = args.size.unwrap_or(default_size);
    ImageGenerationRequest {
        model,
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone(),
        width,
        height,
        n: args.n,
        steps: args.steps,
        true_cfg_scale: args.true_cfg_scale,
        seed: args.seed,
        output_format: format,
        quality: args.quality.into(),
        backend: args.backend.into(),
        offload: args.offload.into(),
        resize_policy: args.resize_policy.into(),
        preview_interval: args.preview_interval_steps,
    }
}

fn progress_sink() -> Arc<dyn ImageProgressSink> {
    Arc::new(|event: &ImageProgressEvent| {
        let detail = match (event.step, event.total_steps) {
            (Some(step), Some(total)) => format!(" {}/{}", step.saturating_add(1), total),
            _ => String::new(),
        };
        eprint!(
            "\routput {}: {}{}",
            event.output_index.saturating_add(1),
            phase_name(event.phase),
            detail
        );
        if event.phase == ImageProgressPhase::Complete {
            eprintln!();
        }
        let _ = io::stderr().flush();
    })
}

fn phase_name(phase: ImageProgressPhase) -> &'static str {
    match phase {
        ImageProgressPhase::Admitted => "admitted",
        ImageProgressPhase::PromptEncoding => "prompt encoding",
        ImageProgressPhase::SourceEncoding => "source encoding",
        ImageProgressPhase::Denoising => "denoising",
        ImageProgressPhase::VaeDecode => "VAE decode",
        ImageProgressPhase::Encoding => "encoding",
        ImageProgressPhase::Complete => "complete",
    }
}

fn read_and_decode(
    path: &Path,
    limits: ImageIoLimits,
) -> Result<xrt_image::DecodedImage, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 || metadata.len() > limits.max_encoded_bytes as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "encoded image `{}` must be between 1 and {} bytes",
                path.display(),
                limits.max_encoded_bytes
            ),
        )
        .into());
    }
    Ok(decode_image(&fs::read(path)?, limits)?)
}

fn infer_output_format(path: &Path) -> ImageOutputFormat {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => ImageOutputFormat::Jpeg,
        Some("webp") => ImageOutputFormat::Webp,
        _ => ImageOutputFormat::Png,
    }
}

fn parse_size(value: &str) -> Result<(u32, u32), String> {
    let (width, height) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| "size must use WIDTHxHEIGHT".to_string())?;
    let width = width
        .parse::<u32>()
        .map_err(|_| "size width must be an unsigned integer".to_string())?;
    let height = height
        .parse::<u32>()
        .map_err(|_| "size height must be an unsigned integer".to_string())?;
    if width == 0 || height == 0 {
        return Err("size dimensions must be non-zero".to_string());
    }
    Ok((width, height))
}

fn output_path(base: &Path, index: usize, total: usize) -> PathBuf {
    if total <= 1 {
        return base.to_path_buf();
    }
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("image");
    let mut name = format!("{stem}-{:02}", index + 1);
    if let Some(extension) = base.extension().and_then(|extension| extension.to_str()) {
        name.push('.');
        name.push_str(extension);
    }
    base.with_file_name(name)
}

fn ensure_create_new_output(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("refusing to overwrite {}", path.display()),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn atomic_write(path: &Path, bytes: &[u8], overwrite: bool) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    let parent = parent.unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let (temporary, mut file) = create_temporary(parent, file_name)?;
    let mut guard = TemporaryGuard(Some(temporary.clone()));
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);

    if overwrite {
        atomic_replace(&temporary, path)?;
    } else {
        fs::hard_link(&temporary, path)?;
        fs::remove_file(&temporary)?;
    }
    guard.0 = None;
    Ok(())
}

fn create_temporary(parent: &Path, file_name: &str) -> io::Result<(PathBuf, File)> {
    for _ in 0..64 {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{file_name}.{}.{}.tmp",
            std::process::id(),
            counter
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique output staging file",
    ))
}

struct TemporaryGuard(Option<PathBuf>);

impl Drop for TemporaryGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(unix)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic output replacement is unsupported on this platform",
        ));
    }
    fs::rename(source, destination)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[allow(dead_code)]
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_embeds_frozen_quality_suite_identity() {
        let identity = image_quality_suite_identity().unwrap();
        assert_eq!(identity.version, "qwen-image-release-v1");
        assert_eq!(
            identity.sha256,
            "eab7ceca3f39705c3f4e8829376c23f554f85fec99de08160414839b79544c88"
        );
    }

    #[test]
    fn quality_corpus_accepts_only_the_frozen_paired_protocol() {
        let plan: QualityCorpusPlan = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "object": "xeno.image.quality_plan",
            "suite": {
                "version": "qwen-image-release-v1",
                "sha256": "eab7ceca3f39705c3f4e8829376c23f554f85fec99de08160414839b79544c88"
            },
            "tier": "Q4_K_M",
            "execution": {
                "paired_bf16_reference": true,
                "identical_xeno_initial_latent": true,
                "size": "1024x1024",
                "steps": 50,
                "true_cfg_scale": 4.0,
                "rng_schema": IMAGE_RNG_SCHEMA_V1,
                "conditional_inpaint_admitted": false
            },
            "cases": [],
            "identity_preservation_pairs": []
        }))
        .unwrap();
        validate_quality_corpus_plan(&plan).unwrap();

        let mut drifted = plan;
        drifted.execution.steps = 49;
        assert!(validate_quality_corpus_plan(&drifted).is_err());
    }

    #[test]
    fn quality_corpus_artifact_paths_are_derived_not_trusted() {
        let mut case = QualityCorpusCase {
            id: "gen-general-001".to_string(),
            category: "generation_general".to_string(),
            request: QualityCorpusRequest {
                prompt: "fixture".to_string(),
                seed: 10001,
                source_fixture: None,
                source_fixtures: Vec::new(),
                mask_fixture: None,
            },
            artifacts: QualityCorpusArtifacts {
                bf16: "bf16/generation_general/gen-general-001.png".to_string(),
                candidate: "candidate/generation_general/gen-general-001.png".to_string(),
            },
        };
        assert_eq!(
            quality_artifact_path(&case, ImageQualityCorpusSide::Candidate).unwrap(),
            "candidate/generation_general/gen-general-001.png"
        );
        case.artifacts.candidate = "../outside.png".to_string();
        assert!(quality_artifact_path(&case, ImageQualityCorpusSide::Candidate).is_err());
    }

    #[test]
    fn quality_corpus_source_fixture_shapes_are_unambiguous() {
        let singular = QualityCorpusRequest {
            prompt: "fixture".to_string(),
            seed: 1,
            source_fixture: Some("avatar-01".to_string()),
            source_fixtures: Vec::new(),
            mask_fixture: None,
        };
        assert_eq!(
            quality_source_fixture_ids(&singular).unwrap(),
            ["avatar-01"]
        );

        let ambiguous = QualityCorpusRequest {
            source_fixtures: vec!["avatar-02".to_string()],
            ..singular
        };
        assert!(quality_source_fixture_ids(&ambiguous).is_err());
    }

    fn local_bundle_fixture(root: &Path) -> String {
        let bytes = b"fixture-component";
        let sha = sha256(bytes);
        let components = [
            "transformer",
            "text_encoder",
            "tokenizer",
            "vae",
            "scheduler",
        ]
        .into_iter()
        .map(|role| {
            let path = format!("{role}/fixture.bin");
            fs::create_dir_all(root.join(role)).unwrap();
            fs::write(root.join(&path), bytes).unwrap();
            serde_json::json!({
                "role": role,
                "format": "json",
                "optional": false,
                "files": [{
                    "path": path,
                    "size_bytes": bytes.len(),
                    "sha256": sha.clone(),
                    "source_kind": "local"
                }]
            })
        })
        .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "id": "local-image-fixture",
            "family": "local-image-fixture",
            "revision": "local-fixture-v1",
            "capabilities": ["image.generate"],
            "license": {
                "spdx": "Apache-2.0",
                "evidence": "https://example.com/licenses/local-fixture",
                "files": []
            },
            "quantization": "fixture",
            "components": components,
            "limits": {
                "max_sequence_length": 32,
                "max_width": 64,
                "max_height": 64,
                "max_pixels": 4096
            }
        });
        fs::write(
            root.join("xrt.bundle.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        "local-image-fixture".to_string()
    }

    #[test]
    fn parses_image_size_and_rejects_invalid_values() {
        assert_eq!(parse_size("1024x768").unwrap(), (1024, 768));
        assert_eq!(parse_size("16X32").unwrap(), (16, 32));
        assert!(parse_size("1024").is_err());
        assert!(parse_size("0x16").is_err());
    }

    #[test]
    fn multi_output_paths_preserve_extension() {
        assert_eq!(
            output_path(Path::new("render.png"), 1, 3),
            PathBuf::from("render-02.png")
        );
    }

    #[test]
    fn image_benchmark_status_reports_the_active_cuda_backend() {
        let resources = GpuResourceManager::new(Default::default());

        assert!(!image_gpu_resource_status(&resources, ImageBackendKind::Cpu).cuda_available);
        assert!(image_gpu_resource_status(&resources, ImageBackendKind::Cuda).cuda_available);
    }

    #[test]
    fn pinned_catalog_manifest_builds_a_complete_bounded_install_plan() {
        let bytes =
            include_bytes!("../../../reference/image/qwen/manifests/qwen-image-2512-q4_k_m.json")
                .to_vec();
        let manifest = BundleManifest::from_json_bytes(&bytes, ManifestMode::Catalog).unwrap();
        let declared_files = manifest
            .components
            .iter()
            .map(|component| component.files.len())
            .sum::<usize>()
            + manifest.license.files.len();
        let plan = catalog_install_plan("qwen-image-2512-q4_k_m", bytes).unwrap();
        assert_eq!(plan.id, "qwen-image-2512-q4_k_m");
        assert_eq!(plan.artifacts.len(), declared_files);
        assert!(plan
            .allowed_hosts
            .iter()
            .any(|host| host == "huggingface.co"));
        assert!(plan
            .allowed_hosts
            .iter()
            .any(|host| host == "cas-bridge.xethub.hf.co"));
        assert!(plan.artifacts.iter().all(|artifact| {
            artifact.size_bytes > 0
                && artifact.sha256.len() == 64
                && artifact.source.contains("/resolve/")
        }));
    }

    #[test]
    fn catalog_plan_rejects_an_identity_mismatch() {
        let bytes =
            include_bytes!("../../../reference/image/qwen/manifests/qwen-image-2512-q4_k_m.json")
                .to_vec();
        let error = catalog_install_plan("qwen-image-2512-q8_0", bytes).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match requested bundle"));
    }

    #[test]
    fn audited_qwen_catalogs_localize_without_urls_or_absolute_paths() {
        for (catalog, expected_id) in [
            (QWEN_IMAGE_2512_BF16_MANIFEST, "qwen-image-2512-bf16"),
            (
                QWEN_IMAGE_EDIT_2511_BF16_MANIFEST,
                "qwen-image-edit-2511-bf16",
            ),
        ] {
            let bytes = localize_catalog_manifest(catalog).unwrap();
            let manifest =
                BundleManifest::from_json_bytes(&bytes, ManifestMode::LocalImport).unwrap();
            assert_eq!(manifest.id, expected_id);
            assert!(manifest
                .components
                .iter()
                .flat_map(|component| &component.files)
                .all(|file| file.source.is_none() && file.source_kind.as_deref() == Some("local")));
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(!text.contains(":\\"));
            assert!(!text.contains("/resolve/"));
        }
    }

    #[test]
    fn raw_diffusers_discovery_rejects_unknown_or_remote_code_pipelines() {
        let root = std::env::temp_dir().join(format!(
            "xrt-image-raw-discovery-test-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("model_index.json"),
            br#"{"_class_name":"CustomPipeline"}"#,
        )
        .unwrap();
        let error = synthesize_audited_raw_diffusers_manifest(&root).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported raw Diffusers pipeline"));

        fs::write(
            root.join("model_index.json"),
            br#"{"_class_name":"QwenImagePipeline","trust_remote_code":true}"#,
        )
        .unwrap();
        let error = synthesize_audited_raw_diffusers_manifest(&root).unwrap_err();
        assert!(error
            .to_string()
            .contains("never executes remote model code"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recognized_raw_diffusers_class_still_requires_exact_audited_bytes() {
        let root = std::env::temp_dir().join(format!(
            "xrt-image-raw-exact-test-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("model_index.json"),
            br#"{"_class_name":"QwenImagePipeline"}"#,
        )
        .unwrap();
        let bytes = synthesize_audited_raw_diffusers_manifest(&root).unwrap();
        let manifest = BundleManifest::from_json_bytes(&bytes, ManifestMode::LocalImport).unwrap();
        assert_eq!(manifest.id, "qwen-image-2512-bf16");
        let error = ImageModelBundle::open_local_import(&root, &bytes).unwrap_err();
        assert!(matches!(
            error.kind(),
            xrt_image::ImageErrorKind::CorruptComponent
                | xrt_image::ImageErrorKind::MissingComponent
        ));
        assert!(!root.join("xrt.bundle.json").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn development_catalog_resolution_is_path_contained() {
        let path = resolve_catalog_manifest("qwen-image-2512-q4_k_m", None, None).unwrap();
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("qwen-image-2512-q4_k_m.json")
        );
        assert!(resolve_catalog_manifest("../outside", None, None).is_err());
    }

    #[test]
    fn local_import_validates_and_atomically_installs_for_offline_resolution() {
        let root = std::env::temp_dir().join(format!(
            "xrt-image-import-test-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let source = root.join("source");
        let cache = root.join("cache");
        fs::create_dir_all(&source).unwrap();
        let id = local_bundle_fixture(&source);
        run_import(ImageImportArgs {
            path: source,
            install: true,
            cache_dir: Some(cache.clone()),
        })
        .unwrap();
        let installed = ModelHub::with_cache_dir(&cache)
            .unwrap()
            .resolve_installed_bundle(&id, None)
            .unwrap();
        let bundle = ImageModelBundle::open(installed).unwrap();
        assert_eq!(bundle.manifest().id, id);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_write_is_create_new_by_default_and_replaces_explicitly() {
        let root = std::env::temp_dir().join(format!(
            "xrt-image-cli-test-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("output.png");
        ensure_create_new_output(&path).unwrap();
        atomic_write(&path, b"first", false).unwrap();
        assert_eq!(
            ensure_create_new_output(&path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(&path).unwrap(), b"first");
        assert_eq!(
            atomic_write(&path, b"second", false).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        atomic_write(&path, b"second", true).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        fs::remove_dir_all(root).unwrap();
    }
}
