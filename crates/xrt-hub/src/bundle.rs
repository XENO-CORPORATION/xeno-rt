use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::IpAddr,
    path::{Component, Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::Url;
use xrt_core::{Result, XrtError};

use crate::ModelHub;

const BUNDLE_MANIFEST_FILE: &str = "xrt.bundle.json";
const MAX_BUNDLE_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const BUNDLE_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const DEFAULT_LOCK_WAIT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleArtifact {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleInstallPlan {
    pub id: String,
    pub digest: String,
    pub manifest_bytes: Vec<u8>,
    pub artifacts: Vec<BundleArtifact>,
    pub allowed_hosts: Vec<String>,
    pub max_total_bytes: u64,
    pub lock_wait: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleImportArtifact {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleImportPlan {
    pub id: String,
    pub digest: String,
    pub manifest_bytes: Vec<u8>,
    pub artifacts: Vec<BundleImportArtifact>,
    pub max_total_bytes: u64,
    pub lock_wait: Duration,
}

impl BundleImportPlan {
    pub fn new(
        id: impl Into<String>,
        digest: impl Into<String>,
        manifest_bytes: Vec<u8>,
        artifacts: Vec<BundleImportArtifact>,
        max_total_bytes: u64,
    ) -> Self {
        Self {
            id: id.into(),
            digest: digest.into(),
            manifest_bytes,
            artifacts,
            max_total_bytes,
            lock_wait: DEFAULT_LOCK_WAIT,
        }
    }

    fn validate(&self) -> Result<()> {
        validate_plan_header(
            &self.id,
            &self.digest,
            &self.manifest_bytes,
            self.max_total_bytes,
            self.lock_wait,
        )?;
        if self.artifacts.is_empty() {
            return Err(XrtError::Runtime(
                "bundle import artifacts must be non-empty".to_string(),
            ));
        }
        let mut paths = HashSet::new();
        let mut total = 0u64;
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.path)?;
            if !paths.insert(artifact.path.as_str()) {
                return Err(XrtError::Runtime(format!(
                    "duplicate bundle import artifact path `{}`",
                    artifact.path
                )));
            }
            if artifact.size_bytes == 0 {
                return Err(XrtError::Runtime(format!(
                    "bundle import artifact `{}` has zero size",
                    artifact.path
                )));
            }
            validate_sha256(&artifact.sha256, "import artifact SHA-256")?;
            total = total.checked_add(artifact.size_bytes).ok_or_else(|| {
                XrtError::Runtime("bundle import byte count overflowed".to_string())
            })?;
        }
        if total > self.max_total_bytes {
            return Err(XrtError::Runtime(format!(
                "bundle import declares {total} bytes, above the {}-byte cap",
                self.max_total_bytes
            )));
        }
        Ok(())
    }
}

impl BundleInstallPlan {
    pub fn new(
        id: impl Into<String>,
        digest: impl Into<String>,
        manifest_bytes: Vec<u8>,
        artifacts: Vec<BundleArtifact>,
        allowed_hosts: Vec<String>,
        max_total_bytes: u64,
    ) -> Self {
        Self {
            id: id.into(),
            digest: digest.into(),
            manifest_bytes,
            artifacts,
            allowed_hosts,
            max_total_bytes,
            lock_wait: DEFAULT_LOCK_WAIT,
        }
    }

    fn validate(&self) -> Result<()> {
        validate_plan_header(
            &self.id,
            &self.digest,
            &self.manifest_bytes,
            self.max_total_bytes,
            self.lock_wait,
        )?;
        if self.artifacts.is_empty() {
            return Err(XrtError::Runtime(
                "bundle artifacts must be non-zero".to_string(),
            ));
        }
        let allowed_hosts = self
            .allowed_hosts
            .iter()
            .map(|host| host.trim().to_ascii_lowercase())
            .collect::<HashSet<_>>();
        if allowed_hosts.is_empty() || allowed_hosts.iter().any(|host| !safe_host(host)) {
            return Err(XrtError::Runtime(
                "bundle allowed_hosts must contain only reviewed DNS hosts".to_string(),
            ));
        }
        let mut paths = HashSet::new();
        let mut total = 0u64;
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.path)?;
            if !paths.insert(artifact.path.as_str()) {
                return Err(XrtError::Runtime(format!(
                    "duplicate bundle artifact path `{}`",
                    artifact.path
                )));
            }
            if artifact.size_bytes == 0 {
                return Err(XrtError::Runtime(format!(
                    "bundle artifact `{}` has zero size",
                    artifact.path
                )));
            }
            validate_sha256(&artifact.sha256, "artifact SHA-256")?;
            total = total.checked_add(artifact.size_bytes).ok_or_else(|| {
                XrtError::Runtime("bundle aggregate byte count overflowed".to_string())
            })?;
            let url = validate_source_url(&artifact.source, false)?;
            let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
            if !allowed_hosts.contains(&host) {
                return Err(XrtError::Runtime(format!(
                    "bundle source host `{host}` is not in allowed_hosts"
                )));
            }
        }
        if total > self.max_total_bytes {
            return Err(XrtError::Runtime(format!(
                "bundle declares {total} bytes, above the {}-byte install cap",
                self.max_total_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleInstallProgress {
    pub artifact_path: String,
    pub artifact_downloaded: u64,
    pub artifact_total: u64,
    pub bundle_downloaded: u64,
    pub bundle_total: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledBundle {
    pub id: String,
    pub digest: String,
    pub path: PathBuf,
    pub was_cached: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BundleRecoveryReport {
    pub scanned: usize,
    pub reindexed: usize,
    pub invalid: usize,
    pub ambiguous: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BundleIndexEntry {
    schema_version: u32,
    id: String,
    digest: String,
    relative_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryManifest {
    schema_version: u32,
    id: String,
    family: String,
    revision: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    source_revisions: BTreeMap<String, String>,
    capabilities: Vec<String>,
    license: RecoveryLicense,
    quantization: String,
    components: Vec<RecoveryComponent>,
    limits: RecoveryLimits,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryComponent {
    role: String,
    format: String,
    #[serde(default)]
    optional: bool,
    files: Vec<RecoveryFile>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryLicense {
    spdx: String,
    evidence: String,
    #[serde(default)]
    files: Vec<RecoveryFile>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryFile {
    path: String,
    size_bytes: u64,
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_kind: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryLimits {
    max_sequence_length: usize,
    max_width: u32,
    max_height: u32,
    max_pixels: u64,
}

impl ModelHub {
    pub fn install_bundle(&self, plan: &BundleInstallPlan) -> Result<InstalledBundle> {
        self.install_bundle_with_control(plan, || false, |_| {})
    }

    pub fn install_bundle_with_control<C, P>(
        &self,
        plan: &BundleInstallPlan,
        mut is_cancelled: C,
        mut on_progress: P,
    ) -> Result<InstalledBundle>
    where
        C: FnMut() -> bool,
        P: FnMut(BundleInstallProgress),
    {
        self.install_bundle_with_fetcher(
            plan,
            &mut is_cancelled,
            &mut on_progress,
            |artifact, destination, cancelled, progress| {
                self.download_bundle_artifact(
                    artifact,
                    &plan.allowed_hosts,
                    destination,
                    cancelled,
                    progress,
                )
            },
        )
    }

    pub fn import_bundle(
        &self,
        source_root: impl AsRef<Path>,
        plan: &BundleImportPlan,
    ) -> Result<InstalledBundle> {
        self.import_bundle_with_control(source_root, plan, || false, |_| {})
    }

    pub fn import_bundle_with_control<C, P>(
        &self,
        source_root: impl AsRef<Path>,
        plan: &BundleImportPlan,
        mut is_cancelled: C,
        mut on_progress: P,
    ) -> Result<InstalledBundle>
    where
        C: FnMut() -> bool,
        P: FnMut(BundleInstallProgress),
    {
        plan.validate()?;
        let source_metadata = fs::symlink_metadata(source_root.as_ref())?;
        if source_metadata.file_type().is_symlink() || !source_metadata.is_dir() {
            return Err(XrtError::Runtime(
                "bundle import root must be a real directory, not a symlink".to_string(),
            ));
        }
        let source_root = fs::canonicalize(source_root.as_ref())?;
        let internal_plan = BundleInstallPlan {
            id: plan.id.clone(),
            digest: plan.digest.clone(),
            manifest_bytes: plan.manifest_bytes.clone(),
            artifacts: plan
                .artifacts
                .iter()
                .map(|artifact| BundleArtifact {
                    path: artifact.path.clone(),
                    size_bytes: artifact.size_bytes,
                    sha256: artifact.sha256.clone(),
                    source: String::new(),
                })
                .collect(),
            allowed_hosts: Vec::new(),
            max_total_bytes: plan.max_total_bytes,
            lock_wait: plan.lock_wait,
        };
        self.install_validated_bundle(
            &internal_plan,
            &mut is_cancelled,
            &mut on_progress,
            |artifact, destination, cancelled, progress| {
                copy_local_bundle_artifact(&source_root, artifact, destination, cancelled, progress)
            },
        )
    }

    pub fn resolve_installed_bundle(&self, id: &str, digest: Option<&str>) -> Result<PathBuf> {
        if !safe_identifier(id) {
            return Err(XrtError::Runtime("invalid bundle id".to_string()));
        }
        let index_path = self.cache_dir.join("manifests").join(format!("{id}.json"));
        let bytes = fs::read(&index_path).map_err(|error| {
            XrtError::Runtime(format!("bundle `{id}` is not installed: {error}"))
        })?;
        if bytes.len() > MAX_BUNDLE_MANIFEST_BYTES {
            return Err(XrtError::Runtime(
                "bundle index entry is oversized".to_string(),
            ));
        }
        let entry: BundleIndexEntry = serde_json::from_slice(&bytes).map_err(|error| {
            XrtError::Runtime(format!("bundle index entry is corrupt: {error}"))
        })?;
        if entry.schema_version != 1 || entry.id != id {
            return Err(XrtError::Runtime(
                "bundle index identity does not match its filename".to_string(),
            ));
        }
        validate_sha256(&entry.digest, "indexed bundle digest")?;
        if digest.is_some_and(|expected| expected != entry.digest) {
            return Err(XrtError::Runtime(format!(
                "installed bundle `{id}` has digest {}, not {}",
                entry.digest,
                digest.unwrap_or_default()
            )));
        }
        validate_relative_path(&entry.relative_path)?;
        let root = fs::canonicalize(&self.cache_dir)?;
        let path = fs::canonicalize(root.join(&entry.relative_path))?;
        if !path.starts_with(&root) || !path.is_dir() || !path.join(BUNDLE_MANIFEST_FILE).is_file()
        {
            return Err(XrtError::Runtime(
                "indexed bundle path is missing or escapes the cache".to_string(),
            ));
        }
        Ok(path)
    }

    pub fn recover_bundle_staging(&self, minimum_age: Duration) -> Result<usize> {
        let staging_root = self.cache_dir.join(".staging");
        if !staging_root.exists() {
            return Ok(0);
        }
        let now = std::time::SystemTime::now();
        let mut removed = 0usize;
        for entry in fs::read_dir(&staging_root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let modified = entry.metadata()?.modified()?;
            if now.duration_since(modified).unwrap_or_default() < minimum_age {
                continue;
            }
            fs::remove_dir_all(entry.path())?;
            removed += 1;
        }
        Ok(removed)
    }

    /// Recover complete bundles orphaned by a crash between the atomic bundle
    /// directory rename and the atomic index publication. Existing valid index
    /// entries are never replaced, and multiple valid unindexed digests for one
    /// ID remain untouched because choosing between them could be a rollback.
    pub fn recover_orphaned_bundles(&self, minimum_age: Duration) -> Result<BundleRecoveryReport> {
        let bundles_root = self.cache_dir.join("bundles");
        if !bundles_root.exists() {
            return Ok(BundleRecoveryReport::default());
        }
        let now = std::time::SystemTime::now();
        let mut report = BundleRecoveryReport::default();
        for id_entry in fs::read_dir(&bundles_root)? {
            let id_entry = id_entry?;
            if !id_entry.file_type()?.is_dir() {
                continue;
            }
            let id = id_entry.file_name().to_string_lossy().into_owned();
            if !safe_identifier(&id) || valid_indexed_bundle(&self.cache_dir, &id) {
                continue;
            }
            let mut valid = Vec::new();
            for digest_entry in fs::read_dir(id_entry.path())? {
                let digest_entry = digest_entry?;
                if !digest_entry.file_type()?.is_dir() {
                    continue;
                }
                let modified = digest_entry.metadata()?.modified()?;
                if now.duration_since(modified).unwrap_or_default() < minimum_age {
                    continue;
                }
                report.scanned += 1;
                let digest = digest_entry.file_name().to_string_lossy().into_owned();
                if validate_sha256(&digest, "orphaned bundle digest").is_err() {
                    report.invalid += 1;
                    continue;
                }
                let mut never_cancelled = || false;
                let lock = acquire_bundle_lock(
                    &self.cache_dir.join(".locks"),
                    &digest,
                    DEFAULT_LOCK_WAIT,
                    &mut never_cancelled,
                )?;
                let verified = verify_recovery_candidate(digest_entry.path(), &id, &digest);
                drop(lock);
                match verified {
                    Ok(()) => valid.push(digest),
                    Err(_) => report.invalid += 1,
                }
            }
            match valid.as_slice() {
                [digest] => {
                    publish_index_entry(&self.cache_dir, &id, digest)?;
                    report.reindexed += 1;
                }
                [] => {}
                candidates => report.ambiguous += candidates.len(),
            }
        }
        Ok(report)
    }

    fn install_bundle_with_fetcher<C, P, F>(
        &self,
        plan: &BundleInstallPlan,
        is_cancelled: &mut C,
        on_progress: &mut P,
        fetch: F,
    ) -> Result<InstalledBundle>
    where
        C: FnMut() -> bool,
        P: FnMut(BundleInstallProgress),
        F: FnMut(
            &BundleArtifact,
            &Path,
            &mut dyn FnMut() -> bool,
            &mut dyn FnMut(u64),
        ) -> Result<()>,
    {
        plan.validate()?;
        self.install_validated_bundle(plan, is_cancelled, on_progress, fetch)
    }

    fn install_validated_bundle<C, P, F>(
        &self,
        plan: &BundleInstallPlan,
        is_cancelled: &mut C,
        on_progress: &mut P,
        mut fetch: F,
    ) -> Result<InstalledBundle>
    where
        C: FnMut() -> bool,
        P: FnMut(BundleInstallProgress),
        F: FnMut(
            &BundleArtifact,
            &Path,
            &mut dyn FnMut() -> bool,
            &mut dyn FnMut(u64),
        ) -> Result<()>,
    {
        if is_cancelled() {
            return Err(cancelled_error());
        }
        let _lock = acquire_bundle_lock(
            &self.cache_dir.join(".locks"),
            &plan.digest,
            plan.lock_wait,
            is_cancelled,
        )?;
        let final_dir = self
            .cache_dir
            .join("bundles")
            .join(&plan.id)
            .join(&plan.digest);
        if final_dir.exists() {
            verify_installed(&final_dir, plan)?;
            publish_index(&self.cache_dir, plan)?;
            let path = fs::canonicalize(&final_dir)?;
            return Ok(InstalledBundle {
                id: plan.id.clone(),
                digest: plan.digest.clone(),
                path,
                was_cached: true,
            });
        }

        let staging_root = self.cache_dir.join(".staging");
        fs::create_dir_all(&staging_root)?;
        let staging = create_staging_dir(&staging_root, &plan.id, &plan.digest)?;
        let result = (|| {
            write_new_synced(&staging.join(BUNDLE_MANIFEST_FILE), &plan.manifest_bytes)?;
            let bundle_total = plan.artifacts.iter().map(|item| item.size_bytes).sum();
            let mut completed = 0u64;
            for artifact in &plan.artifacts {
                if is_cancelled() {
                    return Err(cancelled_error());
                }
                let destination = staging.join(&artifact.path);
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                let artifact_path = artifact.path.clone();
                let artifact_total = artifact.size_bytes;
                let completed_before = completed;
                let mut progress = |downloaded: u64| {
                    on_progress(BundleInstallProgress {
                        artifact_path: artifact_path.clone(),
                        artifact_downloaded: downloaded,
                        artifact_total,
                        bundle_downloaded: completed_before.saturating_add(downloaded),
                        bundle_total,
                    });
                };
                fetch(artifact, &destination, is_cancelled, &mut progress)?;
                verify_artifact(&destination, artifact)?;
                completed = completed
                    .checked_add(artifact.size_bytes)
                    .ok_or_else(|| XrtError::Runtime("bundle progress overflowed".to_string()))?;
            }
            sync_directory(&staging)?;
            let final_parent = final_dir.parent().ok_or_else(|| {
                XrtError::Runtime("bundle final directory has no parent".to_string())
            })?;
            fs::create_dir_all(final_parent)?;
            fs::rename(&staging, &final_dir)?;
            sync_directory(final_parent)?;
            publish_index(&self.cache_dir, plan)?;
            let path = fs::canonicalize(&final_dir)?;
            Ok(InstalledBundle {
                id: plan.id.clone(),
                digest: plan.digest.clone(),
                path,
                was_cached: false,
            })
        })();
        if result.is_err() && staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    fn download_bundle_artifact(
        &self,
        artifact: &BundleArtifact,
        allowed_hosts: &[String],
        destination: &Path,
        is_cancelled: &mut dyn FnMut() -> bool,
        on_progress: &mut dyn FnMut(u64),
    ) -> Result<()> {
        let agent = ureq::AgentBuilder::new()
            .user_agent("xrt-hub/0.2 bundle-installer")
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(300))
            .redirects(0)
            .build();
        let original = validate_source_url(&artifact.source, false)?;
        let allowed_hosts = allowed_hosts
            .iter()
            .map(|host| host.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let mut current = original.clone();
        let mut response = None;
        for redirect_count in 0..=MAX_REDIRECTS {
            if is_cancelled() {
                return Err(cancelled_error());
            }
            let mut request = agent.get(current.as_str());
            if same_origin(&current, &original) {
                if let Some(token) = &self.auth_token {
                    request = request.set("Authorization", &format!("Bearer {token}"));
                }
            }
            match request.call() {
                Ok(candidate) if (300..400).contains(&candidate.status()) => {
                    if redirect_count == MAX_REDIRECTS {
                        return Err(XrtError::Runtime(
                            "bundle download exceeded redirect limit".to_string(),
                        ));
                    }
                    current = reviewed_redirect_target(&current, &candidate, &allowed_hosts)?;
                }
                Ok(candidate) => {
                    response = Some(candidate);
                    break;
                }
                Err(ureq::Error::Status(status, candidate)) if (300..400).contains(&status) => {
                    if redirect_count == MAX_REDIRECTS {
                        return Err(XrtError::Runtime(
                            "bundle download exceeded redirect limit".to_string(),
                        ));
                    }
                    current = reviewed_redirect_target(&current, &candidate, &allowed_hosts)?;
                }
                Err(error) => return Err(map_bundle_ureq_error(error)),
            }
        }
        let response = response.ok_or_else(|| {
            XrtError::Runtime("bundle download did not produce a response".to_string())
        })?;
        if let Some(length) = response.header("Content-Length") {
            let length = length.parse::<u64>().map_err(|error| {
                XrtError::Runtime(format!("invalid bundle Content-Length: {error}"))
            })?;
            if length != artifact.size_bytes {
                return Err(XrtError::Runtime(format!(
                    "bundle Content-Length {length} does not match declared {}",
                    artifact.size_bytes
                )));
            }
        }
        let mut source = response.into_reader();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        let mut buffer = vec![0u8; BUNDLE_BUFFER_BYTES];
        let mut downloaded = 0u64;
        on_progress(0);
        loop {
            if is_cancelled() {
                return Err(cancelled_error());
            }
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            downloaded = downloaded.checked_add(read as u64).ok_or_else(|| {
                XrtError::Runtime("bundle download byte count overflowed".to_string())
            })?;
            if downloaded > artifact.size_bytes {
                return Err(XrtError::Runtime(format!(
                    "bundle artifact `{}` exceeded its declared size",
                    artifact.path
                )));
            }
            file.write_all(&buffer[..read])?;
            on_progress(downloaded);
        }
        file.flush()?;
        file.sync_all()?;
        if downloaded != artifact.size_bytes {
            return Err(XrtError::Runtime(format!(
                "bundle artifact `{}` ended at {downloaded} bytes, expected {}",
                artifact.path, artifact.size_bytes
            )));
        }
        Ok(())
    }
}

fn reviewed_redirect_target(
    current: &Url,
    response: &ureq::Response,
    allowed_hosts: &HashSet<String>,
) -> Result<Url> {
    let location = response
        .header("Location")
        .ok_or_else(|| XrtError::Runtime("bundle redirect omitted Location".to_string()))?;
    let target = current
        .join(location)
        .map_err(|error| XrtError::Runtime(format!("invalid bundle redirect: {error}")))?;
    validate_redirect_url(&target)?;
    let redirect_host = target.host_str().unwrap_or_default().to_ascii_lowercase();
    if !allowed_hosts.contains(&redirect_host) {
        return Err(XrtError::Runtime(format!(
            "bundle redirect host `{redirect_host}` is not reviewed"
        )));
    }
    Ok(target)
}

struct BundleLock {
    file: File,
}

impl Drop for BundleLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

fn acquire_bundle_lock(
    lock_root: &Path,
    digest: &str,
    wait: Duration,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<BundleLock> {
    fs::create_dir_all(lock_root)?;
    let path = lock_root.join(format!("{digest}.lock"));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let started = Instant::now();
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(BundleLock { file }),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if is_cancelled() {
                    return Err(cancelled_error());
                }
                if started.elapsed() >= wait {
                    return Err(XrtError::Runtime(
                        "timed out waiting for bundle install lock".to_string(),
                    ));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn create_staging_dir(root: &Path, id: &str, digest: &str) -> Result<PathBuf> {
    for _ in 0..16 {
        let nonce = rand::random::<u64>();
        let path = root.join(format!("{id}-{}-{nonce:016x}", &digest[..16]));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(XrtError::Runtime(
        "failed to allocate a unique bundle staging directory".to_string(),
    ))
}

fn copy_local_bundle_artifact(
    source_root: &Path,
    artifact: &BundleArtifact,
    destination: &Path,
    is_cancelled: &mut dyn FnMut() -> bool,
    on_progress: &mut dyn FnMut(u64),
) -> Result<()> {
    let source = contained_regular_file(source_root, &artifact.path)?;
    let mut input = File::open(source)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut copied = 0u64;
    let mut buffer = vec![0u8; BUNDLE_BUFFER_BYTES];
    on_progress(0);
    loop {
        if is_cancelled() {
            return Err(cancelled_error());
        }
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        copied = copied
            .checked_add(read as u64)
            .ok_or_else(|| XrtError::Runtime("bundle import byte count overflowed".to_string()))?;
        if copied > artifact.size_bytes {
            return Err(XrtError::Runtime(format!(
                "bundle import artifact `{}` grew beyond its declared size",
                artifact.path
            )));
        }
        on_progress(copied);
    }
    output.sync_all()?;
    if copied != artifact.size_bytes {
        return Err(XrtError::Runtime(format!(
            "bundle import artifact `{}` copied {copied} bytes, expected {}",
            artifact.path, artifact.size_bytes
        )));
    }
    Ok(())
}

fn contained_regular_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    let mut current = root.to_path_buf();
    for component in Path::new(relative).components() {
        let Component::Normal(component) = component else {
            return Err(XrtError::Runtime(
                "bundle import path contains a non-normal component".to_string(),
            ));
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            return Err(XrtError::Runtime(format!(
                "bundle import artifact `{relative}` traverses a symlink"
            )));
        }
    }
    let canonical = fs::canonicalize(&current)?;
    let metadata = fs::metadata(&canonical)?;
    if !canonical.starts_with(root) || !metadata.is_file() {
        return Err(XrtError::Runtime(format!(
            "bundle import artifact `{relative}` escapes the source root or is not a regular file"
        )));
    }
    Ok(canonical)
}

fn verify_installed(root: &Path, plan: &BundleInstallPlan) -> Result<()> {
    let manifest = fs::read(root.join(BUNDLE_MANIFEST_FILE))?;
    if manifest != plan.manifest_bytes {
        return Err(XrtError::Runtime(
            "installed bundle manifest bytes do not match the requested digest".to_string(),
        ));
    }
    for artifact in &plan.artifacts {
        verify_artifact(&root.join(&artifact.path), artifact)?;
    }
    Ok(())
}

fn verify_artifact(path: &Path, artifact: &BundleArtifact) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(XrtError::Runtime(format!(
            "bundle artifact `{}` is not a regular file",
            artifact.path
        )));
    }
    if metadata.len() != artifact.size_bytes {
        return Err(XrtError::Runtime(format!(
            "bundle artifact `{}` has size {}, expected {}",
            artifact.path,
            metadata.len(),
            artifact.size_bytes
        )));
    }
    let actual = sha256_file(path)?;
    if actual != artifact.sha256 {
        return Err(XrtError::Runtime(format!(
            "bundle artifact `{}` failed SHA-256 verification",
            artifact.path
        )));
    }
    Ok(())
}

fn valid_indexed_bundle(cache_root: &Path, id: &str) -> bool {
    let path = cache_root.join("manifests").join(format!("{id}.json"));
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    if bytes.len() > MAX_BUNDLE_MANIFEST_BYTES {
        return false;
    }
    let Ok(entry) = serde_json::from_slice::<BundleIndexEntry>(&bytes) else {
        return false;
    };
    if entry.schema_version != 1
        || entry.id != id
        || validate_sha256(&entry.digest, "indexed bundle digest").is_err()
        || validate_relative_path(&entry.relative_path).is_err()
        || entry.relative_path != format!("bundles/{id}/{}", entry.digest)
    {
        return false;
    }
    let root = cache_root.join(&entry.relative_path);
    root.is_dir() && root.join(BUNDLE_MANIFEST_FILE).is_file()
}

fn verify_recovery_candidate(root: PathBuf, id: &str, digest: &str) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let manifest_path = root.join(BUNDLE_MANIFEST_FILE);
    let manifest_metadata = fs::symlink_metadata(&manifest_path)?;
    if manifest_metadata.file_type().is_symlink()
        || !manifest_metadata.is_file()
        || manifest_metadata.len() as usize > MAX_BUNDLE_MANIFEST_BYTES
    {
        return Err(XrtError::Runtime(
            "orphaned bundle manifest is missing, linked, or oversized".to_string(),
        ));
    }
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: RecoveryManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        XrtError::Runtime(format!("orphaned bundle manifest is invalid: {error}"))
    })?;
    if manifest.schema_version != 1 || manifest.id != id {
        return Err(XrtError::Runtime(
            "orphaned bundle manifest identity is inconsistent".to_string(),
        ));
    }
    if canonical_recovery_digest(&manifest)? != digest {
        return Err(XrtError::Runtime(
            "orphaned bundle directory digest does not match its manifest".to_string(),
        ));
    }

    let mut declared = BTreeSet::from([BUNDLE_MANIFEST_FILE.to_string()]);
    for file in manifest
        .components
        .iter()
        .flat_map(|component| component.files.iter())
        .chain(manifest.license.files.iter())
    {
        validate_relative_path(&file.path)?;
        validate_sha256(&file.sha256, "orphaned artifact SHA-256")?;
        if file.size_bytes == 0 || !declared.insert(file.path.clone()) {
            return Err(XrtError::Runtime(format!(
                "orphaned bundle has a zero-sized or duplicate declaration `{}`",
                file.path
            )));
        }
        verify_recovery_file(&root.join(&file.path), file)?;
    }

    let mut actual = BTreeSet::new();
    collect_bundle_files(&root, &root, &mut actual)?;
    if actual != declared {
        return Err(XrtError::Runtime(format!(
            "orphaned bundle file set differs from its manifest: actual={actual:?}, declared={declared:?}"
        )));
    }
    Ok(())
}

fn verify_recovery_file(path: &Path, file: &RecoveryFile) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(XrtError::Runtime(format!(
            "orphaned artifact `{}` is not a regular file",
            file.path
        )));
    }
    if metadata.len() != file.size_bytes || sha256_file(path)? != file.sha256 {
        return Err(XrtError::Runtime(format!(
            "orphaned artifact `{}` failed size or SHA-256 verification",
            file.path
        )));
    }
    Ok(())
}

fn collect_bundle_files(root: &Path, directory: &Path, files: &mut BTreeSet<String>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            return Err(XrtError::Runtime(
                "orphaned bundle contains a symlink".to_string(),
            ));
        }
        if metadata.is_dir() {
            collect_bundle_files(root, &entry.path(), files)?;
        } else if metadata.is_file() {
            let path = entry.path();
            let relative = path.strip_prefix(root).map_err(|_| {
                XrtError::Runtime("orphaned bundle file escaped its root".to_string())
            })?;
            let relative = relative
                .components()
                .map(|component| match component {
                    Component::Normal(value) => {
                        value.to_str().map(ToOwned::to_owned).ok_or_else(|| {
                            XrtError::Runtime("orphaned bundle path is not valid UTF-8".to_string())
                        })
                    }
                    _ => Err(XrtError::Runtime(
                        "orphaned bundle path contains an unsafe component".to_string(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?
                .join("/");
            files.insert(relative);
        } else {
            return Err(XrtError::Runtime(
                "orphaned bundle contains a non-file filesystem object".to_string(),
            ));
        }
    }
    Ok(())
}

fn canonical_recovery_digest(manifest: &RecoveryManifest) -> Result<String> {
    let mut value = serde_json::to_value(manifest).map_err(|error| {
        XrtError::Runtime(format!("failed to normalize orphaned manifest: {error}"))
    })?;
    let object = value.as_object_mut().ok_or_else(|| {
        XrtError::Runtime("orphaned bundle manifest must be an object".to_string())
    })?;
    if let Some(Value::Array(components)) = object.get_mut("components") {
        for component in components.iter_mut() {
            sort_recovery_files(component.get_mut("files"));
        }
        components.sort_by(|left, right| {
            recovery_component_key(left).cmp(&recovery_component_key(right))
        });
    }
    if let Some(Value::Object(license)) = object.get_mut("license") {
        sort_recovery_files(license.get_mut("files"));
    }
    let mut canonical = Vec::new();
    write_canonical_json(&value, &mut canonical)?;
    let mut hash = Sha256::new();
    hash.update(b"xrt-bundle-v1\0");
    hash.update(canonical);
    Ok(format!("{:x}", hash.finalize()))
}

fn sort_recovery_files(value: Option<&mut Value>) {
    if let Some(Value::Array(files)) = value {
        files.sort_by(|left, right| recovery_file_path(left).cmp(recovery_file_path(right)));
    }
}

fn recovery_component_key(value: &Value) -> (&str, &str) {
    let role = value.get("role").and_then(Value::as_str).unwrap_or("");
    let path = value
        .get("files")
        .and_then(Value::as_array)
        .and_then(|files| files.first())
        .map(recovery_file_path)
        .unwrap_or("");
    (role, path)
}

fn recovery_file_path(value: &Value) -> &str {
    value.get("path").and_then(Value::as_str).unwrap_or("")
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::String(value) => output.extend_from_slice(
            serde_json::to_string(value)
                .map_err(|error| XrtError::Runtime(error.to_string()))?
                .as_bytes(),
        ),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(
                    serde_json::to_string(key)
                        .map_err(|error| XrtError::Runtime(error.to_string()))?
                        .as_bytes(),
                );
                output.push(b':');
                write_canonical_json(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn publish_index(cache_root: &Path, plan: &BundleInstallPlan) -> Result<()> {
    publish_index_entry(cache_root, &plan.id, &plan.digest)
}

fn publish_index_entry(cache_root: &Path, id: &str, digest: &str) -> Result<()> {
    let manifests = cache_root.join("manifests");
    fs::create_dir_all(&manifests)?;
    let entry = BundleIndexEntry {
        schema_version: 1,
        id: id.to_string(),
        digest: digest.to_string(),
        relative_path: format!("bundles/{id}/{digest}"),
    };
    let bytes = serde_json::to_vec(&entry)
        .map_err(|error| XrtError::Runtime(format!("failed to serialize bundle index: {error}")))?;
    let final_path = manifests.join(format!("{id}.json"));
    let temporary = manifests.join(format!(".{id}.{}.tmp", rand::random::<u64>()));
    write_new_synced(&temporary, &bytes)?;
    if final_path.exists() && fs::read(&final_path)? == bytes {
        fs::remove_file(&temporary)?;
    } else {
        atomic_replace(&temporary, &final_path)?;
    }
    sync_directory(&manifests)
}

#[cfg(unix)]
fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
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
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        return Err(XrtError::Runtime(
            "atomic bundle index replacement is unsupported on this platform".to_string(),
        ));
    }
    fs::rename(source, destination)?;
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; BUNDLE_BUFFER_BYTES];
    let mut hash = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(XrtError::Runtime(format!(
            "unsafe bundle-relative path `{value}`"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(XrtError::Runtime(format!(
            "{label} must be lowercase 64-hex"
        )));
    }
    Ok(())
}

fn validate_plan_header(
    id: &str,
    digest: &str,
    manifest_bytes: &[u8],
    max_total_bytes: u64,
    lock_wait: Duration,
) -> Result<()> {
    if !safe_identifier(id) {
        return Err(XrtError::Runtime(
            "bundle id must be a bounded lowercase-safe identifier".to_string(),
        ));
    }
    validate_sha256(digest, "bundle digest")?;
    if manifest_bytes.is_empty() || manifest_bytes.len() > MAX_BUNDLE_MANIFEST_BYTES {
        return Err(XrtError::Runtime(format!(
            "bundle manifest byte length must be between 1 and {MAX_BUNDLE_MANIFEST_BYTES}"
        )));
    }
    serde_json::from_slice::<serde_json::Value>(manifest_bytes).map_err(|error| {
        XrtError::Runtime(format!("bundle manifest is not valid JSON: {error}"))
    })?;
    if max_total_bytes == 0 {
        return Err(XrtError::Runtime(
            "bundle max_total_bytes must be non-zero".to_string(),
        ));
    }
    if lock_wait.is_zero() || lock_wait > Duration::from_secs(300) {
        return Err(XrtError::Runtime(
            "bundle lock wait must be between 1 millisecond and 300 seconds".to_string(),
        ));
    }
    Ok(())
}

fn validate_source_url(value: &str, allow_query: bool) -> Result<Url> {
    let url = Url::parse(value)
        .map_err(|error| XrtError::Runtime(format!("invalid bundle source URL: {error}")))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || (!allow_query && url.query().is_some())
        || url.host_str().is_none()
        || url.path().contains("/resolve/main/")
    {
        return Err(XrtError::Runtime(
            "bundle source URL must be credential-free HTTPS at an immutable revision".to_string(),
        ));
    }
    reject_private_host(&url)?;
    Ok(url)
}

fn validate_redirect_url(url: &Url) -> Result<()> {
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
    {
        return Err(XrtError::Runtime(
            "bundle redirect must remain credential-free HTTPS".to_string(),
        ));
    }
    reject_private_host(url)
}

fn reject_private_host(url: &Url) -> Result<()> {
    let host = url.host_str().unwrap_or_default();
    if !safe_host(host) {
        return Err(XrtError::Runtime(
            "bundle URL host is local, private, or malformed".to_string(),
        ));
    }
    Ok(())
}

fn safe_host(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty()
        || host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
    {
        return false;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(address)) => {
            !(address.is_private()
                || address.is_loopback()
                || address.is_link_local()
                || address.is_unspecified()
                || address.is_broadcast())
        }
        Ok(IpAddr::V6(address)) => {
            let unique_local = address.segments()[0] & 0xfe00 == 0xfc00;
            !(address.is_loopback() || address.is_unspecified() || unique_local)
        }
        Err(_) => host.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
        }),
    }
}

fn same_origin(left: &Url, right: &Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn safe_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
}

fn cancelled_error() -> XrtError {
    XrtError::Runtime("bundle installation cancelled".to_string())
}

fn map_bundle_ureq_error(error: ureq::Error) -> XrtError {
    match error {
        ureq::Error::Status(status, response) => XrtError::Runtime(format!(
            "bundle download failed with HTTP {status} {}",
            response.status_text()
        )),
        ureq::Error::Transport(error) => XrtError::Io(std::io::Error::other(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(bytes: &[u8]) -> BundleInstallPlan {
        BundleInstallPlan::new(
            "fixture-bundle",
            "ab".repeat(32),
            br#"{"schema_version":1}"#.to_vec(),
            vec![BundleArtifact {
                path: "transformer/model.gguf".to_string(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
                source: "https://example.com/model/resolve/0123456789abcdef/model.gguf".to_string(),
            }],
            vec!["example.com".to_string()],
            1024,
        )
    }

    fn recovery_plan(bytes: &[u8]) -> BundleInstallPlan {
        let artifact_sha = format!("{:x}", Sha256::digest(bytes));
        let source = "https://example.com/model/resolve/0123456789abcdef/model.gguf";
        let manifest = RecoveryManifest {
            schema_version: 1,
            id: "recovery-fixture".to_string(),
            family: "qwen-image".to_string(),
            revision: "fixture-revision".to_string(),
            source_revisions: BTreeMap::new(),
            capabilities: vec!["image.generate".to_string()],
            license: RecoveryLicense {
                spdx: "Apache-2.0".to_string(),
                evidence: "https://example.com/license".to_string(),
                files: Vec::new(),
            },
            quantization: "Q4_K_M".to_string(),
            components: vec![RecoveryComponent {
                role: "transformer".to_string(),
                format: "gguf".to_string(),
                optional: false,
                files: vec![RecoveryFile {
                    path: "transformer/model.gguf".to_string(),
                    size_bytes: bytes.len() as u64,
                    sha256: artifact_sha.clone(),
                    source: Some(source.to_string()),
                    source_kind: None,
                }],
            }],
            limits: RecoveryLimits {
                max_sequence_length: 1_024,
                max_width: 1_024,
                max_height: 1_024,
                max_pixels: 1_048_576,
            },
        };
        let digest = canonical_recovery_digest(&manifest).unwrap();
        BundleInstallPlan::new(
            manifest.id.clone(),
            digest,
            serde_json::to_vec_pretty(&manifest).unwrap(),
            vec![BundleArtifact {
                path: "transformer/model.gguf".to_string(),
                size_bytes: bytes.len() as u64,
                sha256: artifact_sha,
                source: source.to_string(),
            }],
            vec!["example.com".to_string()],
            1024,
        )
    }

    #[test]
    fn atomic_bundle_publish_is_cached_and_offline_resolvable() {
        let directory = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_cache_dir(directory.path()).unwrap();
        let bytes = b"fixture-model";
        let plan = plan(bytes);
        let mut cancelled = || false;
        let mut progress = |_| {};
        let first = hub
            .install_bundle_with_fetcher(
                &plan,
                &mut cancelled,
                &mut progress,
                |_artifact, destination, _cancelled, progress| {
                    write_new_synced(destination, bytes)?;
                    progress(bytes.len() as u64);
                    Ok(())
                },
            )
            .unwrap();
        assert!(!first.was_cached);
        assert_eq!(
            hub.resolve_installed_bundle(&plan.id, Some(&plan.digest))
                .unwrap(),
            first.path
        );
        let second = hub
            .install_bundle_with_fetcher(
                &plan,
                &mut cancelled,
                &mut progress,
                |_artifact, _destination, _cancelled, _progress| {
                    panic!("cached bundle must not download again")
                },
            )
            .unwrap();
        assert!(second.was_cached);
    }

    #[test]
    fn verified_local_import_is_atomic_cached_and_offline_resolvable() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source");
        let cache = directory.path().join("cache");
        fs::create_dir_all(source.join("transformer")).unwrap();
        let bytes = b"local-fixture-model";
        fs::write(source.join("transformer/model.gguf"), bytes).unwrap();
        let plan = BundleImportPlan::new(
            "local-fixture",
            "cd".repeat(32),
            br#"{"schema_version":1,"source_kind":"local"}"#.to_vec(),
            vec![BundleImportArtifact {
                path: "transformer/model.gguf".to_string(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }],
            1024,
        );
        let hub = ModelHub::with_cache_dir(&cache).unwrap();
        let first = hub.import_bundle(&source, &plan).unwrap();
        assert!(!first.was_cached);
        assert_eq!(
            fs::read(first.path.join("transformer/model.gguf")).unwrap(),
            bytes
        );
        assert_eq!(
            hub.resolve_installed_bundle(&plan.id, Some(&plan.digest))
                .unwrap(),
            first.path
        );
        fs::remove_dir_all(&source).unwrap();
        let second = hub.import_bundle(directory.path(), &plan).unwrap();
        assert!(second.was_cached);
    }

    #[test]
    fn local_import_rejects_path_traversal_before_copying() {
        let directory = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_cache_dir(directory.path().join("cache")).unwrap();
        let plan = BundleImportPlan::new(
            "local-fixture",
            "ef".repeat(32),
            br#"{"schema_version":1}"#.to_vec(),
            vec![BundleImportArtifact {
                path: "../escape.gguf".to_string(),
                size_bytes: 1,
                sha256: "00".repeat(32),
            }],
            1024,
        );
        assert!(hub.import_bundle(directory.path(), &plan).is_err());
        assert!(!directory
            .path()
            .join("cache/manifests/local-fixture.json")
            .exists());
    }

    #[test]
    fn failed_fetch_removes_partial_staging_and_does_not_publish_index() {
        let directory = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_cache_dir(directory.path()).unwrap();
        let plan = plan(b"fixture-model");
        let error = hub
            .install_bundle_with_fetcher(
                &plan,
                &mut || false,
                &mut |_| {},
                |_artifact, destination, _cancelled, _progress| {
                    write_new_synced(destination, b"wrong")?;
                    Err(XrtError::Runtime("injected interruption".to_string()))
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("injected interruption"));
        assert!(!directory
            .path()
            .join("manifests/fixture-bundle.json")
            .exists());
        let staging_count = fs::read_dir(directory.path().join(".staging"))
            .unwrap()
            .count();
        assert_eq!(staging_count, 0);
    }

    #[test]
    fn startup_recovery_reindexes_a_verified_orphaned_directory() {
        let directory = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_cache_dir(directory.path()).unwrap();
        let bytes = b"recovery-model";
        let plan = recovery_plan(bytes);
        hub.install_bundle_with_fetcher(
            &plan,
            &mut || false,
            &mut |_| {},
            |_artifact, destination, _cancelled, _progress| write_new_synced(destination, bytes),
        )
        .unwrap();
        fs::remove_file(
            directory
                .path()
                .join("manifests")
                .join(format!("{}.json", plan.id)),
        )
        .unwrap();

        let report = hub.recover_orphaned_bundles(Duration::ZERO).unwrap();
        assert_eq!(report.scanned, 1);
        assert_eq!(report.reindexed, 1);
        assert_eq!(report.invalid, 0);
        assert!(hub
            .resolve_installed_bundle(&plan.id, Some(&plan.digest))
            .is_ok());
    }

    #[test]
    fn startup_recovery_never_indexes_a_tampered_orphan() {
        let directory = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_cache_dir(directory.path()).unwrap();
        let bytes = b"recovery-model";
        let plan = recovery_plan(bytes);
        let installed = hub
            .install_bundle_with_fetcher(
                &plan,
                &mut || false,
                &mut |_| {},
                |_artifact, destination, _cancelled, _progress| {
                    write_new_synced(destination, bytes)
                },
            )
            .unwrap();
        fs::remove_file(
            directory
                .path()
                .join("manifests")
                .join(format!("{}.json", plan.id)),
        )
        .unwrap();
        fs::write(
            installed.path.join("transformer/model.gguf"),
            b"tampered-model!",
        )
        .unwrap();

        let report = hub.recover_orphaned_bundles(Duration::ZERO).unwrap();
        assert_eq!(report.reindexed, 0);
        assert_eq!(report.invalid, 1);
        assert!(!directory
            .path()
            .join("manifests/recovery-fixture.json")
            .exists());
    }

    #[test]
    fn plan_rejects_mutable_urls_and_path_traversal() {
        let mut candidate = plan(b"x");
        candidate.artifacts[0].source =
            "https://example.com/model/resolve/main/model.gguf".to_string();
        assert!(candidate.validate().is_err());
        candidate.artifacts[0].source =
            "https://example.com/model/resolve/0123456789abcdef/model.gguf".to_string();
        candidate.artifacts[0].path = "../escape".to_string();
        assert!(candidate.validate().is_err());
    }
}
