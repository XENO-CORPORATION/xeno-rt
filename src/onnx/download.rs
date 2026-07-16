use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

use super::{
    registry::{ModelArtifact, ModelDescriptor},
    OnnxError, Result,
};

const DOWNLOAD_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl DownloadProgress {
    pub fn percent(self) -> Option<f32> {
        self.total.and_then(|total| {
            (total > 0).then_some((self.downloaded as f32 / total as f32) * 100.0)
        })
    }
}

#[derive(Debug, Clone)]
pub struct DownloadedArtifact {
    pub model_id: String,
    pub file_name: String,
    pub path: PathBuf,
    pub size: u64,
    pub was_cached: bool,
}

pub struct OnnxDownloader {
    cache_dir: PathBuf,
    agent: ureq::Agent,
}

impl OnnxDownloader {
    pub fn new() -> Result<Self> {
        Self::with_cache_dir(default_cache_dir()?)
    }

    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;

        let agent = ureq::AgentBuilder::new()
            .user_agent("xrt-onnx/0.1")
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(300))
            .redirects(5)
            .build();

        Ok(Self { cache_dir, agent })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn download_model(&self, model: &ModelDescriptor) -> Result<Vec<DownloadedArtifact>> {
        self.download_model_with_progress(model, |_artifact, _progress| {})
    }

    pub fn download_model_with_progress<F>(
        &self,
        model: &ModelDescriptor,
        mut on_progress: F,
    ) -> Result<Vec<DownloadedArtifact>>
    where
        F: FnMut(&ModelArtifact, DownloadProgress),
    {
        model.validate()?;

        let mut downloaded = Vec::with_capacity(model.artifacts.len());
        for artifact in &model.artifacts {
            downloaded.push(
                self.download_artifact_with_progress(model, artifact, |progress| {
                    on_progress(artifact, progress)
                })?,
            );
        }

        Ok(downloaded)
    }

    pub fn artifact_path(
        &self,
        model: &ModelDescriptor,
        artifact: &ModelArtifact,
    ) -> Result<PathBuf> {
        model.validate()?;
        artifact.validate()?;
        Ok(self
            .cache_dir
            .join(model.bundle_dir())
            .join(&artifact.file_name))
    }

    pub fn is_cached(&self, model: &ModelDescriptor, artifact: &ModelArtifact) -> Result<bool> {
        let path = self.artifact_path(model, artifact)?;
        validate_cached_file(&path, artifact)
    }

    fn download_artifact_with_progress<F>(
        &self,
        model: &ModelDescriptor,
        artifact: &ModelArtifact,
        mut on_progress: F,
    ) -> Result<DownloadedArtifact>
    where
        F: FnMut(DownloadProgress),
    {
        let destination = self.artifact_path(model, artifact)?;
        if validate_cached_file(&destination, artifact)? {
            let size = fs::metadata(&destination)?.len();
            on_progress(DownloadProgress {
                downloaded: size,
                total: Some(size),
            });
            return Ok(DownloadedArtifact {
                model_id: model.id.clone(),
                file_name: artifact.file_name.clone(),
                path: destination,
                size,
                was_cached: true,
            });
        }

        if destination.exists() {
            let _ = fs::remove_file(&destination);
        }

        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }

        let temp_path = partial_download_path(&destination)?;
        if temp_path.exists() {
            let _ = fs::remove_file(&temp_path);
        }

        let response = self
            .agent
            .get(&artifact.source_url)
            .call()
            .map_err(map_ureq_error)?;
        let total = artifact
            .size_bytes
            .or_else(|| parse_content_length(&response));

        let mut reader = response.into_reader();
        let mut file = File::create(&temp_path)?;
        let mut buffer = vec![0u8; DOWNLOAD_BUFFER_SIZE];
        let mut downloaded = 0u64;
        let mut last_update = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        on_progress(DownloadProgress {
            downloaded: 0,
            total,
        });

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            downloaded += read as u64;

            if last_update.elapsed() >= Duration::from_millis(250)
                || total.is_some_and(|expected| downloaded == expected)
            {
                on_progress(DownloadProgress { downloaded, total });
                last_update = Instant::now();
            }
        }

        file.flush()?;
        verify_downloaded_file(&temp_path, artifact, downloaded, total)?;
        fs::rename(&temp_path, &destination)?;

        Ok(DownloadedArtifact {
            model_id: model.id.clone(),
            file_name: artifact.file_name.clone(),
            path: destination,
            size: downloaded,
            was_cached: false,
        })
    }
}

pub fn default_cache_dir() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .or_else(
            || match (env::var_os("HOMEDRIVE"), env::var_os("HOMEPATH")) {
                (Some(drive), Some(path)) => {
                    Some(format!("{}{}", drive.to_string_lossy(), path.to_string_lossy()).into())
                }
                _ => None,
            },
        )
        .map(PathBuf::from)
        .ok_or_else(|| {
            OnnxError::InvalidMetadata(
                "could not determine the user's home directory for the ONNX cache".to_string(),
            )
        })?;

    Ok(home.join(".cache").join("xrt").join("models").join("onnx"))
}

fn validate_cached_file(path: &Path, artifact: &ModelArtifact) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let size = fs::metadata(path)?.len();
    if let Some(expected_size) = artifact.size_bytes {
        if size != expected_size {
            return Ok(false);
        }
    }

    if let Some(expected_sha256) = artifact.sha256.as_deref() {
        let actual_sha256 = sha256_file(path)?;
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            return Ok(false);
        }
    }

    Ok(true)
}

fn verify_downloaded_file(
    path: &Path,
    artifact: &ModelArtifact,
    actual_size: u64,
    fallback_size: Option<u64>,
) -> Result<()> {
    if let Some(expected_size) = artifact.size_bytes.or(fallback_size) {
        if actual_size != expected_size {
            let _ = fs::remove_file(path);
            return Err(OnnxError::SizeMismatch {
                path: path.to_path_buf(),
                expected: expected_size,
                actual: actual_size,
            });
        }
    }

    if let Some(expected_sha256) = artifact.sha256.as_deref() {
        let actual_sha256 = sha256_file(path)?;
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            let _ = fs::remove_file(path);
            return Err(OnnxError::ChecksumMismatch {
                path: path.to_path_buf(),
                expected: expected_sha256.to_string(),
                actual: actual_sha256,
            });
        }
    }

    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; DOWNLOAD_BUFFER_SIZE];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn parse_content_length(response: &ureq::Response) -> Option<u64> {
    response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok())
}

fn partial_download_path(destination: &Path) -> Result<PathBuf> {
    let file_name = destination.file_name().ok_or_else(|| {
        OnnxError::InvalidMetadata(format!(
            "cannot create a partial download path for {}",
            destination.display()
        ))
    })?;

    Ok(destination.with_file_name(format!("{}.part", file_name.to_string_lossy())))
}

fn map_ureq_error(error: ureq::Error) -> OnnxError {
    match error {
        ureq::Error::Status(status, response) => {
            let status_text = response.status_text().to_string();
            let body = response
                .into_string()
                .ok()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty());
            let detail = body.unwrap_or_else(|| status_text.clone());
            OnnxError::Download(format!("HTTP {status} {status_text}: {detail}"))
        }
        ureq::Error::Transport(transport) => OnnxError::Download(transport.to_string()),
    }
}
