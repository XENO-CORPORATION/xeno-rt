use serde::Deserialize;
use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use xrt_core::{Result, XrtError};

const HUGGING_FACE_API_BASE: &str = "https://huggingface.co/api/models";
const HUGGING_FACE_BASE: &str = "https://huggingface.co";
const DOWNLOAD_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
}

impl DownloadProgress {
    pub fn percent(self) -> Option<f32> {
        (self.total > 0).then_some((self.downloaded as f32 / self.total as f32) * 100.0)
    }
}

#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub repo_id: String,
    pub filename: String,
    pub path: PathBuf,
    pub size: u64,
    pub was_cached: bool,
}

#[derive(Debug, Clone)]
pub struct CachedModel {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub size: u64,
}

pub struct ModelHub {
    cache_dir: PathBuf,
    agent: ureq::Agent,
    auth_token: Option<String>,
}

impl ModelHub {
    pub fn new() -> Result<Self> {
        Self::with_cache_dir(default_cache_dir()?)
    }

    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;
        let agent = ureq::AgentBuilder::new()
            .user_agent("xrt-hub/0.1")
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(300))
            .redirects(5)
            .build();

        Ok(Self {
            cache_dir,
            agent,
            auth_token: auth_token_from_env(),
        })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn cached_path(&self, repo_id: &str, filename: &str) -> Result<PathBuf> {
        Ok(self.cache_dir.join(relative_cache_path(repo_id, filename)?))
    }

    pub fn list_cached(&self) -> Result<Vec<CachedModel>> {
        let mut entries = Vec::new();
        collect_cached_models(&self.cache_dir, &self.cache_dir, &mut entries)?;
        entries.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));
        Ok(entries)
    }

    pub fn clean_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
        }
        fs::create_dir_all(&self.cache_dir)?;
        Ok(())
    }

    pub fn resolve_gguf_by_quantization(
        &self,
        repo_id: &str,
        quantization: &str,
    ) -> Result<String> {
        validate_path_like(repo_id, "repo id")?;
        let needle = quantization.trim().to_ascii_lowercase();
        if needle.is_empty() {
            return Err(XrtError::Runtime(
                "quantization selector must not be empty".to_string(),
            ));
        }

        let matches = self
            .list_repo_files(repo_id)?
            .into_iter()
            .filter(|filename| filename.to_ascii_lowercase().ends_with(".gguf"))
            .filter(|filename| filename.to_ascii_lowercase().contains(&needle))
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [] => Err(XrtError::Runtime(format!(
                "no GGUF files in {repo_id} matched quantization {quantization}"
            ))),
            [single] => Ok(single.clone()),
            _ => Err(XrtError::Runtime(format!(
                "quantization {quantization} matched multiple GGUF files in {repo_id}: {}",
                matches.join(", ")
            ))),
        }
    }

    pub fn download(&self, repo_id: &str, filename: &str) -> Result<DownloadedModel> {
        self.download_with_progress(repo_id, filename, |_| {})
    }

    pub fn download_with_progress<F>(
        &self,
        repo_id: &str,
        filename: &str,
        mut on_progress: F,
    ) -> Result<DownloadedModel>
    where
        F: FnMut(DownloadProgress),
    {
        let expected_size = self.fetch_expected_size(repo_id, filename)?;
        let destination = self.cached_path(repo_id, filename)?;
        if let Ok(metadata) = fs::metadata(&destination) {
            if metadata.len() == expected_size {
                return Ok(DownloadedModel {
                    repo_id: repo_id.to_string(),
                    filename: filename.to_string(),
                    path: destination,
                    size: expected_size,
                    was_cached: true,
                });
            }
            let _ = fs::remove_file(&destination);
        }

        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let temp_path = partial_download_path(&destination)?;
        if temp_path.exists() {
            let _ = fs::remove_file(&temp_path);
        }

        let url = download_url(repo_id, filename);
        let response = self.call(self.authorized(self.agent.get(&url)))?;
        let mut reader = response.into_reader();
        let mut file = File::create(&temp_path)?;
        let mut buffer = vec![0u8; DOWNLOAD_BUFFER_SIZE];
        let mut downloaded = 0u64;
        let mut last_update = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        on_progress(DownloadProgress {
            downloaded: 0,
            total: expected_size,
        });

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            downloaded += read as u64;
            if last_update.elapsed() >= Duration::from_millis(250) || downloaded == expected_size {
                on_progress(DownloadProgress {
                    downloaded,
                    total: expected_size,
                });
                last_update = Instant::now();
            }
        }

        file.flush()?;
        if downloaded != expected_size {
            let _ = fs::remove_file(&temp_path);
            return Err(XrtError::Runtime(format!(
                "downloaded size mismatch for {repo_id}/{filename}: expected {expected_size} bytes, got {downloaded}"
            )));
        }

        fs::rename(&temp_path, &destination)?;
        Ok(DownloadedModel {
            repo_id: repo_id.to_string(),
            filename: filename.to_string(),
            path: destination,
            size: expected_size,
            was_cached: false,
        })
    }

    pub fn download_by_quantization<F>(
        &self,
        repo_id: &str,
        quantization: &str,
        on_progress: F,
    ) -> Result<DownloadedModel>
    where
        F: FnMut(DownloadProgress),
    {
        let filename = self.resolve_gguf_by_quantization(repo_id, quantization)?;
        self.download_with_progress(repo_id, &filename, on_progress)
    }

    fn fetch_expected_size(&self, repo_id: &str, filename: &str) -> Result<u64> {
        validate_path_like(repo_id, "repo id")?;
        validate_gguf_filename(filename)?;
        let url = download_url(repo_id, filename);

        match self.call(self.authorized(self.agent.head(&url))) {
            Ok(response) => parse_expected_size(&response, repo_id, filename),
            Err(_) => {
                let response = self.call(self.authorized(self.agent.get(&url)))?;
                parse_expected_size(&response, repo_id, filename)
            }
        }
    }

    fn list_repo_files(&self, repo_id: &str) -> Result<Vec<String>> {
        let url = model_info_url(repo_id);
        let response = self.call(self.authorized(self.agent.get(&url)))?;
        let payload: HuggingFaceModelInfo = response.into_json().map_err(|error| {
            XrtError::Runtime(format!(
                "failed to parse Hugging Face model listing for {repo_id}: {error}"
            ))
        })?;
        Ok(payload
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .collect())
    }

    fn authorized(&self, request: ureq::Request) -> ureq::Request {
        if let Some(token) = &self.auth_token {
            request.set("Authorization", &format!("Bearer {token}"))
        } else {
            request
        }
    }

    fn call(&self, request: ureq::Request) -> Result<ureq::Response> {
        request.call().map_err(map_ureq_error)
    }
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelInfo {
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

fn parse_expected_size(response: &ureq::Response, repo_id: &str, filename: &str) -> Result<u64> {
    let value = response
        .header("X-Linked-Size")
        .or_else(|| response.header("Content-Length"))
        .ok_or_else(|| {
            XrtError::Runtime(format!(
                "Hugging Face did not return a content length for {repo_id}/{filename}"
            ))
        })?;

    value.parse::<u64>().map_err(|error| {
        XrtError::Runtime(format!(
            "invalid content length for {repo_id}/{filename}: {value} ({error})"
        ))
    })
}

fn collect_cached_models(
    root: &Path,
    current: &Path,
    entries: &mut Vec<CachedModel>,
) -> Result<()> {
    if !current.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_cached_models(root, &path, entries)?;
            continue;
        }

        let metadata = entry.metadata()?;
        let relative_path = path
            .strip_prefix(root)
            .map(Path::to_path_buf)
            .map_err(|error| {
                XrtError::Runtime(format!(
                    "failed to compute cache-relative path for {}: {error}",
                    path.display()
                ))
            })?;
        entries.push(CachedModel {
            path,
            relative_path,
            size: metadata.len(),
        });
    }

    Ok(())
}

fn default_cache_dir() -> Result<PathBuf> {
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
            XrtError::Runtime(
                "could not determine the user's home directory for the model cache".to_string(),
            )
        })?;
    Ok(home.join(".cache").join("xrt").join("models"))
}

fn auth_token_from_env() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn relative_cache_path(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let mut path = PathBuf::new();
    for segment in split_path_like(repo_id, "repo id")? {
        path.push(segment);
    }
    for segment in split_path_like(filename, "filename")? {
        path.push(segment);
    }
    validate_gguf_path(&path)?;
    Ok(path)
}

fn validate_path_like(value: &str, label: &str) -> Result<()> {
    let _ = split_path_like(value, label)?;
    Ok(())
}

fn validate_gguf_filename(filename: &str) -> Result<()> {
    let path = PathBuf::from(filename.replace('/', std::path::MAIN_SEPARATOR_STR));
    validate_gguf_path(&path)?;
    Ok(())
}

fn validate_gguf_path(path: &Path) -> Result<()> {
    let is_gguf = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !is_gguf {
        return Err(XrtError::Runtime(format!(
            "expected a .gguf filename, got {}",
            path.display()
        )));
    }
    Ok(())
}

fn split_path_like<'a>(value: &'a str, label: &str) -> Result<Vec<&'a str>> {
    let segments = value.split('/').collect::<Vec<_>>();
    if segments.is_empty() || segments.iter().any(|segment| segment.is_empty()) {
        return Err(XrtError::Runtime(format!("{label} must not be empty")));
    }
    for segment in &segments {
        if matches!(*segment, "." | "..")
            || segment.contains('\\')
            || segment.contains(':')
            || segment.contains('\0')
        {
            return Err(XrtError::Runtime(format!(
                "{label} contains an invalid path segment: {segment}"
            )));
        }
    }
    Ok(segments)
}

fn partial_download_path(destination: &Path) -> Result<PathBuf> {
    let file_name = destination.file_name().ok_or_else(|| {
        XrtError::Runtime(format!(
            "cannot create a partial download path for {}",
            destination.display()
        ))
    })?;
    Ok(destination.with_file_name(format!("{}.part", file_name.to_string_lossy())))
}

fn model_info_url(repo_id: &str) -> String {
    format!("{HUGGING_FACE_API_BASE}/{}", encode_path(repo_id))
}

fn download_url(repo_id: &str, filename: &str) -> String {
    format!(
        "{HUGGING_FACE_BASE}/{}/resolve/main/{}?download=true",
        encode_path(repo_id),
        encode_path(filename)
    )
}

fn encode_path(path: &str) -> String {
    split_path_like(path, "path")
        .unwrap_or_default()
        .into_iter()
        .map(encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_segment(segment: &str) -> String {
    let mut output = String::new();
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn map_ureq_error(error: ureq::Error) -> XrtError {
    match error {
        ureq::Error::Status(status, response) => {
            let status_text = response.status_text().to_string();
            let body = response
                .into_string()
                .ok()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty());
            let detail = body.unwrap_or(status_text.clone());
            XrtError::Runtime(format!(
                "Hugging Face request failed with HTTP {status} {status_text}: {detail}"
            ))
        }
        ureq::Error::Transport(transport) => {
            XrtError::Io(std::io::Error::other(transport.to_string()))
        }
    }
}
