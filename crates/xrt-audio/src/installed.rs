//! Resolve model bundles from the shared runtime cache. No network is used
//! during inference resolution; installation is a separate explicit operation.
use crate::AudioError;
use std::path::{Path, PathBuf};

pub fn model_dir(id: &str) -> Result<PathBuf, AudioError> {
    let hub = xrt_hub::ModelHub::new().map_err(|e| AudioError::Inference(e.to_string()))?;
    hub.resolve_installed_bundle(id, None)
        .map_err(|e| AudioError::ModelMissing {
            path: id.into(),
            message: format!("install the pinned bundle with `xrt bundle install` first: {e}"),
        })
}

/// Verify a managed bundle before constructing ONNX sessions. Explicit legacy
/// developer directories remain supported; they do not claim managed integrity.
pub fn verify_managed(dir: &Path) -> Result<(), AudioError> {
    let path = dir.join("xrt.bundle.json");
    if !path.exists() {
        return Ok(());
    }
    let bytes = std::fs::read(&path).map_err(|e| AudioError::Inference(e.to_string()))?;
    let manifest = xrt_hub::ArtifactManifest::parse(&bytes)
        .map_err(|e| AudioError::Inference(e.to_string()))?;
    let digest = manifest
        .digest()
        .map_err(|e| AudioError::Inference(e.to_string()))?;
    if dir.file_name().and_then(|v| v.to_str()) != Some(digest.as_str()) {
        return Err(AudioError::Inference(
            "managed model directory does not match manifest digest".into(),
        ));
    }
    manifest
        .verify_directory(dir)
        .map_err(|e| AudioError::Inference(e.to_string()))
}
