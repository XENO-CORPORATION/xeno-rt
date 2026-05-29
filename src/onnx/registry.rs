use std::{
    collections::{btree_map::Values, BTreeMap},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{OnnxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub file_name: String,
    pub source_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

impl ModelArtifact {
    pub fn validate(&self) -> Result<()> {
        validate_segment(&self.file_name, "artifact file name")?;
        if self.source_url.trim().is_empty() {
            return Err(OnnxError::InvalidMetadata(
                "artifact source_url must not be empty".to_string(),
            ));
        }
        if let Some(sha256) = &self.sha256 {
            if !is_valid_sha256(sha256) {
                return Err(OnnxError::InvalidMetadata(format!(
                    "artifact {} has an invalid sha256 digest",
                    self.file_name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub task: String,
    pub family: String,
    pub version: String,
    pub artifacts: Vec<ModelArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_ram_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_vram_mb: Option<u64>,
    #[serde(default)]
    pub execution_providers: Vec<ExecutionProviderKind>,
}

impl ModelDescriptor {
    pub fn validate(&self) -> Result<()> {
        validate_segment(&self.id, "model id")?;
        validate_segment(&self.task, "task")?;
        validate_segment(&self.family, "family")?;
        validate_segment(&self.version, "version")?;

        if self.artifacts.is_empty() {
            return Err(OnnxError::InvalidMetadata(format!(
                "model {} must include at least one artifact",
                self.id
            )));
        }

        for artifact in &self.artifacts {
            artifact.validate()?;
        }

        Ok(())
    }

    pub fn bundle_dir(&self) -> PathBuf {
        PathBuf::from(&self.task).join(&self.id).join(&self.version)
    }

    pub fn primary_artifact(&self) -> Option<&ModelArtifact> {
        self.artifacts.first()
    }

    pub fn supports_provider(&self, provider: ExecutionProviderKind) -> bool {
        if self.execution_providers.is_empty() {
            return provider == ExecutionProviderKind::Cpu;
        }
        self.execution_providers.contains(&provider)
    }

    pub fn artifact_cache_path(&self, cache_root: &Path, file_name: &str) -> Result<PathBuf> {
        let artifact = self
            .artifacts
            .iter()
            .find(|artifact| artifact.file_name == file_name)
            .ok_or_else(|| {
                OnnxError::InvalidMetadata(format!(
                    "model {} does not contain artifact {}",
                    self.id, file_name
                ))
            })?;
        Ok(cache_root.join(self.bundle_dir()).join(&artifact.file_name))
    }
}

#[derive(Debug, Default, Clone)]
pub struct ModelRegistry {
    models: BTreeMap<String, ModelDescriptor>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn contains(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    pub fn register(&mut self, model: ModelDescriptor) -> Result<()> {
        model.validate()?;
        if self.models.contains_key(&model.id) {
            return Err(OnnxError::DuplicateModel(model.id));
        }
        self.models.insert(model.id.clone(), model);
        Ok(())
    }

    pub fn upsert(&mut self, model: ModelDescriptor) -> Result<Option<ModelDescriptor>> {
        model.validate()?;
        Ok(self.models.insert(model.id.clone(), model))
    }

    pub fn get(&self, model_id: &str) -> Option<&ModelDescriptor> {
        self.models.get(model_id)
    }

    pub fn remove(&mut self, model_id: &str) -> Option<ModelDescriptor> {
        self.models.remove(model_id)
    }

    pub fn list(&self) -> Values<'_, String, ModelDescriptor> {
        self.models.values()
    }

    pub fn list_by_task<'a>(
        &'a self,
        task: &'a str,
    ) -> impl Iterator<Item = &'a ModelDescriptor> + 'a {
        self.models.values().filter(move |model| model.task == task)
    }

    pub fn artifact_path(
        &self,
        cache_root: &Path,
        model_id: &str,
        file_name: &str,
    ) -> Result<PathBuf> {
        let model = self
            .get(model_id)
            .ok_or_else(|| OnnxError::ModelNotFound(model_id.to_string()))?;
        model.artifact_cache_path(cache_root, file_name)
    }
}

fn validate_segment(value: &str, label: &str) -> Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(OnnxError::InvalidMetadata(format!(
            "{label} must not be empty"
        )));
    }

    if matches!(trimmed, "." | "..")
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains(':')
        || trimmed.contains('\0')
    {
        return Err(OnnxError::InvalidMetadata(format!(
            "{label} contains an invalid path segment: {trimmed}"
        )));
    }

    Ok(())
}

fn is_valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}
