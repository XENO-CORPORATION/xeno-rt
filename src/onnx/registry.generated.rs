use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{OnnxError, Result};

const MODEL_BASE_URL: &str = "https://updates.xenostudio.ai/models";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Upscale,
    BackgroundRemoval,
    FaceRestore,
    Colorize,
    Inpaint,
    FaceDetect,
    DepthEstimation,
    StyleTransfer,
    Ocr,
    PoseEstimation,
    FaceAnalysis,
    FrameInterpolation,
    Transcription,
    AudioSeparation,
    NoiseReduction,
}

impl TaskType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upscale => "upscale",
            Self::BackgroundRemoval => "background-removal",
            Self::FaceRestore => "face-restore",
            Self::Colorize => "colorize",
            Self::Inpaint => "inpaint",
            Self::FaceDetect => "face-detect",
            Self::DepthEstimation => "depth-estimation",
            Self::StyleTransfer => "style-transfer",
            Self::Ocr => "ocr",
            Self::PoseEstimation => "pose-estimation",
            Self::FaceAnalysis => "face-analysis",
            Self::FrameInterpolation => "frame-interpolation",
            Self::Transcription => "transcription",
            Self::AudioSeparation => "audio-separation",
            Self::NoiseReduction => "noise-reduction",
        }
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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
        PathBuf::from(&self.task)
            .join(&self.id)
            .join(&self.version)
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub task: TaskType,
    pub filename: String,
    pub size_mb: u64,
    pub url: String,
}

impl ModelEntry {
    pub fn new(
        name: impl Into<String>,
        task: TaskType,
        filename: impl Into<String>,
        size_mb: u64,
        url: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            task,
            filename: filename.into(),
            size_mb,
            url: url.into(),
        }
    }

    pub fn artifact(&self) -> ModelArtifact {
        ModelArtifact {
            file_name: self.filename.clone(),
            source_url: self.url.clone(),
            sha256: None,
            size_bytes: Some(self.size_mb * 1_000_000),
        }
    }

    pub fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.name.clone(),
            task: self.task.as_str().to_string(),
            family: self.name.clone(),
            version: "xeno-lib-default".to_string(),
            artifacts: vec![self.artifact()],
            estimated_ram_mb: Some(self.size_mb),
            estimated_vram_mb: None,
            execution_providers: vec![
                ExecutionProviderKind::Cpu,
                ExecutionProviderKind::Cuda,
            ],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ModelRegistry {
    models: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, entry: ModelEntry) -> Option<ModelEntry> {
        self.models.insert(entry.name.clone(), entry)
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    pub fn list(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    pub fn list_by_task(&self, task: TaskType) -> Vec<&ModelEntry> {
        self.models
            .values()
            .filter(|entry| entry.task == task)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn default_models() -> Self {
        let mut registry = Self {
            models: BTreeMap::new(),
        };

        for spec in DEFAULT_MODELS {
            registry.register(spec.into_model_entry());
        }

        registry
    }
}

#[derive(Debug, Clone, Copy)]
struct DefaultModelSpec {
    name: &'static str,
    task: TaskType,
    filename: &'static str,
    size_mb: u64,
}

impl DefaultModelSpec {
    fn into_model_entry(self) -> ModelEntry {
        ModelEntry::new(
            self.name,
            self.task,
            self.filename,
            self.size_mb,
            format!("{MODEL_BASE_URL}/{}", self.filename),
        )
    }
}

const DEFAULT_MODELS: [DefaultModelSpec; 19] = [
    DefaultModelSpec {
        name: "real-esrgan-x4",
        task: TaskType::Upscale,
        filename: "realesrgan_x4plus.onnx",
        size_mb: 67,
    },
    DefaultModelSpec {
        name: "birefnet",
        task: TaskType::BackgroundRemoval,
        filename: "birefnet-general.onnx",
        size_mb: 112,
    },
    DefaultModelSpec {
        name: "gfpgan",
        task: TaskType::FaceRestore,
        filename: "gfpgan.onnx",
        size_mb: 87,
    },
    DefaultModelSpec {
        name: "codeformer",
        task: TaskType::FaceRestore,
        filename: "codeformer.onnx",
        size_mb: 75,
    },
    DefaultModelSpec {
        name: "ddcolor",
        task: TaskType::Colorize,
        filename: "ddcolor.onnx",
        size_mb: 82,
    },
    DefaultModelSpec {
        name: "lama",
        task: TaskType::Inpaint,
        filename: "lama.onnx",
        size_mb: 102,
    },
    DefaultModelSpec {
        name: "scrfd",
        task: TaskType::FaceDetect,
        filename: "scrfd_10g.onnx",
        size_mb: 16,
    },
    DefaultModelSpec {
        name: "midas-large",
        task: TaskType::DepthEstimation,
        filename: "midas_v31_large.onnx",
        size_mb: 105,
    },
    DefaultModelSpec {
        name: "style-mosaic",
        task: TaskType::StyleTransfer,
        filename: "style_mosaic.onnx",
        size_mb: 7,
    },
    DefaultModelSpec {
        name: "paddle-ocr-det",
        task: TaskType::Ocr,
        filename: "paddle_det.onnx",
        size_mb: 4,
    },
    DefaultModelSpec {
        name: "paddle-ocr-rec",
        task: TaskType::Ocr,
        filename: "paddle_rec.onnx",
        size_mb: 5,
    },
    DefaultModelSpec {
        name: "movenet-lightning",
        task: TaskType::PoseEstimation,
        filename: "movenet_lightning.onnx",
        size_mb: 9,
    },
    DefaultModelSpec {
        name: "age-estimation",
        task: TaskType::FaceAnalysis,
        filename: "age_estimation.onnx",
        size_mb: 5,
    },
    DefaultModelSpec {
        name: "gender-classification",
        task: TaskType::FaceAnalysis,
        filename: "gender_classification.onnx",
        size_mb: 5,
    },
    DefaultModelSpec {
        name: "emotion-recognition",
        task: TaskType::FaceAnalysis,
        filename: "emotion_recognition.onnx",
        size_mb: 5,
    },
    DefaultModelSpec {
        name: "rife-v4",
        task: TaskType::FrameInterpolation,
        filename: "rife-v4.6.onnx",
        size_mb: 30,
    },
    DefaultModelSpec {
        name: "whisper-base",
        task: TaskType::Transcription,
        filename: "whisper-base.onnx",
        size_mb: 74,
    },
    DefaultModelSpec {
        name: "demucs-hybrid",
        task: TaskType::AudioSeparation,
        filename: "demucs_hybrid.onnx",
        size_mb: 83,
    },
    DefaultModelSpec {
        name: "rnnoise",
        task: TaskType::NoiseReduction,
        filename: "rnnoise.onnx",
        size_mb: 2,
    },
];

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

#[cfg(test)]
mod tests {
    use super::{ModelRegistry, TaskType};

    #[test]
    fn default_models_populate_expected_inventory() {
        let registry = ModelRegistry::default_models();

        assert_eq!(registry.len(), 19);
        assert_eq!(registry.list().len(), 19);
        assert!(!registry.is_empty());
    }

    #[test]
    fn bundled_tasks_return_all_required_entries() {
        let registry = ModelRegistry::default_models();

        assert_eq!(registry.list_by_task(TaskType::FaceRestore).len(), 2);
        assert_eq!(registry.list_by_task(TaskType::Ocr).len(), 2);
        assert_eq!(registry.list_by_task(TaskType::FaceAnalysis).len(), 3);
    }

    #[test]
    fn lookups_return_expected_metadata() {
        let registry = ModelRegistry::default_models();
        let upscale = registry
            .get("real-esrgan-x4")
            .expect("upscale model should exist");
        let depth = registry
            .get("midas-large")
            .expect("depth model should exist");

        assert_eq!(upscale.task, TaskType::Upscale);
        assert_eq!(upscale.filename, "realesrgan_x4plus.onnx");
        assert_eq!(
            upscale.url,
            "https://updates.xenostudio.ai/models/realesrgan_x4plus.onnx"
        );

        assert_eq!(depth.task, TaskType::DepthEstimation);
        assert_eq!(depth.filename, "midas_v31_large.onnx");
    }
}
