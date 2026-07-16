use std::collections::HashMap;
use std::fmt;

const MODEL_BASE_URL: &str = "https://updates.xenostudio.ai/models";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn register(&mut self, entry: ModelEntry) -> Option<ModelEntry> {
        self.models.insert(entry.name.clone(), entry)
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    pub fn list(&self) -> Vec<&ModelEntry> {
        let mut entries = self.models.values().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        entries
    }

    pub fn list_by_task(&self, task: TaskType) -> Vec<&ModelEntry> {
        let mut entries = self
            .models
            .values()
            .filter(|entry| entry.task == task)
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        entries
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn default_models() -> Self {
        let mut registry = Self {
            models: HashMap::with_capacity(DEFAULT_MODELS.len()),
        };

        // This mirrors xeno-lib's current default task surface: one default model for
        // single-model tasks, both primary face-restoration choices, the OCR pair,
        // and the face-analysis trio.
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
        let upscale = registry.get("real-esrgan-x4").expect("upscale model should exist");
        let depth = registry.get("midas-large").expect("depth model should exist");

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
