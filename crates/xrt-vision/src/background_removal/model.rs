use std::path::PathBuf;

use ndarray::{Array2, Array4};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::VisionError;

#[derive(Debug, Clone)]
pub(super) struct BackgroundRemovalConfig {
    pub model_path: PathBuf,
    pub use_gpu: bool,
    pub gpu_device_id: i32,
    pub confidence_threshold: f32,
}

pub(super) struct ModelSession {
    session: Session,
    config: BackgroundRemovalConfig,
}

impl ModelSession {
    pub fn config(&self) -> &BackgroundRemovalConfig {
        &self.config
    }

    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array2<f32>, VisionError> {
        let tensor = Tensor::from_array(input.view())
            .map_err(|err| VisionError::Inference(format!("failed to create tensor: {err}")))?;
        let inputs = ort::inputs![tensor]
            .map_err(|err| VisionError::Inference(format!("failed to create inputs: {err}")))?;
        let outputs = self
            .session
            .run(inputs)
            .map_err(|err| VisionError::Inference(format!("ONNX inference failed: {err}")))?;
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| VisionError::Inference("model returned no output tensor".to_string()))?;
        let (shape, data) = output
            .1
            .try_extract_raw_tensor::<f32>()
            .map_err(|err| VisionError::Inference(format!("invalid output tensor: {err}")))?;
        let dims = shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let (height, width) = match dims.as_slice() {
            [_, _, height, width] | [_, height, width] | [height, width]
                if *height > 0 && *width > 0 =>
            {
                (*height, *width)
            }
            _ => {
                return Err(VisionError::Inference(format!(
                    "unsupported output tensor shape {dims:?}"
                )))
            }
        };
        Array2::from_shape_vec((height, width), data.to_vec()).map_err(|err| {
            VisionError::Inference(format!("failed to reshape output tensor: {err}"))
        })
    }
}

pub(super) fn load_model(config: &BackgroundRemovalConfig) -> Result<ModelSession, VisionError> {
    if !config.model_path.is_file() {
        return Err(VisionError::ModelMissing {
            path: config.model_path.display().to_string(),
            message: "BiRefNet ONNX file is not a regular file".to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|err| VisionError::Inference(format!("failed to create ONNX session: {err}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| {
            VisionError::Inference(format!("failed to configure ONNX optimization: {err}"))
        })?
        .with_memory_pattern(true)
        .map_err(|err| {
            VisionError::Inference(format!("failed to configure ONNX memory pattern: {err}"))
        })?
        .with_intra_threads(4)
        .map_err(|err| {
            VisionError::Inference(format!("failed to configure ONNX threads: {err}"))
        })?;

    builder = if config.use_gpu && cfg!(feature = "cuda") {
        builder.with_execution_providers([
            CUDAExecutionProvider::default()
                .with_device_id(config.gpu_device_id)
                .build(),
            CPUExecutionProvider::default().build(),
        ])
    } else {
        builder.with_execution_providers([CPUExecutionProvider::default().build()])
    }
    .map_err(|err| VisionError::Inference(format!("failed to configure ONNX providers: {err}")))?;

    let session = builder
        .commit_from_file(&config.model_path)
        .map_err(|err| {
            VisionError::Inference(format!(
                "failed to load ONNX model `{}`: {err}",
                config.model_path.display()
            ))
        })?;
    Ok(ModelSession {
        session,
        config: config.clone(),
    })
}
