use std::path::PathBuf;

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::VisionError;

#[derive(Debug, Clone)]
pub(super) struct UpscaleConfig {
    pub model_path: PathBuf,
    pub use_gpu: bool,
    pub gpu_device_id: i32,
    pub scale_factor: u32,
    pub tile_size: u32,
    pub tile_pad: u32,
}

pub(super) struct ModelSession {
    session: Session,
    config: UpscaleConfig,
}

impl ModelSession {
    pub fn config(&self) -> &UpscaleConfig {
        &self.config
    }

    pub fn run(&mut self, input: Array4<f32>) -> Result<Array4<f32>, VisionError> {
        let input_tensor = Tensor::from_array(input.view())
            .map_err(|e| VisionError::Inference(format!("failed to construct input tensor: {e}")))?;

        let inputs = ort::inputs![input_tensor]
            .map_err(|e| VisionError::Inference(format!("failed to create inputs: {e}")))?;

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| VisionError::Inference(format!("ort session.run failed: {e}")))?;

        let output = outputs
            .iter()
            .next()
            .map(|(_, value)| value)
            .ok_or_else(|| VisionError::Inference("model produced no output tensor".to_string()))?;

        let output_view = output.try_extract_tensor::<f32>().map_err(|e| {
            VisionError::Inference(format!("failed to extract output tensor as f32: {e}"))
        })?;

        let shape = output_view.shape();
        if shape.len() != 4 {
            return Err(VisionError::Inference(format!(
                "expected 4D output tensor [B, C, H, W], got {:?}",
                shape
            )));
        }

        let array = output_view.to_owned().into_dimensionality::<ndarray::Ix4>().map_err(|e| {
            VisionError::Inference(format!("failed to reshape output to 4D: {e}"))
        })?;

        Ok(array)
    }
}

pub(super) fn create_session(config: &UpscaleConfig) -> Result<ModelSession, VisionError> {
    if !config.model_path.exists() {
        return Err(VisionError::ModelMissing {
            path: config.model_path.display().to_string(),
            message: "ONNX model file not found".to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|e| VisionError::Inference(format!("failed to create session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VisionError::Inference(format!("failed to set opt level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| VisionError::Inference(format!("failed to set thread count: {e}")))?;

    if config.use_gpu {
        let cuda_ep = CUDAExecutionProvider::default()
            .with_device_id(config.gpu_device_id);
        builder = builder
            .with_execution_providers([cuda_ep.build()])
            .map_err(|e| VisionError::Inference(format!("failed to register CUDA EP: {e}")))?;
    } else {
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| VisionError::Inference(format!("failed to register CPU EP: {e}")))?;
    }

    let session = builder
        .commit_from_file(&config.model_path)
        .map_err(|e| VisionError::Inference(format!("failed to load ONNX model: {e}")))?;

    Ok(ModelSession {
        session,
        config: config.clone(),
    })
}
