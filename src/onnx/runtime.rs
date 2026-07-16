use std::path::{Path, PathBuf};

use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
};
use tracing::warn;

use super::{OnnxError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDevice {
    Cpu,
    Cuda { device_id: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreference {
    CpuOnly,
    PreferGpu,
    RequireGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxOptimizationLevel {
    Disable,
    Basic,
    Extended,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxRuntimeOptions {
    pub gpu_preference: GpuPreference,
    pub device_id: i32,
    pub gpu_memory_limit_bytes: Option<usize>,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub parallel_execution: bool,
    pub optimization_level: OnnxOptimizationLevel,
    pub cpu_arena: bool,
}

impl Default for OnnxRuntimeOptions {
    fn default() -> Self {
        Self {
            gpu_preference: GpuPreference::PreferGpu,
            device_id: 0,
            gpu_memory_limit_bytes: None,
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
            optimization_level: OnnxOptimizationLevel::All,
            cpu_arena: true,
        }
    }
}

pub struct OnnxRuntime {
    model_path: PathBuf,
    session: Session,
    execution_device: ExecutionDevice,
    options: OnnxRuntimeOptions,
}

impl OnnxRuntime {
    pub fn load<P>(model_path: P, options: OnnxRuntimeOptions) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        ensure_ort_environment()?;

        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(OnnxError::ModelPathMissing { path: model_path });
        }

        match options.gpu_preference {
            GpuPreference::CpuOnly => {
                let session = build_cpu_session(&model_path, options)?;
                Ok(Self {
                    model_path,
                    session,
                    execution_device: ExecutionDevice::Cpu,
                    options,
                })
            }
            GpuPreference::PreferGpu => match build_gpu_session(&model_path, options) {
                Ok(session) => Ok(Self {
                    model_path,
                    session,
                    execution_device: ExecutionDevice::Cuda {
                        device_id: options.device_id,
                    },
                    options,
                }),
                Err(error) => {
                    warn!(
                        path = %model_path.display(),
                        device_id = options.device_id,
                        error = %error,
                        "CUDA ONNX session initialization failed, falling back to CPU"
                    );
                    let session = build_cpu_session(&model_path, options)?;
                    Ok(Self {
                        model_path,
                        session,
                        execution_device: ExecutionDevice::Cpu,
                        options,
                    })
                }
            },
            GpuPreference::RequireGpu => {
                let session = build_gpu_session(&model_path, options)?;
                Ok(Self {
                    model_path,
                    session,
                    execution_device: ExecutionDevice::Cuda {
                        device_id: options.device_id,
                    },
                    options,
                })
            }
        }
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn execution_device(&self) -> ExecutionDevice {
        self.execution_device
    }

    pub fn options(&self) -> &OnnxRuntimeOptions {
        &self.options
    }

    pub fn using_gpu(&self) -> bool {
        matches!(self.execution_device, ExecutionDevice::Cuda { .. })
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

impl From<OnnxOptimizationLevel> for GraphOptimizationLevel {
    fn from(value: OnnxOptimizationLevel) -> Self {
        match value {
            OnnxOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            OnnxOptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OnnxOptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OnnxOptimizationLevel::All => GraphOptimizationLevel::Level3,
        }
    }
}

fn ensure_ort_environment() -> Result<()> {
    ort::init().with_name("xrt-onnx").commit()?;
    Ok(())
}

fn build_cpu_session(model_path: &Path, options: OnnxRuntimeOptions) -> Result<Session> {
    configure_session_builder(options)?
        .with_execution_providers([CPUExecutionProvider::default()
            .with_arena_allocator(options.cpu_arena)
            .build()
            .error_on_failure()])?
        .commit_from_file(model_path)
        .map_err(Into::into)
}

fn build_gpu_session(model_path: &Path, options: OnnxRuntimeOptions) -> Result<Session> {
    let mut cuda = CUDAExecutionProvider::default().with_device_id(options.device_id);
    if let Some(limit) = options.gpu_memory_limit_bytes {
        cuda = cuda.with_memory_limit(limit);
    }

    configure_session_builder(options)?
        .with_execution_providers([cuda.build().error_on_failure()])?
        .commit_from_file(model_path)
        .map_err(Into::into)
}

fn configure_session_builder(options: OnnxRuntimeOptions) -> Result<SessionBuilder> {
    let mut builder = Session::builder()?;

    if let Some(intra_threads) = options.intra_threads {
        builder = builder.with_intra_threads(intra_threads)?;
    }
    if let Some(inter_threads) = options.inter_threads {
        builder = builder.with_inter_threads(inter_threads)?;
    }

    builder = builder.with_parallel_execution(options.parallel_execution)?;
    builder = builder.with_optimization_level(options.optimization_level.into())?;
    Ok(builder)
}
