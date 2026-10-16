//! Reusable native model owner. Public server admission remains separate.
//! Requests own all sequence state; only immutable weights/device cache persist.
use super::{
    checkpoint::{Checkpoint, ProjectionStats},
    codec::{Codec, Decoded},
    generate::{self, FinishReason},
    prompt::{self, JsonTokenizer, PriorTake, Reference},
    tower::Sampling,
};
use crate::control::InferenceControl;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};
#[cfg(feature = "cuda")]
use xrt_core::gpu_resource::GpuAllocationClass;
use xrt_core::{
    gpu_resource::{GpuAllocationArena, GpuAllocationLease},
    XrtError,
};

#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    #[error("native audio inference was cancelled")]
    Cancelled,
    #[error("native audio engine is busy")]
    Busy,
    #[error("native host resource admission refused: {0}")]
    Resources(String),
    #[error("native speech reached its frame limit; incomplete audio was discarded")]
    FrameLimit,
    #[error("native speech produced no acoustic frames")]
    Empty,
    #[error(transparent)]
    Core(#[from] XrtError),
    #[error(transparent)]
    Audio(#[from] crate::AudioError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
type Result<T> = std::result::Result<T, NativeError>;

#[derive(Clone, Copy)]
pub struct Budgets {
    pub host_bytes: u64,
    pub session_bytes: u64,
    pub codec_bytes: usize,
    pub expert_bytes: u64,
    pub dense_bytes: u64,
    pub cuda_ordinal: usize,
    pub numerics: super::numerics::Numerics,
}
impl Default for Budgets {
    fn default() -> Self {
        Self {
            host_bytes: 64 * 1024 * 1024 * 1024,
            session_bytes: 2 * 1024 * 1024 * 1024,
            codec_bytes: 2 * 1024 * 1024 * 1024,
            expert_bytes: 0,
            dense_bytes: 0,
            cuda_ordinal: 0,
            numerics: Default::default(),
        }
    }
}
pub struct Request<'a> {
    pub text: &'a str,
    pub references: &'a [Reference<'a>],
    pub history: &'a [PriorTake<'a>],
    pub max_frames: usize,
    pub max_prompt_rows: usize,
    pub seed: u64,
    pub sampling: Sampling,
}
impl<'a> Request<'a> {
    pub fn plain(text: &'a str) -> Self {
        Self {
            text,
            references: &[],
            history: &[],
            max_frames: 128,
            max_prompt_rows: 8192,
            seed: 42,
            sampling: Sampling::default(),
        }
    }
}
#[derive(Debug, serde::Serialize)]
pub struct Timings {
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub codec_seconds: f64,
    pub seconds: f64,
}
pub struct Speech {
    pub audio: Decoded,
    pub frames: Vec<Vec<u32>>,
    pub timings: Timings,
    pub projections: BTreeMap<&'static str, ProjectionStats>,
}
pub struct Engine {
    pub(crate) checkpoint: Checkpoint,
    codec: Codec,
    tokenizer: JsonTokenizer,
    budgets: Budgets,
    active: Mutex<()>,
    pub load_seconds: f64,
    pub planned_host_bytes: u64,
    // Declared after GPU owners: fields drop in declaration order, so leases
    // survive CUDA cleanup, including failed/partially constructed loads.
    _gpu_leases: Vec<GpuAllocationLease>,
    // Also drops after mapped checkpoint weights and device owners.
    _bundle_reader: xrt_hub::BundleReadLease,
}
impl Engine {
    /// Host preflight can run before the host initializes a CUDA context.
    /// Current OS commit admission is implemented for the Windows target.
    pub fn preflight_host(budgets: Budgets) -> Result<()> {
        #[cfg(windows)]
        check_host_headroom(
            budgets,
            budgets
                .expert_bytes
                .checked_add(budgets.dense_bytes)
                .ok_or_else(|| NativeError::Resources("pending device bytes overflow".into()))?,
        )?;
        #[cfg(not(windows))]
        let _ = budgets;
        Ok(())
    }
    /// The caller supplies the SAME process/device admission arena used by text
    /// and image runtimes. A GPU path refuses an absent/unconfigured arena.
    pub fn load(
        model: &Path,
        codec: &Path,
        budgets: Budgets,
        arena: Option<Arc<GpuAllocationArena>>,
    ) -> Result<Self> {
        let start = std::time::Instant::now();
        Self::preflight_host(budgets)?;
        let bundle_reader = xrt_hub::BundleReadLease::acquire_at(model)?;
        let manifest = model.join("xrt.bundle.json");
        if !manifest.is_file() {
            return Err(XrtError::InvalidMetadata(
                "native engine requires a verified managed model bundle".into(),
            )
            .into());
        }
        if std::fs::metadata(&manifest)?.len() > 1024 * 1024 {
            return Err(
                XrtError::InvalidMetadata("native model manifest exceeds 1 MiB".into()).into(),
            );
        }
        let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)?)
            .map_err(|e| XrtError::InvalidMetadata(e.to_string()))?;
        if identity["revision"] != "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
            || identity["family"] != "fish-omni-drama3"
        {
            return Err(
                XrtError::InvalidMetadata("native model revision/family mismatch".into()).into(),
            );
        }
        let parsed = xrt_hub::ArtifactManifest::parse(&std::fs::read(&manifest)?)
            .map_err(|e| XrtError::InvalidMetadata(e.to_string()))?;
        if parsed
            .digest()
            .map_err(|e| XrtError::InvalidMetadata(e.to_string()))?
            != "2c67787970d622f410f1e606ea9f6cdd1c8775934ced465ded211b78540ed3e3"
        {
            return Err(XrtError::InvalidMetadata(
                "native model manifest differs from the pinned Drama 3 checkpoint".into(),
            )
            .into());
        }
        crate::installed::verify_managed(model)?;
        let mut checkpoint = Checkpoint::open(model)?;
        checkpoint.numerics = budgets.numerics;
        let tokenizer = JsonTokenizer::open(&model.join("tokenizer.json"))?;
        let codec = Codec::open(codec, budgets.codec_bytes)?;
        // Charge all mapped weights, maximum request workspace and up to 256 MiB
        // for tokenizer, source descriptors and bounded projection expansion.
        let planned_host_bytes = checkpoint
            .tensors
            .mapped_bytes()?
            .checked_add(codec.mapped_bytes()?)
            .and_then(|n| n.checked_add(budgets.session_bytes))
            .and_then(|n| n.checked_add(budgets.codec_bytes as u64))
            .and_then(|n| n.checked_add(256 * 1024 * 1024))
            .ok_or_else(|| XrtError::Runtime("native host admission overflow".into()))?;
        if planned_host_bytes > budgets.host_bytes {
            return Err(XrtError::Runtime(format!(
                "native host plan needs {planned_host_bytes} bytes, budget {}",
                budgets.host_bytes
            ))
            .into());
        }
        let mut engine = Self {
            checkpoint,
            codec,
            tokenizer,
            budgets,
            active: Mutex::new(()),
            load_seconds: 0.0,
            planned_host_bytes,
            _gpu_leases: Vec::new(),
            _bundle_reader: bundle_reader,
        };
        if budgets.expert_bytes > 0 || budgets.dense_bytes > 0 {
            #[cfg(not(feature = "cuda"))]
            return Err(XrtError::Unsupported(
                "native GPU execution requires the CUDA feature".into(),
            )
            .into());
            #[cfg(feature = "cuda")]
            {
                let arena = arena.ok_or_else(|| {
                    XrtError::Cuda("native GPU engine requires the shared admission arena".into())
                })?;
                let dense = if budgets.dense_bytes > 0 {
                    engine.checkpoint.dense_cuda_plan()?.1
                } else {
                    0
                };
                if dense > budgets.dense_bytes {
                    return Err(XrtError::Cuda(
                        "native dense plan exceeds requested budget".into(),
                    )
                    .into());
                }
                // Workspace includes the bounded FP8 pool and conservative
                // transient dense projections/module overhead, separate from
                // encoded expert/dense resident allocations.
                for (class, bytes) in [
                    (GpuAllocationClass::ModelWeights, dense),
                    (GpuAllocationClass::ExpertWeights, budgets.expert_bytes),
                    (
                        GpuAllocationClass::Scratch,
                        if dense > 0 && budgets.numerics == super::numerics::Numerics::Bf16W8a8 {
                            104 * 1024 * 1024
                        } else {
                            68 * 1024 * 1024
                        },
                    ),
                ] {
                    engine._gpu_leases.push(arena.reserve(class, bytes)?);
                }
                if dense > 0 && budgets.numerics == super::numerics::Numerics::Bf16W8a8 {
                    let t = &engine.checkpoint.config.text_config;
                    let per_layer = xrt_cuda::GpuOrderedDeltaState::planned_bytes(
                        t.linear_num_key_heads,
                        t.linear_num_value_heads,
                    )?;
                    let layers = t
                        .layer_types
                        .iter()
                        .filter(|k| *k == "linear_attention")
                        .count() as u64;
                    let planned = per_layer
                        .checked_mul(layers)
                        .ok_or_else(|| XrtError::Shape("GPU delta admission overflow".into()))?;
                    engine
                        ._gpu_leases
                        .push(arena.reserve(GpuAllocationClass::RecurrentState, planned)?);
                }
                if dense > 0 {
                    engine
                        .checkpoint
                        .enable_dense_cuda(budgets.cuda_ordinal, budgets.dense_bytes)?;
                }
                if budgets.expert_bytes > 0 {
                    engine
                        .checkpoint
                        .enable_expert_cache(budgets.cuda_ordinal, budgets.expert_bytes)?;
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = arena;
        engine.load_seconds = start.elapsed().as_secs_f64();
        Ok(engine)
    }
    pub fn synthesize(
        &self,
        request: Request<'_>,
        control: &InferenceControl,
        mut progress: impl FnMut(&str, usize),
    ) -> Result<Speech> {
        if control.is_cancelled() {
            return Err(NativeError::Cancelled);
        }
        let _active = self.active.try_lock().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => NativeError::Busy,
            _ => XrtError::Runtime("native engine lock poisoned".into()).into(),
        })?;
        let result = (|| {
            #[cfg(windows)]
            {
                #[cfg(feature = "cuda")]
                let resident = self
                    .checkpoint
                    .cache_stats()?
                    .map_or(0, |v| v.resident_bytes);
                #[cfg(not(feature = "cuda"))]
                let resident = 0;
                check_host_headroom(
                    self.budgets,
                    self.budgets.expert_bytes.saturating_sub(resident),
                )?;
            }
            request.sampling.validate()?;
            if Codec::decoding_workspace_bytes(request.max_frames)? > self.budgets.codec_bytes {
                return Err(XrtError::Runtime(
                    "native request codec workspace exceeds admitted budget".into(),
                )
                .into());
            }
            let prompt = prompt::build(
                &self.checkpoint.config,
                &self.tokenizer,
                request.text,
                request.references,
                request.history,
                request.max_prompt_rows,
            )?;
            self.checkpoint.reset_projection_profile()?;
            let start = std::time::Instant::now();
            let generated = generate::generate_with_progress(
                &self.checkpoint,
                &prompt,
                request.max_frames,
                self.budgets.session_bytes,
                request.sampling,
                request.seed,
                control,
                &mut progress,
            )?;
            if generated.finish != FinishReason::Eos {
                return Err(NativeError::FrameLimit);
            }
            if generated.frames.is_empty() {
                return Err(NativeError::Empty);
            }
            let codec_start = std::time::Instant::now();
            progress("codec", generated.frames.len());
            let audio = self.codec.decode(&generated.frames, control)?;
            let codec_seconds = codec_start.elapsed().as_secs_f64();
            Ok(Speech {
                audio,
                frames: generated.frames,
                timings: Timings {
                    prefill_seconds: generated.prefill_seconds,
                    decode_seconds: generated.decode_seconds,
                    codec_seconds,
                    seconds: start.elapsed().as_secs_f64(),
                },
                projections: self.checkpoint.projection_profile()?,
            })
        })();
        if control.is_cancelled() {
            Err(NativeError::Cancelled)
        } else {
            result
        }
    }
    pub fn encode_reference(
        &self,
        samples: &[f32],
        rate: u32,
        control: &InferenceControl,
    ) -> Result<super::codec::EncodedReference> {
        if control.is_cancelled() {
            return Err(NativeError::Cancelled);
        }
        let _active = self.active.try_lock().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => NativeError::Busy,
            _ => XrtError::Runtime("native engine lock poisoned".into()).into(),
        })?;
        let result = (|| {
            #[cfg(windows)]
            check_host_headroom(self.budgets, 0)?;
            self.codec
                .encode(samples, rate, control)
                .map_err(NativeError::from)
        })();
        if control.is_cancelled() {
            Err(NativeError::Cancelled)
        } else {
            result
        }
    }
    pub fn dense_resident_bytes(&self) -> u64 {
        #[cfg(feature = "cuda")]
        {
            self.checkpoint.dense_resident_bytes()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
    #[cfg(feature = "cuda")]
    pub fn cache_stats(&self) -> Result<Option<xrt_cuda::Fp8CacheStats>> {
        Ok(self.checkpoint.cache_stats()?)
    }
}

#[cfg(windows)]
fn check_host_headroom(budgets: Budgets, pending_device_bytes: u64) -> Result<()> {
    // Keep a 2 GiB margin for runtime/driver allocations and other processes.
    // File-backed read-only mappings are covered by the configured host plan;
    // Windows CUDA may additionally charge device allocations to system commit.
    let workspace = budgets
        .session_bytes
        .checked_add(budgets.codec_bytes as u64)
        .and_then(|v| v.checked_add(2 * 1024 * 1024 * 1024))
        .ok_or_else(|| NativeError::Resources("host workspace size overflow".into()))?;
    let commit = workspace
        .checked_add(pending_device_bytes)
        .ok_or_else(|| NativeError::Resources("host commit plan overflow".into()))?;
    xrt_core::host_memory::windows_snapshot()?
        .admit(workspace, commit)
        .map_err(|e| NativeError::Resources(e.to_string()))
}
