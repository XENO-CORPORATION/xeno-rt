use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use xrt_runtime::GpuResourceManager;

use crate::{
    encode_image,
    pipeline::{ImagePipeline, ImageRequest, PipelineExecutionPlan},
    scheduler::FlowMatchEulerSchedule,
    DecodedImage, ImageBackendKind, ImageBatchResult, ImageBatchTimings, ImageCancellation,
    ImageCapability, ImageEditRequest, ImageError, ImageExecutionPlan, ImageGenerationRequest,
    ImageModelBundle, ImageOffloadPolicy, ImageOutputFormat, ImagePreviewEvent, ImageProgressPhase,
    ImageProgressSink, ImageQuality, ImageRequestKind, ImageRequestLimits, ImageResizePolicy,
    ImageResult, ImageTimings, NormalRngV1, PlannedImageOutput,
};

use super::{
    edit_processor::qwen_image_edit_output_dimensions,
    load_vae_decoder_f32_weights, load_vae_encoder_f32_weights, open_vae_safetensors, pack_latents,
    pipeline::{
        checked_product, decoded_ncthw_to_rgba8, denormalize_vae_latents, emit, load_denoiser,
        run_denoising, PredictionPass, QwenImageDenoiser,
    },
    qwen_image_vae_decode_tiled_f32_with_control, qwen_image_vae_encode_f32_with_control,
    unpack_latents, QwenImageBundleConfig, QwenImageCpuTextEncoder, QwenImageCpuVisionEncoder,
    QwenImageEditImageBatch, QwenImageEditProcessor, QwenImagePromptEmbeddings,
    QwenImageVaeF32Weights, QwenImageVaeSource, QwenImageVaeTiling,
};

const EDIT_CPU_ACTIVATION_MULTIPLIER: u64 = 20;
const EDIT_CUDA_ACTIVATION_MULTIPLIER: u64 = 24;
const CONSERVATIVE_SOURCE_PIXELS: u64 = 1_200_000;
const CONSERVATIVE_VISUAL_TOKENS_PER_SOURCE: u64 = 256;

/// Native Qwen-Image-Edit-2511 semantic-edit adapter. CPU remains the automatic
/// fallback. Explicit CUDA selection places only the denoiser on device under
/// the sequential plan and remains experimental until its real-model gate.
pub(crate) struct QwenImageEditPipeline {
    model: String,
    bundle_digest: String,
    quantization: String,
    backend: ImageBackendKind,
    offload: ImageOffloadPolicy,
    limits: ImageRequestLimits,
    component_bytes: u64,
    config: QwenImageBundleConfig,
    processor: QwenImageEditProcessor,
    vision_encoder: QwenImageCpuVisionEncoder,
    text_encoder: QwenImageCpuTextEncoder,
    transformer: QwenImageDenoiser,
    vae_encoder_weights: QwenImageVaeF32Weights,
    vae_decoder_weights: QwenImageVaeF32Weights,
    vae_tiling: QwenImageVaeTiling,
    execution_lock: Mutex<()>,
}

struct EncodedEditSources {
    packed: Vec<f32>,
    shapes: Vec<[usize; 3]>,
    vision: super::QwenImageVisionEmbeddings,
}

impl QwenImageEditPipeline {
    pub(crate) fn new(
        bundle: ImageModelBundle,
        requested_backend: ImageBackendKind,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        if bundle.manifest().family != "qwen-image-edit" {
            return Err(ImageError::UnsupportedCapability(format!(
                "edit adapter cannot load family `{}`",
                bundle.manifest().family
            )));
        }
        let config = QwenImageBundleConfig::load(&bundle)?;
        if !config.transformer.zero_cond_t {
            return Err(ImageError::UnsupportedCapability(
                "Qwen Image Edit requires a transformer with zero_cond_t=true".to_string(),
            ));
        }
        if config.vae.input_channels != 3 || config.text_encoder.vision.is_none() {
            return Err(ImageError::UnsupportedShape(
                "Qwen Image Edit requires an RGB VAE and Qwen2.5-VL vision configuration"
                    .to_string(),
            ));
        }
        let dimension_multiple = config
            .vae
            .scale_factor()?
            .checked_mul(config.transformer.patch_size)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "Qwen Image Edit output dimension multiple overflowed".to_string(),
                )
            })?;
        let component_bytes = bundle
            .manifest()
            .components
            .iter()
            .flat_map(|component| component.files.iter())
            .try_fold(0u64, |total, file| total.checked_add(file.size_bytes))
            .ok_or_else(|| {
                ImageError::Admission("edit component byte estimate overflowed".to_string())
            })?;
        let limits = ImageRequestLimits {
            max_prompt_bytes: config.max_sequence_length.saturating_mul(16),
            max_outputs: 4,
            max_steps: 100,
            max_width: bundle.manifest().limits.max_width,
            max_height: bundle.manifest().limits.max_height,
            max_pixels: bundle.manifest().limits.max_pixels,
            dimension_multiple,
            max_source_images: 3,
        };
        let processor = QwenImageEditProcessor::load_with_config(&bundle, &config)?;
        let vision_encoder = QwenImageCpuVisionEncoder::load_with_config(&bundle, &config)?;
        let text_encoder = QwenImageCpuTextEncoder::load_with_config(&bundle, &config)?;
        // Auto stays on the portable CPU path until the complete Edit-2511
        // CUDA workload passes its admission gate. Operators may select CUDA
        // explicitly to run the bounded experimental path.
        let selected_backend = if requested_backend == ImageBackendKind::Auto {
            ImageBackendKind::Cpu
        } else {
            requested_backend
        };
        let (transformer, backend, offload) = load_denoiser(
            &bundle,
            &config,
            &bundle.manifest().quantization,
            selected_backend,
            resources,
        )?;
        let vae_store = open_vae_safetensors(&bundle, &config.vae)?;
        let vae_encoder_weights = load_vae_encoder_f32_weights(&vae_store, &config.vae)?;
        let vae_decoder_weights = load_vae_decoder_f32_weights(&vae_store, &config.vae)?;
        Ok(Self {
            model: bundle.manifest().id.clone(),
            bundle_digest: bundle.digest().to_string(),
            quantization: bundle.manifest().quantization.clone(),
            backend,
            offload,
            limits,
            component_bytes,
            config,
            processor,
            vision_encoder,
            text_encoder,
            transformer,
            vae_encoder_weights,
            vae_decoder_weights,
            vae_tiling: QwenImageVaeTiling::default(),
            execution_lock: Mutex::new(()),
        })
    }

    fn normalize_edit_request(
        &self,
        request: &ImageEditRequest,
    ) -> Result<ImageEditRequest, ImageError> {
        self.limits.validate_generation(&request.generation)?;
        if request.generation.model != self.model {
            return Err(ImageError::InvalidRequest(format!(
                "request model `{}` does not match loaded model `{}`",
                request.generation.model, self.model
            )));
        }
        if request.images.is_empty() || request.images.len() > self.limits.max_source_images {
            return Err(ImageError::InvalidRequest(format!(
                "Qwen Image Edit requires 1..={} ordered source images",
                self.limits.max_source_images
            )));
        }
        if request.mask.is_some() {
            return Err(ImageError::UnsupportedCapability(
                "mask requires image.inpaint; Qwen-Image-Edit-2511 provides semantic editing only"
                    .to_string(),
            ));
        }
        if !request.strength.is_finite() || (request.strength - 1.0).abs() > f32::EPSILON {
            return Err(ImageError::UnsupportedCapability(
                "Qwen-Image-Edit-2511 has no denoising-strength parameter; strength must remain 1.0"
                    .to_string(),
            ));
        }
        if request.generation.backend != ImageBackendKind::Auto
            && request.generation.backend != self.backend
        {
            return Err(ImageError::UnsupportedBackend(format!(
                "request selected `{}` but the loaded Qwen Image Edit plan selected `{}`",
                request.generation.backend.as_str(),
                self.backend.as_str()
            )));
        }
        if self.backend == ImageBackendKind::Cuda
            && request.generation.offload != ImageOffloadPolicy::Sequential
        {
            return Err(ImageError::UnsupportedBackend(format!(
                "the experimental Qwen Image Edit CUDA plan requires offload=sequential, found {:?}",
                request.generation.offload
            )));
        }
        let mut normalized = request.clone();
        normalized.generation.backend = self.backend;
        normalized.generation.offload = self.offload;
        if normalized.generation.resize_policy == ImageResizePolicy::RoundDown {
            normalized.generation.width -=
                normalized.generation.width % self.limits.dimension_multiple;
            normalized.generation.height -=
                normalized.generation.height % self.limits.dimension_multiple;
            normalized.generation.resize_policy = ImageResizePolicy::Reject;
            self.limits.validate_generation(&normalized.generation)?;
        }
        Ok(normalized)
    }

    fn plan_edit(&self, request: &ImageEditRequest) -> Result<ImageExecutionPlan, ImageError> {
        let generation = &request.generation;
        let outputs = (0..generation.n)
            .map(|output_index| {
                generation
                    .seed
                    .checked_add(output_index as u64)
                    .map(|seed| PlannedImageOutput { output_index, seed })
                    .ok_or_else(|| {
                        ImageError::InvalidRequest("derived output seed overflows u64".to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scale = u64::try_from(self.config.vae.scale_factor()?)
            .map_err(|_| ImageError::Admission("VAE scale does not fit u64".to_string()))?;
        let patch = u64::try_from(self.config.transformer.patch_size)
            .map_err(|_| ImageError::Admission("patch size does not fit u64".to_string()))?;
        let packed_stride = scale
            .checked_mul(patch)
            .ok_or_else(|| ImageError::Admission("packed stride overflowed".to_string()))?;
        let output_sequence = u64::from(generation.width)
            .checked_mul(u64::from(generation.height))
            .and_then(|pixels| pixels.checked_div(packed_stride.checked_mul(packed_stride)?))
            .ok_or_else(|| {
                ImageError::Admission("output sequence estimate overflowed".to_string())
            })?;
        let source_sequence = CONSERVATIVE_SOURCE_PIXELS
            .checked_div(packed_stride.checked_mul(packed_stride).ok_or_else(|| {
                ImageError::Admission("source packed stride overflowed".to_string())
            })?)
            .and_then(|sequence| sequence.checked_mul(request.images.len() as u64))
            .ok_or_else(|| {
                ImageError::Admission("source sequence estimate overflowed".to_string())
            })?;
        let visual_context = u64::try_from(request.images.len())
            .ok()
            .and_then(|count| count.checked_mul(CONSERVATIVE_VISUAL_TOKENS_PER_SOURCE))
            .ok_or_else(|| {
                ImageError::Admission("visual context estimate overflowed".to_string())
            })?;
        let inner = u64::try_from(self.config.transformer.inner_dim()?)
            .map_err(|_| ImageError::Admission("transformer width does not fit u64".to_string()))?;
        let activation_multiplier = if self.backend == ImageBackendKind::Cuda {
            EDIT_CUDA_ACTIVATION_MULTIPLIER
        } else {
            EDIT_CPU_ACTIVATION_MULTIPLIER
        };
        let activation_bytes = output_sequence
            .checked_add(source_sequence)
            .and_then(|sequence| sequence.checked_add(self.config.max_sequence_length as u64))
            .and_then(|sequence| sequence.checked_add(visual_context))
            .and_then(|rows| rows.checked_mul(inner))
            .and_then(|values| values.checked_mul(activation_multiplier))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| {
                ImageError::Admission("edit activation estimate overflowed".to_string())
            })?;
        let source_pixel_bytes = CONSERVATIVE_SOURCE_PIXELS
            .checked_mul(request.images.len() as u64)
            .and_then(|pixels| pixels.checked_mul(3 * 4))
            .ok_or_else(|| ImageError::Admission("source pixel estimate overflowed".to_string()))?;
        let vae_weight_bytes = self
            .vae_encoder_weights
            .values()
            .chain(self.vae_decoder_weights.values())
            .try_fold(0u64, |total, values| {
                total.checked_add((values.len() as u64).saturating_mul(4))
            })
            .ok_or_else(|| ImageError::Admission("edit VAE estimate overflowed".to_string()))?;
        let estimated_host_bytes = self
            .component_bytes
            .checked_add(vae_weight_bytes)
            .and_then(|bytes| bytes.checked_add(activation_bytes))
            .and_then(|bytes| bytes.checked_add(source_pixel_bytes))
            .ok_or_else(|| ImageError::Admission("edit host estimate overflowed".to_string()))?;
        let estimated_device_bytes = if self.backend == ImageBackendKind::Cuda {
            self.transformer
                .device_weight_bytes()
                .checked_add(activation_bytes)
                .ok_or_else(|| {
                    ImageError::Admission("edit device byte estimate overflowed".to_string())
                })?
        } else {
            0
        };
        Ok(ImageExecutionPlan {
            request_kind: ImageRequestKind::Edit,
            model: generation.model.clone(),
            bundle_digest: self.bundle_digest.clone(),
            backend: self.backend,
            offload: self.offload,
            width: generation.width,
            height: generation.height,
            steps: generation.steps,
            outputs,
            estimated_host_bytes,
            estimated_device_bytes,
        })
    }

    fn encode_sources(
        &self,
        batch: QwenImageEditImageBatch,
        cancellation: &ImageCancellation,
    ) -> Result<EncodedEditSources, ImageError> {
        let QwenImageEditImageBatch {
            vision,
            vae_sources,
            condition_sizes: _,
            vae_sizes: _,
        } = batch;
        let vision_embeddings = self
            .vision_encoder
            .encode_with_control(&vision, |_| cancellation.check())?;
        drop(vision);
        let scale = self.config.vae.scale_factor()?;
        let patch = self.config.transformer.patch_size;
        let mut packed_sources = Vec::new();
        let mut shapes = Vec::with_capacity(vae_sources.len());
        for source in vae_sources {
            cancellation.check()?;
            let (packed, shape) = self.encode_vae_source(source, scale, patch, cancellation)?;
            packed_sources.extend(packed);
            shapes.push(shape);
        }
        Ok(EncodedEditSources {
            packed: packed_sources,
            shapes,
            vision: vision_embeddings,
        })
    }

    fn encode_vae_source(
        &self,
        source: QwenImageVaeSource,
        scale: usize,
        patch: usize,
        cancellation: &ImageCancellation,
    ) -> Result<(Vec<f32>, [usize; 3]), ImageError> {
        let height = usize::try_from(source.height).map_err(|_| {
            ImageError::UnsupportedShape("source height does not fit usize".to_string())
        })?;
        let width = usize::try_from(source.width).map_err(|_| {
            ImageError::UnsupportedShape("source width does not fit usize".to_string())
        })?;
        let packed_stride = scale
            .checked_mul(patch)
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "source packed stride overflowed or is zero".to_string(),
                )
            })?;
        if height % packed_stride != 0 || width % packed_stride != 0 {
            return Err(ImageError::UnsupportedShape(format!(
                "source reconstruction size {width}x{height} is not divisible by {}",
                packed_stride
            )));
        }
        let mut latents = qwen_image_vae_encode_f32_with_control(
            &self.config.vae,
            &self.vae_encoder_weights,
            &source.values,
            1,
            1,
            height,
            width,
            |_| cancellation.check(),
        )?;
        normalize_vae_latents(
            &mut latents,
            &self.config.vae.latents_mean,
            &self.config.vae.latents_std,
        )?;
        let latent_height = height / scale;
        let latent_width = width / scale;
        let packed = pack_latents(
            &latents,
            1,
            self.config.vae.z_dim,
            latent_height,
            latent_width,
        )?;
        Ok((packed, [1, latent_height / patch, latent_width / patch]))
    }

    fn encode_prompt(
        &self,
        prompt: &str,
        vision: &super::QwenImageVisionEmbeddings,
        cancellation: &ImageCancellation,
    ) -> Result<QwenImagePromptEmbeddings, ImageError> {
        let tokens = self
            .processor
            .tokenize_prompt(prompt, &vision.image_token_counts)?;
        self.text_encoder
            .encode_multimodal_tokens_with_control(&tokens, vision, |_, _| cancellation.check())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_packed_latents(
        &self,
        packed: &[f32],
        latent_height: usize,
        latent_width: usize,
        patch_height: usize,
        patch_width: usize,
        output_height: u32,
        output_width: u32,
        cancellation: &ImageCancellation,
    ) -> Result<DecodedImage, ImageError> {
        let mut latents = unpack_latents(
            packed,
            1,
            self.config.transformer.in_channels,
            patch_height,
            patch_width,
        )?;
        denormalize_vae_latents(
            &mut latents,
            &self.config.vae.latents_mean,
            &self.config.vae.latents_std,
        )?;
        let decoded = qwen_image_vae_decode_tiled_f32_with_control(
            &self.config.vae,
            &self.vae_decoder_weights,
            &latents,
            1,
            1,
            latent_height,
            latent_width,
            self.vae_tiling,
            |_| cancellation.check(),
        )?;
        cancellation.check()?;
        decoded_ncthw_to_rgba8(
            &decoded,
            self.config.vae.input_channels,
            usize::try_from(output_height).map_err(|_| {
                ImageError::UnsupportedShape("output height does not fit usize".to_string())
            })?,
            usize::try_from(output_width).map_err(|_| {
                ImageError::UnsupportedShape("output width does not fit usize".to_string())
            })?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_one(
        &self,
        request: &ImageGenerationRequest,
        positive_prompt: &QwenImagePromptEmbeddings,
        negative_prompt: Option<&QwenImagePromptEmbeddings>,
        source_packed: &[f32],
        source_shapes: &[[usize; 3]],
        output_index: usize,
        seed: u64,
        prompt_encoding_ms: f64,
        source_encoding_ms: f64,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageResult, ImageError> {
        let started = Instant::now();
        let scale = self.config.vae.scale_factor()?;
        let latent_height = usize::try_from(request.height)
            .map_err(|_| ImageError::UnsupportedShape("height does not fit usize".to_string()))?
            / scale;
        let latent_width = usize::try_from(request.width)
            .map_err(|_| ImageError::UnsupportedShape("width does not fit usize".to_string()))?
            / scale;
        let latent_values = checked_product(
            &[self.config.vae.z_dim, latent_height, latent_width],
            "edit initial latent tensor",
        )?;
        let mut unpacked = vec![0.0f32; latent_values];
        NormalRngV1::new(seed).fill_f32(&mut unpacked);
        let mut packed = pack_latents(
            &unpacked,
            1,
            self.config.vae.z_dim,
            latent_height,
            latent_width,
        )?;
        let patch_height = latent_height / self.config.transformer.patch_size;
        let patch_width = latent_width / self.config.transformer.patch_size;
        let image_sequence = checked_product(&[patch_height, patch_width], "edit output sequence")?;
        let schedule = FlowMatchEulerSchedule::new(
            self.config.scheduler.clone(),
            request.steps,
            image_sequence,
        )?;
        let mut image_shapes = Vec::with_capacity(source_shapes.len() + 1);
        image_shapes.push([1, patch_height, patch_width]);
        image_shapes.extend_from_slice(source_shapes);
        let combined_values = packed
            .len()
            .checked_add(source_packed.len())
            .ok_or_else(|| {
                ImageError::UnsupportedShape("combined edit latent length overflow".to_string())
            })?;
        let mut transformer_input = Vec::with_capacity(combined_values);

        let denoise_started = Instant::now();
        let do_true_cfg = request.true_cfg_scale > 1.0 && negative_prompt.is_some();
        run_denoising(
            &mut packed,
            &schedule,
            request.true_cfg_scale,
            do_true_cfg,
            self.config.transformer.in_channels,
            |latents, timestep, pass| {
                let prompt = match pass {
                    PredictionPass::Conditional => positive_prompt,
                    PredictionPass::Unconditional => negative_prompt.ok_or_else(|| {
                        ImageError::Internal(
                            "unconditional edit denoising requested without embeddings".to_string(),
                        )
                    })?,
                };
                transformer_input.clear();
                transformer_input.extend_from_slice(latents);
                transformer_input.extend_from_slice(source_packed);
                let prediction = self.transformer.forward_edit_with_control(
                    &transformer_input,
                    prompt,
                    &[timestep],
                    &image_shapes,
                    |_| cancellation.check(),
                )?;
                slice_edit_prediction(prediction, transformer_input.len(), latents.len())
            },
            |step| {
                cancellation.check()?;
                emit(
                    progress,
                    output_index,
                    ImageProgressPhase::Denoising,
                    Some(step),
                    Some(request.steps),
                );
                Ok(())
            },
            |completed_steps, latents| {
                let should_preview = request.preview_interval.is_some_and(|interval| {
                    completed_steps < request.steps && completed_steps % interval == 0
                }) && progress.is_some_and(|sink| sink.wants_previews());
                if !should_preview {
                    return Ok(());
                }
                let image = self.decode_packed_latents(
                    latents,
                    latent_height,
                    latent_width,
                    patch_height,
                    patch_width,
                    request.height,
                    request.width,
                    cancellation,
                )?;
                let bytes = encode_image(
                    &image,
                    request.output_format,
                    output_quality(request.output_format, request.quality),
                    64 * 1024 * 1024,
                )?;
                if let Some(progress) = progress {
                    progress.on_preview(&ImagePreviewEvent {
                        output_index,
                        step: completed_steps,
                        total_steps: request.steps,
                        bytes: Arc::from(bytes),
                        mime_type: request.output_format.mime_type().to_string(),
                        width: request.width,
                        height: request.height,
                    });
                }
                Ok(())
            },
        )?;
        let denoising_ms = denoise_started.elapsed().as_secs_f64() * 1_000.0;

        cancellation.check()?;
        emit(
            progress,
            output_index,
            ImageProgressPhase::VaeDecode,
            None,
            None,
        );
        let vae_started = Instant::now();
        let image = self.decode_packed_latents(
            &packed,
            latent_height,
            latent_width,
            patch_height,
            patch_width,
            request.height,
            request.width,
            cancellation,
        )?;
        let vae_decode_ms = vae_started.elapsed().as_secs_f64() * 1_000.0;
        emit(
            progress,
            output_index,
            ImageProgressPhase::Encoding,
            None,
            None,
        );
        let encoding_started = Instant::now();
        let bytes = encode_image(
            &image,
            request.output_format,
            output_quality(request.output_format, request.quality),
            64 * 1024 * 1024,
        )?;
        let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;
        emit(
            progress,
            output_index,
            ImageProgressPhase::Complete,
            None,
            None,
        );
        Ok(ImageResult {
            bytes,
            mime_type: request.output_format.mime_type().to_string(),
            width: request.width,
            height: request.height,
            seed,
            model: self.model.clone(),
            bundle_digest: self.bundle_digest.clone(),
            backend: self.backend,
            quantization: self.quantization.clone(),
            timings: ImageTimings {
                prompt_encoding_ms,
                source_encoding_ms,
                denoising_ms,
                vae_decode_ms,
                encoding_ms,
                total_ms: prompt_encoding_ms
                    + source_encoding_ms
                    + started.elapsed().as_secs_f64() * 1_000.0,
            },
        })
    }
}

impl ImagePipeline for QwenImageEditPipeline {
    fn capabilities(&self) -> &[ImageCapability] {
        const CAPABILITIES: &[ImageCapability] = &[ImageCapability::Edit];
        CAPABILITIES
    }

    fn backend(&self) -> ImageBackendKind {
        self.backend
    }

    fn default_edit_dimensions(&self, images: &[DecodedImage]) -> Result<(u32, u32), ImageError> {
        let source = images.last().ok_or_else(|| {
            ImageError::InvalidRequest(
                "Qwen Image Edit default size requires a source image".to_string(),
            )
        })?;
        qwen_image_edit_output_dimensions(source.width(), source.height())
    }

    fn plan(&self, request: &ImageRequest) -> Result<PipelineExecutionPlan, ImageError> {
        match request {
            ImageRequest::Generation(_) => Err(ImageError::UnsupportedCapability(
                "Qwen-Image-Edit-2511 does not advertise image.generate".to_string(),
            )),
            ImageRequest::Edit(request) => {
                let normalized = self.normalize_edit_request(request)?;
                let public = self.plan_edit(&normalized)?;
                Ok(PipelineExecutionPlan {
                    public,
                    request: ImageRequest::Edit(normalized),
                })
            }
        }
    }

    fn execute(
        &self,
        plan: PipelineExecutionPlan,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageBatchResult, ImageError> {
        let total_started = Instant::now();
        let queue_started = Instant::now();
        let _execution_guard = loop {
            cancellation.check()?;
            if let Some(guard) = self.execution_lock.try_lock_for(Duration::from_millis(50)) {
                break guard;
            }
        };
        let queue_ms = queue_started.elapsed().as_secs_f64() * 1_000.0;
        cancellation.check()?;
        let execution_started = Instant::now();
        let ImageRequest::Edit(request) = &plan.request else {
            return Err(ImageError::Internal(
                "Qwen edit pipeline received a non-edit plan".to_string(),
            ));
        };
        for output in &plan.public.outputs {
            emit(
                progress,
                output.output_index,
                ImageProgressPhase::Admitted,
                None,
                None,
            );
            emit(
                progress,
                output.output_index,
                ImageProgressPhase::SourceEncoding,
                None,
                None,
            );
        }
        let source_started = Instant::now();
        let processed = self.processor.process_images(&request.images)?;
        let sources = self.encode_sources(processed, cancellation)?;
        let source_encoding_ms = source_started.elapsed().as_secs_f64() * 1_000.0;

        for output in &plan.public.outputs {
            emit(
                progress,
                output.output_index,
                ImageProgressPhase::PromptEncoding,
                None,
                None,
            );
        }
        let prompt_started = Instant::now();
        let positive_prompt =
            self.encode_prompt(&request.generation.prompt, &sources.vision, cancellation)?;
        let negative_prompt = if request.generation.true_cfg_scale > 1.0 {
            request
                .generation
                .negative_prompt
                .as_deref()
                .map(|prompt| self.encode_prompt(prompt, &sources.vision, cancellation))
                .transpose()?
        } else {
            None
        };
        let prompt_encoding_ms = prompt_started.elapsed().as_secs_f64() * 1_000.0;
        cancellation.check()?;

        let mut images = Vec::with_capacity(plan.public.outputs.len());
        for output in &plan.public.outputs {
            images.push(self.execute_one(
                &request.generation,
                &positive_prompt,
                negative_prompt.as_ref(),
                &sources.packed,
                &sources.shapes,
                output.output_index,
                output.seed,
                prompt_encoding_ms,
                source_encoding_ms,
                cancellation,
                progress,
            )?);
        }
        let execution_ms = execution_started.elapsed().as_secs_f64() * 1_000.0;
        Ok(ImageBatchResult {
            images,
            timings: ImageBatchTimings {
                admission_ms: 0.0,
                queue_ms,
                execution_ms,
                total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
            },
        })
    }
}

fn normalize_vae_latents(
    latents: &mut [f32],
    means: &[f32],
    standard_deviations: &[f32],
) -> Result<(), ImageError> {
    if means.is_empty()
        || means.len() != standard_deviations.len()
        || latents.len() % means.len() != 0
        || standard_deviations
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ImageError::UnsupportedShape(
            "invalid source VAE latent normalization geometry".to_string(),
        ));
    }
    let plane = latents.len() / means.len();
    for (channel, values) in latents.chunks_exact_mut(plane).enumerate() {
        for value in values {
            *value = (*value - means[channel]) / standard_deviations[channel];
        }
    }
    if latents.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "source_vae_latent_normalization",
            step: 0,
        });
    }
    Ok(())
}

fn slice_edit_prediction(
    mut prediction: Vec<f32>,
    combined_values: usize,
    output_values: usize,
) -> Result<Vec<f32>, ImageError> {
    if prediction.len() != combined_values || output_values > combined_values {
        return Err(ImageError::UnsupportedShape(format!(
            "edit transformer returned {} values, expected {combined_values} before slicing {output_values} output values",
            prediction.len()
        )));
    }
    prediction.truncate(output_values);
    Ok(prediction)
}

fn output_quality(format: ImageOutputFormat, quality: ImageQuality) -> u8 {
    match (format, quality) {
        (ImageOutputFormat::Jpeg, ImageQuality::Hd) => 95,
        _ => 90,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_latent_normalization_is_inverse_of_decode_normalization() {
        let mut values = vec![2.5, 4.5, -3.5, -6.5];
        normalize_vae_latents(&mut values, &[0.5, -0.5], &[2.0, 3.0]).unwrap();
        assert_eq!(values, [1.0, 2.0, -1.0, -2.0]);
        denormalize_vae_latents(&mut values, &[0.5, -0.5], &[2.0, 3.0]).unwrap();
        assert_eq!(values, [2.5, 4.5, -3.5, -6.5]);
    }

    #[test]
    fn edit_prediction_keeps_output_sequence_before_source_conditioning() {
        assert_eq!(
            slice_edit_prediction(vec![1.0, 2.0, 10.0, 20.0], 4, 2).unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            slice_edit_prediction(vec![1.0, 2.0, 3.0], 4, 2)
                .unwrap_err()
                .kind(),
            crate::ImageErrorKind::UnsupportedShape
        );
    }
}
