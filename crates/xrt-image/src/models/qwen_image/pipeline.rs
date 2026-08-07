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
    ImageCapability, ImageError, ImageExecutionPlan, ImageGenerationRequest, ImageModelBundle,
    ImageOffloadPolicy, ImageOutputFormat, ImagePreviewEvent, ImageProgressEvent,
    ImageProgressPhase, ImageProgressSink, ImageQuality, ImageRequestKind, ImageRequestLimits,
    ImageResizePolicy, ImageResult, ImageTimings, NormalRngV1, PlannedImageOutput,
};

use super::{
    load_vae_decoder_f32_weights, open_transformer_adapter, open_transformer_gguf,
    open_transformer_safetensors, open_vae_safetensors, pack_latents,
    qwen_image_vae_decode_tiled_f32_with_control, unpack_latents, QwenImageBf16Transformer,
    QwenImageBundleConfig, QwenImageCpuTextEncoder, QwenImageDistilledProfile,
    QwenImageGgufTransformer, QwenImagePromptEmbeddings, QwenImagePromptTokenizer,
    QwenImageVaeF32Weights, QwenImageVaeTiling,
};

#[cfg(feature = "cuda")]
use super::QwenImageCudaTransformer;

pub(super) enum QwenImageDenoiser {
    Bf16(QwenImageBf16Transformer),
    Gguf(QwenImageGgufTransformer),
    #[cfg(feature = "cuda")]
    Cuda(QwenImageCudaTransformer),
}

impl QwenImageDenoiser {
    #[allow(clippy::too_many_arguments)]
    fn forward_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        frames: usize,
        patch_height: usize,
        patch_width: usize,
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        match self {
            Self::Bf16(transformer) => transformer.forward_with_control(
                packed_latents,
                prompt,
                timestep,
                frames,
                patch_height,
                patch_width,
                checkpoint,
            ),
            Self::Gguf(transformer) => transformer.forward_with_control(
                packed_latents,
                prompt,
                timestep,
                frames,
                patch_height,
                patch_width,
                checkpoint,
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda(transformer) => transformer.forward_with_control(
                packed_latents,
                prompt,
                timestep,
                frames,
                patch_height,
                patch_width,
                checkpoint,
            ),
        }
    }

    pub(super) fn forward_edit_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        image_shapes: &[[usize; 3]],
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        match self {
            Self::Bf16(transformer) => transformer.forward_edit_with_control(
                packed_latents,
                prompt,
                timestep,
                image_shapes,
                checkpoint,
            ),
            Self::Gguf(transformer) => transformer.forward_edit_with_control(
                packed_latents,
                prompt,
                timestep,
                image_shapes,
                checkpoint,
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda(transformer) => transformer.forward_edit_with_control(
                packed_latents,
                prompt,
                timestep,
                image_shapes,
                checkpoint,
            ),
        }
    }

    pub(super) fn device_weight_bytes(&self) -> u64 {
        match self {
            Self::Bf16(_) | Self::Gguf(_) => 0,
            #[cfg(feature = "cuda")]
            Self::Cuda(transformer) => transformer.weight_bytes(),
        }
    }

    fn distilled_profile(&self) -> Option<QwenImageDistilledProfile> {
        match self {
            Self::Bf16(transformer) => transformer.distilled_profile(),
            Self::Gguf(transformer) => transformer.distilled_profile(),
            #[cfg(feature = "cuda")]
            Self::Cuda(transformer) => transformer.distilled_profile(),
        }
    }
}

pub(super) fn load_cpu_denoiser(
    bundle: &ImageModelBundle,
    config: &QwenImageBundleConfig,
    quantization: &str,
) -> Result<QwenImageDenoiser, ImageError> {
    let adapter = open_transformer_adapter(bundle, &config.transformer)?;
    match quantization {
        "BF16" => {
            let store = open_transformer_safetensors(bundle, &config.transformer)?;
            Ok(QwenImageDenoiser::Bf16(
                QwenImageBf16Transformer::from_store_with_adapter(
                    store,
                    config.transformer.clone(),
                    adapter,
                )?,
            ))
        }
        "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M" => {
            let file = open_transformer_gguf(bundle, &config.transformer)?;
            Ok(QwenImageDenoiser::Gguf(
                QwenImageGgufTransformer::from_file_with_adapter(
                    file,
                    config.transformer.clone(),
                    quantization,
                    adapter,
                )?,
            ))
        }
        other => Err(ImageError::UnsupportedQuantization(other.to_string())),
    }
}

#[cfg(feature = "cuda")]
fn load_cuda_denoiser(
    bundle: &ImageModelBundle,
    config: &QwenImageBundleConfig,
    quantization: &str,
    resources: Arc<GpuResourceManager>,
) -> Result<QwenImageDenoiser, ImageError> {
    if quantization == "BF16" {
        return Err(ImageError::UnsupportedBackend(
            "Qwen Image BF16 CUDA residency is not admitted on the initial 24 GiB tier; use CPU or a validated GGUF bundle"
                .to_string(),
        ));
    }
    if !matches!(quantization, "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M") {
        return Err(ImageError::UnsupportedQuantization(
            quantization.to_string(),
        ));
    }
    let adapter = open_transformer_adapter(bundle, &config.transformer)?;
    let file = open_transformer_gguf(bundle, &config.transformer)?;
    Ok(QwenImageDenoiser::Cuda(
        QwenImageCudaTransformer::from_file_with_adapter(
            file,
            config.transformer.clone(),
            quantization,
            resources,
            adapter,
        )?,
    ))
}

pub(super) fn load_denoiser(
    bundle: &ImageModelBundle,
    config: &QwenImageBundleConfig,
    quantization: &str,
    requested_backend: ImageBackendKind,
    resources: Arc<GpuResourceManager>,
) -> Result<(QwenImageDenoiser, ImageBackendKind, ImageOffloadPolicy), ImageError> {
    if !matches!(quantization, "BF16" | "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M") {
        return Err(ImageError::UnsupportedQuantization(
            quantization.to_string(),
        ));
    }

    match requested_backend {
        ImageBackendKind::Cpu => Ok((
            load_cpu_denoiser(bundle, config, quantization)?,
            ImageBackendKind::Cpu,
            ImageOffloadPolicy::Cpu,
        )),
        ImageBackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok((
                    load_cuda_denoiser(bundle, config, quantization, resources)?,
                    ImageBackendKind::Cuda,
                    ImageOffloadPolicy::Sequential,
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = resources;
                Err(ImageError::UnsupportedBackend(
                    "xrt-image was built without the `cuda` feature".to_string(),
                ))
            }
        }
        ImageBackendKind::Auto => {
            #[cfg(feature = "cuda")]
            {
                if quantization != "BF16" {
                    if let Ok(transformer) =
                        load_cuda_denoiser(bundle, config, quantization, Arc::clone(&resources))
                    {
                        return Ok((
                            transformer,
                            ImageBackendKind::Cuda,
                            ImageOffloadPolicy::Sequential,
                        ));
                    }
                }
            }
            let _ = resources;
            Ok((
                load_cpu_denoiser(bundle, config, quantization)?,
                ImageBackendKind::Cpu,
                ImageOffloadPolicy::Cpu,
            ))
        }
    }
}

/// Native Qwen Image generation adapter with portable CPU execution and an
/// eager CUDA denoiser under the explicit sequential component-offload plan.
pub(crate) struct QwenImagePipeline {
    model: String,
    bundle_digest: String,
    quantization: String,
    backend: ImageBackendKind,
    offload: ImageOffloadPolicy,
    limits: ImageRequestLimits,
    component_bytes: u64,
    config: QwenImageBundleConfig,
    tokenizer: QwenImagePromptTokenizer,
    text_encoder: QwenImageCpuTextEncoder,
    transformer: QwenImageDenoiser,
    vae_weights: QwenImageVaeF32Weights,
    vae_tiling: QwenImageVaeTiling,
    execution_lock: Mutex<()>,
}

impl QwenImagePipeline {
    pub(crate) fn new(
        bundle: ImageModelBundle,
        requested_backend: ImageBackendKind,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        if bundle.manifest().family != "qwen-image" {
            return Err(ImageError::UnsupportedCapability(format!(
                "generation adapter cannot load family `{}`",
                bundle.manifest().family
            )));
        }
        let config = QwenImageBundleConfig::load(&bundle)?;
        if config.vae.input_channels != 3 {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image generation requires a three-channel VAE, found {}",
                config.vae.input_channels
            )));
        }
        let dimension_multiple = config
            .vae
            .scale_factor()?
            .checked_mul(config.transformer.patch_size)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "Qwen Image output dimension multiple overflowed".to_string(),
                )
            })?;
        let component_bytes = bundle
            .manifest()
            .components
            .iter()
            .flat_map(|component| component.files.iter())
            .try_fold(0u64, |total, file| total.checked_add(file.size_bytes))
            .ok_or_else(|| {
                ImageError::Admission("component byte estimate overflowed".to_string())
            })?;
        let model = bundle.manifest().id.clone();
        let bundle_digest = bundle.digest().to_string();
        let quantization = bundle.manifest().quantization.clone();
        let limits = ImageRequestLimits {
            max_prompt_bytes: config.max_sequence_length.saturating_mul(16),
            max_outputs: 4,
            max_steps: 100,
            max_width: bundle.manifest().limits.max_width,
            max_height: bundle.manifest().limits.max_height,
            max_pixels: bundle.manifest().limits.max_pixels,
            dimension_multiple,
            max_source_images: 0,
        };

        let tokenizer = QwenImagePromptTokenizer::load(
            &bundle,
            config.max_sequence_length,
            config.text_encoder.vocab_size,
        )?;
        let text_encoder = QwenImageCpuTextEncoder::load_with_config(&bundle, &config)?;
        let (transformer, backend, offload) = load_denoiser(
            &bundle,
            &config,
            &quantization,
            requested_backend,
            resources,
        )?;
        let vae_store = open_vae_safetensors(&bundle, &config.vae)?;
        let vae_weights = load_vae_decoder_f32_weights(&vae_store, &config.vae)?;

        Ok(Self {
            model,
            bundle_digest,
            quantization,
            backend,
            offload,
            limits,
            component_bytes,
            config,
            tokenizer,
            text_encoder,
            transformer,
            vae_weights,
            vae_tiling: QwenImageVaeTiling::default(),
            execution_lock: Mutex::new(()),
        })
    }

    fn normalize_generation_request(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationRequest, ImageError> {
        self.limits.validate_generation(request)?;
        if request.model != self.model {
            return Err(ImageError::InvalidRequest(format!(
                "request model `{}` does not match loaded model `{}`",
                request.model, self.model
            )));
        }
        if request.backend != ImageBackendKind::Auto && request.backend != self.backend {
            return Err(ImageError::UnsupportedBackend(format!(
                "request selected `{}` but the loaded Qwen Image plan selected `{}`",
                request.backend.as_str(),
                self.backend.as_str()
            )));
        }
        if self.backend == ImageBackendKind::Cuda
            && request.offload != ImageOffloadPolicy::Sequential
        {
            return Err(ImageError::UnsupportedBackend(format!(
                "the initial Qwen Image CUDA plan requires offload=sequential, found {:?}",
                request.offload
            )));
        }
        let mut normalized = request.clone();
        normalized.backend = self.backend;
        normalized.offload = self.offload;
        if normalized.resize_policy == ImageResizePolicy::RoundDown {
            normalized.width -= normalized.width % self.limits.dimension_multiple;
            normalized.height -= normalized.height % self.limits.dimension_multiple;
            normalized.resize_policy = ImageResizePolicy::Reject;
            self.limits.validate_generation(&normalized)?;
        }
        if let Some(profile) = self.transformer.distilled_profile() {
            profile.validate_request(&normalized)?;
        }
        Ok(normalized)
    }

    fn plan_generation(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageExecutionPlan, ImageError> {
        let outputs = (0..request.n)
            .map(|output_index| {
                request
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
        let latent_height = u64::from(request.height) / scale;
        let latent_width = u64::from(request.width) / scale;
        let image_sequence = latent_height
            .checked_mul(latent_width)
            .and_then(|value| {
                let patch = self.config.transformer.patch_size as u64;
                value.checked_div(patch.checked_mul(patch)?)
            })
            .ok_or_else(|| {
                ImageError::Admission("image sequence estimate overflowed".to_string())
            })?;
        let inner = u64::try_from(self.config.transformer.inner_dim()?)
            .map_err(|_| ImageError::Admission("transformer width does not fit u64".to_string()))?;
        let text_sequence = self.config.max_sequence_length as u64;
        let activation_multiplier = if self.backend == ImageBackendKind::Cuda {
            24
        } else {
            20
        };
        let activation_elements = image_sequence
            .checked_add(text_sequence)
            .and_then(|rows| rows.checked_mul(inner))
            // Joint attention projections, modulation, and MLP scratch are
            // short-lived but overlap in the scalar reference executor.
            .and_then(|values| values.checked_mul(activation_multiplier))
            .ok_or_else(|| ImageError::Admission("activation estimate overflowed".to_string()))?;
        let activation_bytes = activation_elements.checked_mul(4).ok_or_else(|| {
            ImageError::Admission("activation byte estimate overflowed".to_string())
        })?;
        let vae_decoder_bytes = self
            .vae_weights
            .values()
            .try_fold(0u64, |total, values| {
                total.checked_add((values.len() as u64).saturating_mul(4))
            })
            .ok_or_else(|| ImageError::Admission("VAE byte estimate overflowed".to_string()))?;
        let estimated_host_bytes = self
            .component_bytes
            .checked_add(vae_decoder_bytes)
            .and_then(|bytes| bytes.checked_add(activation_bytes))
            .ok_or_else(|| ImageError::Admission("host byte estimate overflowed".to_string()))?;
        let estimated_device_bytes = if self.backend == ImageBackendKind::Cuda {
            self.transformer
                .device_weight_bytes()
                .checked_add(activation_bytes)
                .ok_or_else(|| {
                    ImageError::Admission("device byte estimate overflowed".to_string())
                })?
        } else {
            0
        };

        Ok(ImageExecutionPlan {
            request_kind: ImageRequestKind::Generate,
            model: request.model.clone(),
            bundle_digest: self.bundle_digest.clone(),
            backend: self.backend,
            offload: self.offload,
            width: request.width,
            height: request.height,
            steps: request.steps,
            outputs,
            estimated_host_bytes,
            estimated_device_bytes,
        })
    }

    fn encode_prompt(
        &self,
        prompt: &str,
        cancellation: &ImageCancellation,
    ) -> Result<QwenImagePromptEmbeddings, ImageError> {
        let tokens = self.tokenizer.encode_batch(&[prompt])?;
        self.text_encoder
            .encode_tokens_with_control(&tokens, |_, _| cancellation.check())
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
            &self.vae_weights,
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
        output_index: usize,
        seed: u64,
        prompt_encoding_ms: f64,
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
            "initial latent tensor",
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
        let image_sequence = checked_product(&[patch_height, patch_width], "image sequence")?;
        let schedule = FlowMatchEulerSchedule::new(
            self.config.scheduler.clone(),
            request.steps,
            image_sequence,
        )?;

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
                            "unconditional denoising requested without embeddings".to_string(),
                        )
                    })?,
                };
                self.transformer.forward_with_control(
                    latents,
                    prompt,
                    &[timestep],
                    1,
                    patch_height,
                    patch_width,
                    |_| cancellation.check(),
                )
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
                cancellation.check()?;
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
                let quality = match (request.output_format, request.quality) {
                    (ImageOutputFormat::Jpeg, ImageQuality::Hd) => 95,
                    (ImageOutputFormat::Jpeg, ImageQuality::Standard) => 90,
                    _ => 90,
                };
                let bytes = encode_image(&image, request.output_format, quality, 64 * 1024 * 1024)?;
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

        cancellation.check()?;
        emit(
            progress,
            output_index,
            ImageProgressPhase::Encoding,
            None,
            None,
        );
        let encoding_started = Instant::now();
        let quality = match (request.output_format, request.quality) {
            (ImageOutputFormat::Jpeg, ImageQuality::Hd) => 95,
            (ImageOutputFormat::Jpeg, ImageQuality::Standard) => 90,
            _ => 90,
        };
        let bytes = encode_image(&image, request.output_format, quality, 64 * 1024 * 1024)?;
        let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = prompt_encoding_ms + started.elapsed().as_secs_f64() * 1_000.0;
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
                source_encoding_ms: 0.0,
                denoising_ms,
                vae_decode_ms,
                encoding_ms,
                total_ms,
            },
        })
    }
}

impl ImagePipeline for QwenImagePipeline {
    fn capabilities(&self) -> &[ImageCapability] {
        const CAPABILITIES: &[ImageCapability] = &[ImageCapability::Generate];
        CAPABILITIES
    }

    fn backend(&self) -> ImageBackendKind {
        self.backend
    }

    fn plan(&self, request: &ImageRequest) -> Result<PipelineExecutionPlan, ImageError> {
        match request {
            ImageRequest::Generation(request) => {
                let normalized = self.normalize_generation_request(request)?;
                let public = self.plan_generation(&normalized)?;
                Ok(PipelineExecutionPlan {
                    public,
                    request: ImageRequest::Generation(normalized),
                })
            }
            ImageRequest::Edit(request) => Err(ImageError::UnsupportedCapability(format!(
                "image.edit with {} source image(s) is not implemented by the Qwen Image generation adapter",
                request.images.len()
            ))),
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
        let ImageRequest::Generation(request) = &plan.request else {
            return Err(ImageError::Internal(
                "Qwen generation pipeline received a non-generation plan".to_string(),
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
                ImageProgressPhase::PromptEncoding,
                None,
                None,
            );
        }
        let prompt_started = Instant::now();
        let positive_prompt = self.encode_prompt(&request.prompt, cancellation)?;
        cancellation.check()?;
        let negative_prompt = if request.true_cfg_scale > 1.0 {
            request
                .negative_prompt
                .as_deref()
                .map(|prompt| self.encode_prompt(prompt, cancellation))
                .transpose()?
        } else {
            None
        };
        cancellation.check()?;
        let prompt_encoding_ms = prompt_started.elapsed().as_secs_f64() * 1_000.0;

        let mut images = Vec::with_capacity(plan.public.outputs.len());
        for output in &plan.public.outputs {
            images.push(self.execute_one(
                request,
                &positive_prompt,
                negative_prompt.as_ref(),
                output.output_index,
                output.seed,
                prompt_encoding_ms,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PredictionPass {
    Conditional,
    Unconditional,
}

pub(super) fn run_denoising<F, C, P>(
    latents: &mut [f32],
    schedule: &FlowMatchEulerSchedule,
    true_cfg_scale: f32,
    do_true_cfg: bool,
    packed_channels: usize,
    mut predict: F,
    mut checkpoint: C,
    mut preview: P,
) -> Result<(), ImageError>
where
    F: FnMut(&[f32], f32, PredictionPass) -> Result<Vec<f32>, ImageError>,
    C: FnMut(usize) -> Result<(), ImageError>,
    P: FnMut(usize, &[f32]) -> Result<(), ImageError>,
{
    if !true_cfg_scale.is_finite() || true_cfg_scale < 0.0 {
        return Err(ImageError::InvalidRequest(
            "true_cfg_scale must be finite and non-negative".to_string(),
        ));
    }
    for (step, timestep) in schedule.timesteps().iter().copied().enumerate() {
        checkpoint(step)?;
        // Qwen Image's transformer consumes normalized scheduler time even
        // though FlowMatch exposes the conventional 0..1000 timestep range.
        let timestep = timestep / 1_000.0;
        let conditional = predict(latents, timestep, PredictionPass::Conditional)?;
        let prediction = if do_true_cfg {
            let unconditional = predict(latents, timestep, PredictionPass::Unconditional)?;
            true_cfg_rescale(
                &conditional,
                &unconditional,
                true_cfg_scale,
                packed_channels,
            )?
        } else {
            conditional
        };
        schedule.step(step, &prediction, latents)?;
        preview(step + 1, latents)?;
    }
    Ok(())
}

/// Apply the exact Qwen Image true-CFG combination and per-token conditional
/// norm rescale used by the pinned Diffusers pipeline.
fn true_cfg_rescale(
    conditional: &[f32],
    unconditional: &[f32],
    scale: f32,
    packed_channels: usize,
) -> Result<Vec<f32>, ImageError> {
    if conditional.is_empty()
        || conditional.len() != unconditional.len()
        || packed_channels == 0
        || !scale.is_finite()
        || scale < 0.0
    {
        return Err(ImageError::UnsupportedShape(
            "invalid true-CFG prediction tensors or scale".to_string(),
        ));
    }
    if conditional
        .iter()
        .chain(unconditional)
        .any(|value| !value.is_finite())
    {
        return Err(ImageError::Numerical {
            component: "true_cfg",
            step: 0,
        });
    }

    let mut combined = Vec::with_capacity(conditional.len());
    combined.extend(
        unconditional
            .iter()
            .zip(conditional)
            .map(|(negative, positive)| negative + scale * (positive - negative)),
    );
    if combined.len() % packed_channels != 0 {
        return Err(ImageError::UnsupportedShape(format!(
            "true-CFG prediction length {} is not divisible by {packed_channels}",
            combined.len()
        )));
    }
    for (positive, guided) in conditional
        .chunks_exact(packed_channels)
        .zip(combined.chunks_exact_mut(packed_channels))
    {
        let positive_norm = positive
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let guided_norm = guided.iter().map(|value| value * value).sum::<f32>().sqrt();
        if guided_norm == 0.0 {
            if positive_norm == 0.0 {
                guided.fill(0.0);
                continue;
            }
            return Err(ImageError::Numerical {
                component: "true_cfg",
                step: 0,
            });
        }
        let norm_scale = positive_norm / guided_norm;
        for value in guided {
            *value *= norm_scale;
        }
    }
    if combined.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "true_cfg",
            step: 0,
        });
    }
    Ok(combined)
}

pub(super) fn denormalize_vae_latents(
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
            "invalid VAE latent normalization geometry".to_string(),
        ));
    }
    let plane = latents.len() / means.len();
    for (channel, values) in latents.chunks_exact_mut(plane).enumerate() {
        for value in values {
            *value = *value * standard_deviations[channel] + means[channel];
        }
    }
    if latents.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "vae_latent_denormalization",
            step: 0,
        });
    }
    Ok(())
}

pub(super) fn decoded_ncthw_to_rgba8(
    decoded: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<DecodedImage, ImageError> {
    if channels != 3 || height == 0 || width == 0 {
        return Err(ImageError::UnsupportedShape(format!(
            "RGB output requires [3, 1, height, width], found [{channels}, 1, {height}, {width}]"
        )));
    }
    let plane = checked_product(&[height, width], "decoded image plane")?;
    let expected = checked_product(&[channels, plane], "decoded RGB image")?;
    if decoded.len() != expected || decoded.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::UnsupportedShape(format!(
            "decoded VAE output has {} values, expected {expected}",
            decoded.len()
        )));
    }
    let mut rgba = vec![0u8; checked_product(&[plane, 4], "RGBA output")?];
    for pixel in 0..plane {
        for channel in 0..3 {
            let normalized = (decoded[channel * plane + pixel] * 0.5 + 0.5).clamp(0.0, 1.0);
            rgba[pixel * 4 + channel] = (normalized * 255.0).round() as u8;
        }
        rgba[pixel * 4 + 3] = 255;
    }
    DecodedImage::new_rgba8(
        u32::try_from(width)
            .map_err(|_| ImageError::UnsupportedShape("output width exceeds u32".to_string()))?,
        u32::try_from(height)
            .map_err(|_| ImageError::UnsupportedShape("output height exceeds u32".to_string()))?,
        rgba,
    )
}

pub(super) fn checked_product(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
    })
}

pub(super) fn emit(
    progress: Option<&dyn ImageProgressSink>,
    output_index: usize,
    phase: ImageProgressPhase,
    step: Option<usize>,
    total_steps: Option<usize>,
) {
    if let Some(progress) = progress {
        progress.on_progress(&ImageProgressEvent {
            output_index,
            phase,
            step,
            total_steps,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::FlowMatchEulerConfig;

    fn scheduler() -> FlowMatchEulerSchedule {
        FlowMatchEulerSchedule::new(
            FlowMatchEulerConfig {
                num_train_timesteps: 1_000,
                shift: 1.0,
                use_dynamic_shifting: false,
                base_image_seq_len: 256,
                max_image_seq_len: 4_096,
                base_shift: 0.5,
                max_shift: 1.15,
                shift_terminal: None,
                time_shift_type: "exponential".to_string(),
                invert_sigmas: false,
                use_karras_sigmas: false,
                use_exponential_sigmas: false,
                use_beta_sigmas: false,
            },
            2,
            1,
        )
        .unwrap()
    }

    #[test]
    fn denoising_normalizes_time_and_runs_true_cfg_in_diffusers_order() {
        let mut latents = vec![1.0f32; 64];
        let mut calls = Vec::new();
        run_denoising(
            &mut latents,
            &scheduler(),
            2.0,
            true,
            64,
            |_, timestep, pass| {
                calls.push((timestep, pass));
                Ok(vec![
                    if pass == PredictionPass::Conditional {
                        2.0
                    } else {
                        0.0
                    };
                    64
                ])
            },
            |_| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (1.0, PredictionPass::Conditional));
        assert_eq!(calls[1], (1.0, PredictionPass::Unconditional));
        assert_eq!(calls[2], (0.5, PredictionPass::Conditional));
        assert_eq!(calls[3], (0.5, PredictionPass::Unconditional));
        assert!(latents.iter().all(|value| (*value + 1.0).abs() < 1e-6));
    }

    #[test]
    fn true_cfg_matches_negative_plus_scale_delta_then_conditional_norm() {
        let mut positive = vec![0.0f32; 64];
        let mut negative = vec![0.0f32; 64];
        positive[0] = 3.0;
        positive[1] = 4.0;
        negative[0] = 1.0;
        let guided = true_cfg_rescale(&positive, &negative, 2.0, 64).unwrap();
        let expected_scale = 5.0 / 89.0f32.sqrt();
        assert!((guided[0] - 5.0 * expected_scale).abs() < 1e-6);
        assert!((guided[1] - 8.0 * expected_scale).abs() < 1e-6);
    }

    #[test]
    fn latent_denormalization_matches_qwen_pipeline_formula() {
        let mut latents = vec![1.0, 2.0, -1.0, -2.0];
        denormalize_vae_latents(&mut latents, &[0.5, -0.5], &[2.0, 3.0]).unwrap();
        assert_eq!(latents, [2.5, 4.5, -3.5, -6.5]);
    }

    #[test]
    fn converts_channel_first_vae_output_to_opaque_rgba() {
        let decoded = [-1.0, 1.0, 0.0, 0.0, 1.0, -1.0];
        let image = decoded_ncthw_to_rgba8(&decoded, 3, 1, 2).unwrap();
        assert_eq!(image.rgba8(), &[0, 128, 255, 255, 255, 128, 0, 255]);
    }
}
