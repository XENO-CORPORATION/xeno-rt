use std::{collections::BTreeMap, sync::Arc, time::Instant};

use sha2::{Digest, Sha256};

use crate::{
    encode_image,
    pipeline::{ImagePipeline, ImageRequest, PipelineExecutionPlan},
    scheduler::{FlowMatchEulerConfig, FlowMatchEulerSchedule},
    BundleComponent, BundleFile, BundleLicense, BundleLimits, BundleManifest, ComponentFormat,
    ComponentRole, DecodedImage, ImageBackendKind, ImageBatchResult, ImageBatchTimings,
    ImageCancellation, ImageCapability, ImageEditRequest, ImageError, ImageExecutionPlan,
    ImageGenerationRequest, ImageModelBundle, ImageOffloadPolicy, ImagePreviewEvent,
    ImageProgressEvent, ImageProgressPhase, ImageProgressSink, ImageRequestKind,
    ImageRequestLimits, ImageResult, ImageTimings, NormalRngV1, PlannedImageOutput,
};

pub fn synthetic_bundle_for_tests() -> ImageModelBundle {
    let components = [
        ComponentRole::Transformer,
        ComponentRole::TextEncoder,
        ComponentRole::Tokenizer,
        ComponentRole::Processor,
        ComponentRole::Vae,
        ComponentRole::Scheduler,
    ]
    .into_iter()
    .map(|role| BundleComponent {
        format: ComponentFormat::Json,
        files: vec![BundleFile {
            path: format!("{}/fixture.bin", role.as_str()),
            size_bytes: 1,
            sha256: "00".repeat(32),
            source: None,
            source_kind: Some("local".to_string()),
        }],
        role,
        optional: false,
    })
    .collect();
    ImageModelBundle::synthetic(BundleManifest {
        schema_version: 1,
        id: "xrt-image-synthetic-v1".to_string(),
        family: "xrt-synthetic-image".to_string(),
        revision: "test-fixture-v1".to_string(),
        source_revisions: BTreeMap::new(),
        capabilities: vec![ImageCapability::Generate, ImageCapability::Edit],
        license: BundleLicense {
            spdx: "Apache-2.0".to_string(),
            evidence: "https://example.invalid/models/blob/test-fixture-v1/README.md".to_string(),
            files: Vec::new(),
        },
        quantization: "SYNTHETIC_F32".to_string(),
        components,
        limits: BundleLimits {
            max_sequence_length: 512,
            max_width: 256,
            max_height: 256,
            max_pixels: 65_536,
        },
    })
    .expect("synthetic bundle manifest is a checked-in invariant")
}

pub(crate) struct SyntheticPipeline {
    model: String,
    bundle_digest: String,
    quantization: String,
    capabilities: Vec<ImageCapability>,
    limits: ImageRequestLimits,
    backend: ImageBackendKind,
}

impl SyntheticPipeline {
    pub(crate) fn new(
        bundle: ImageModelBundle,
        backend: ImageBackendKind,
    ) -> Result<Self, ImageError> {
        if backend == ImageBackendKind::Cuda {
            return Err(ImageError::UnsupportedBackend(
                "the synthetic reference pipeline is CPU-only".to_string(),
            ));
        }
        let manifest = bundle.manifest();
        Ok(Self {
            model: manifest.id.clone(),
            bundle_digest: bundle.digest().to_string(),
            quantization: manifest.quantization.clone(),
            capabilities: manifest.capabilities.clone(),
            limits: ImageRequestLimits {
                max_prompt_bytes: manifest.limits.max_sequence_length.saturating_mul(16),
                max_outputs: 4,
                max_steps: 100,
                max_width: manifest.limits.max_width,
                max_height: manifest.limits.max_height,
                max_pixels: manifest.limits.max_pixels,
                dimension_multiple: 16,
                max_source_images: 3,
            },
            backend: ImageBackendKind::Cpu,
        })
    }

    fn plan_generation(
        &self,
        request: &ImageGenerationRequest,
        kind: ImageRequestKind,
    ) -> Result<ImageExecutionPlan, ImageError> {
        self.limits.validate_generation(request)?;
        if request.model != self.model {
            return Err(ImageError::InvalidRequest(format!(
                "request model `{}` does not match loaded model `{}`",
                request.model, self.model
            )));
        }
        if request.backend == ImageBackendKind::Cuda {
            return Err(ImageError::UnsupportedBackend(
                "the synthetic reference pipeline is CPU-only".to_string(),
            ));
        }
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
        let pixel_bytes = u64::from(request.width)
            .checked_mul(u64::from(request.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| {
                ImageError::Admission("synthetic memory estimate overflow".to_string())
            })?;
        let estimated_host_bytes = pixel_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_mul(request.n as u64))
            .ok_or_else(|| {
                ImageError::Admission("synthetic memory estimate overflow".to_string())
            })?;
        Ok(ImageExecutionPlan {
            request_kind: kind,
            model: request.model.clone(),
            bundle_digest: self.bundle_digest.clone(),
            backend: self.backend,
            offload: ImageOffloadPolicy::Cpu,
            width: request.width,
            height: request.height,
            steps: request.steps,
            outputs,
            estimated_host_bytes,
            estimated_device_bytes: 0,
        })
    }

    fn validate_edit(&self, request: &ImageEditRequest) -> Result<(), ImageError> {
        if !self.capabilities.contains(&ImageCapability::Edit) {
            return Err(ImageError::UnsupportedCapability("image.edit".to_string()));
        }
        if request.images.is_empty() || request.images.len() > self.limits.max_source_images {
            return Err(ImageError::InvalidRequest(format!(
                "image edit requires between 1 and {} ordered source images",
                self.limits.max_source_images
            )));
        }
        if request.mask.is_some() && !self.capabilities.contains(&ImageCapability::Inpaint) {
            return Err(ImageError::UnsupportedCapability(
                "mask requires image.inpaint; image.edit does not imply masks".to_string(),
            ));
        }
        if !request.strength.is_finite() || !(0.0..=1.0).contains(&request.strength) {
            return Err(ImageError::InvalidRequest(
                "edit strength must be finite in [0, 1]".to_string(),
            ));
        }
        if request.images.iter().any(|image| {
            image.width() != request.generation.width || image.height() != request.generation.height
        }) {
            return Err(ImageError::UnsupportedShape(
                "synthetic edit sources must match output dimensions".to_string(),
            ));
        }
        Ok(())
    }

    fn execute_one(
        &self,
        request: &ImageGenerationRequest,
        edit: Option<&ImageEditRequest>,
        output_index: usize,
        seed: u64,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageResult, ImageError> {
        let started = Instant::now();
        emit(
            progress,
            output_index,
            ImageProgressPhase::Admitted,
            None,
            None,
        );
        cancellation.check()?;

        let prompt_started = Instant::now();
        emit(
            progress,
            output_index,
            ImageProgressPhase::PromptEncoding,
            None,
            None,
        );
        let mut prompt_hash = Sha256::new();
        prompt_hash.update(request.prompt.as_bytes());
        prompt_hash.update([0]);
        if let Some(negative) = &request.negative_prompt {
            prompt_hash.update(negative.as_bytes());
        }
        let prompt_hash = prompt_hash.finalize();
        let prompt_bias = (u32::from_le_bytes(prompt_hash[..4].try_into().expect("four bytes"))
            as f32
            / u32::MAX as f32
            - 0.5)
            * 0.25;
        let prompt_encoding_ms = prompt_started.elapsed().as_secs_f64() * 1_000.0;
        cancellation.check()?;

        let count = usize::try_from(u64::from(request.width) * u64::from(request.height) * 4)
            .map_err(|_| ImageError::Admission("latent size exceeds usize".to_string()))?;
        let mut latent = vec![0.0f32; count];
        NormalRngV1::new(seed).fill_f32(&mut latent);

        let source_started = Instant::now();
        if let Some(edit) = edit {
            emit(
                progress,
                output_index,
                ImageProgressPhase::SourceEncoding,
                None,
                None,
            );
            let weight = edit.strength / edit.images.len() as f32;
            for source in &edit.images {
                for (value, byte) in latent.iter_mut().zip(source.rgba8()) {
                    let normalized = *byte as f32 / 127.5 - 1.0;
                    *value = *value * (1.0 - edit.strength) + normalized * weight;
                }
            }
        }
        let source_encoding_ms = source_started.elapsed().as_secs_f64() * 1_000.0;
        cancellation.check()?;

        let schedule = FlowMatchEulerSchedule::new(
            qwen_fixture_scheduler(),
            request.steps,
            usize::try_from((request.width / 16) * (request.height / 16))
                .map_err(|_| ImageError::Admission("sequence length overflow".to_string()))?,
        )?;
        let denoise_started = Instant::now();
        let mut prediction = vec![0.0f32; latent.len()];
        for step in 0..request.steps {
            cancellation.check()?;
            emit(
                progress,
                output_index,
                ImageProgressPhase::Denoising,
                Some(step),
                Some(request.steps),
            );
            let timestep_bias = schedule.timesteps()[step] / 1_000.0 * 0.03125;
            for (prediction, sample) in prediction.iter_mut().zip(&latent) {
                *prediction = *sample * 0.125 + prompt_bias + timestep_bias;
            }
            schedule.step(step, &prediction, &mut latent)?;
            let completed_steps = step + 1;
            if request.preview_interval.is_some_and(|interval| {
                completed_steps < request.steps && completed_steps % interval == 0
            }) && progress.is_some_and(ImageProgressSink::wants_previews)
            {
                let pixels = latent
                    .iter()
                    .map(|value| ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8)
                    .collect();
                let preview = DecodedImage::new_rgba8(request.width, request.height, pixels)?;
                let bytes = encode_image(&preview, request.output_format, 90, 64 * 1024 * 1024)?;
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
            }
        }
        let denoising_ms = denoise_started.elapsed().as_secs_f64() * 1_000.0;

        cancellation.check()?;
        let vae_started = Instant::now();
        emit(
            progress,
            output_index,
            ImageProgressPhase::VaeDecode,
            None,
            None,
        );
        let pixels = latent
            .into_iter()
            .map(|value| ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8)
            .collect();
        let image = DecodedImage::new_rgba8(request.width, request.height, pixels)?;
        let vae_decode_ms = vae_started.elapsed().as_secs_f64() * 1_000.0;

        cancellation.check()?;
        let encoding_started = Instant::now();
        emit(
            progress,
            output_index,
            ImageProgressPhase::Encoding,
            None,
            None,
        );
        let bytes = encode_image(&image, request.output_format, 90, 64 * 1024 * 1024)?;
        let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
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
                total_ms,
            },
        })
    }
}

impl ImagePipeline for SyntheticPipeline {
    fn capabilities(&self) -> &[ImageCapability] {
        &self.capabilities
    }

    fn backend(&self) -> ImageBackendKind {
        self.backend
    }

    fn plan(&self, request: &ImageRequest) -> Result<PipelineExecutionPlan, ImageError> {
        let public = match request {
            ImageRequest::Generation(request) => {
                if !self.capabilities.contains(&ImageCapability::Generate) {
                    return Err(ImageError::UnsupportedCapability(
                        "image.generate".to_string(),
                    ));
                }
                self.plan_generation(request, ImageRequestKind::Generate)?
            }
            ImageRequest::Edit(request) => {
                self.validate_edit(request)?;
                self.plan_generation(&request.generation, ImageRequestKind::Edit)?
            }
        };
        Ok(PipelineExecutionPlan {
            public,
            request: request.clone(),
        })
    }

    fn execute(
        &self,
        plan: PipelineExecutionPlan,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageBatchResult, ImageError> {
        let started = Instant::now();
        let (request, edit) = match &plan.request {
            ImageRequest::Generation(request) => (request, None),
            ImageRequest::Edit(edit) => (&edit.generation, Some(edit)),
        };
        let mut images = Vec::with_capacity(plan.public.outputs.len());
        for output in &plan.public.outputs {
            images.push(self.execute_one(
                request,
                edit,
                output.output_index,
                output.seed,
                cancellation,
                progress,
            )?);
        }
        let execution_ms = started.elapsed().as_secs_f64() * 1_000.0;
        Ok(ImageBatchResult {
            images,
            timings: ImageBatchTimings {
                admission_ms: 0.0,
                queue_ms: 0.0,
                execution_ms,
                total_ms: execution_ms,
            },
        })
    }
}

fn qwen_fixture_scheduler() -> FlowMatchEulerConfig {
    FlowMatchEulerConfig {
        num_train_timesteps: 1_000,
        shift: 1.0,
        use_dynamic_shifting: true,
        base_image_seq_len: 256,
        max_image_seq_len: 8_192,
        base_shift: 0.5,
        max_shift: 0.9,
        shift_terminal: Some(0.02),
        time_shift_type: "exponential".to_string(),
        invert_sigmas: false,
        use_karras_sigmas: false,
        use_exponential_sigmas: false,
        use_beta_sigmas: false,
    }
}

fn emit(
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
