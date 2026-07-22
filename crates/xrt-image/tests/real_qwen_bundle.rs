use std::collections::BTreeMap;
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use half::bf16;
use sha2::{Digest, Sha256};
use xrt_gguf::GgufFile;
use xrt_image::{
    decode_image,
    models::qwen_image::{
        load_vae_encoder_f32_weights, open_transformer_gguf, open_vae_safetensors,
        qwen_image_vae_encode_f32_with_control, validate_vae_safetensors, QwenImageBf16Transformer,
        QwenImageBundleConfig, QwenImageCpuTextEncoder, QwenImageCpuVisionEncoder,
        QwenImageEditProcessor, QwenImageGgufTransformer, QwenImagePromptTokenizer,
        QwenImageTextConfig, QwenImageTransformerConfig, QwenImageVaeConfig,
    },
    DecodedImage, ImageBackendKind, ImageCancellation, ImageCapability, ImageEditRequest,
    ImageError, ImageGenerationRequest, ImageIoLimits, ImageModelBundle, ImageOffloadPolicy,
    ImageProgressEvent, ImageProgressSink, ImageRequestKind, ImageResult, ImageRuntime,
};
use xrt_runtime::{GpuResourceConfig, GpuResourceManager};
use xrt_safetensors::{HfModelBundle, SafeTensorLayout, SafeTensorStore};

#[cfg(feature = "cuda")]
use xrt_image::models::qwen_image::{QwenImageCudaTransformer, QwenImagePromptEmbeddings};

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, CUDA, and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_edit_2511_q4_cuda_transformer_matches_cpu() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let bundle = ImageModelBundle::open(&root).unwrap();
    assert_eq!(bundle.manifest().quantization, "Q4_K_M");
    let config = QwenImageBundleConfig::load(&bundle).unwrap();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let cuda = QwenImageCudaTransformer::from_file(
        open_transformer_gguf(&bundle, &config.transformer).unwrap(),
        config.transformer.clone(),
        "Q4_K_M",
        resources,
    )
    .unwrap();
    let cpu = QwenImageGgufTransformer::from_file(
        open_transformer_gguf(&bundle, &config.transformer).unwrap(),
        config.transformer,
        "Q4_K_M",
    )
    .unwrap();
    let packed_latents = (0..128)
        .map(|index| ((index % 31) as f32 - 15.0) / 31.0)
        .collect::<Vec<_>>();
    let prompt = QwenImagePromptEmbeddings {
        embeddings: (0..3_584)
            .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
            .collect(),
        attention_mask: vec![1],
        retained_lengths: vec![1],
        batch_size: 1,
        sequence_length: 1,
        hidden_size: 3_584,
    };
    let shapes = [[1, 1, 1], [1, 1, 1]];
    let before = cuda.transfer_stats();
    let actual = cuda
        .forward_edit_with_control(&packed_latents, &prompt, &[1.0], &shapes, |_| Ok(()))
        .unwrap();
    let transfers = cuda.transfer_stats().saturating_sub(before);
    let expected = cpu
        .forward_edit_with_control(&packed_latents, &prompt, &[1.0], &shapes, |_| Ok(()))
        .unwrap();
    let (max_abs, normalized_rms, cosine) = numerical_metrics(&actual, &expected);
    assert!(max_abs < 1e-4, "max_abs={max_abs}");
    assert!(normalized_rms < 2e-5, "normalized_rms={normalized_rms}");
    assert!(cosine > 0.999_999_9, "cosine={cosine}");
    assert_eq!(transfers.host_to_device_calls, 10);
    assert_eq!(transfers.device_to_host_calls, 1);
    assert_eq!(transfers.device_to_host_bytes, 128 * 4);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, CUDA, and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_edit_2511_q4_cuda_loads_and_plans() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let runtime =
        ImageRuntime::load(bundle, ImageBackendKind::Cuda, Arc::clone(&resources)).unwrap();
    assert_eq!(runtime.backend(), ImageBackendKind::Cuda);
    let plan = runtime
        .plan_edit(&pinned_edit_request_with_backend(
            model,
            ImageBackendKind::Cuda,
            ImageOffloadPolicy::Sequential,
        ))
        .unwrap();
    assert_eq!(plan.backend, ImageBackendKind::Cuda);
    assert_eq!(plan.offload, ImageOffloadPolicy::Sequential);
    let resident = resources
        .allocation_arena()
        .snapshot()
        .by_class
        .image_component_weight_bytes;
    assert!(resident > 13_244_758_624);
    assert!(plan.estimated_device_bytes > resident);
    eprintln!(
        "Qwen Image Edit CUDA plan: resident={resident} estimated_device={} arena={:?}",
        plan.estimated_device_bytes,
        resources.allocation_arena().snapshot()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, CUDA, and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_edit_2511_q4_cuda_edit_smoke() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let runtime =
        ImageRuntime::load(bundle, ImageBackendKind::Cuda, Arc::clone(&resources)).unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let request = pinned_edit_request_with_backend(
        model,
        ImageBackendKind::Cuda,
        ImageOffloadPolicy::Sequential,
    );
    let plan = runtime.plan_edit(&request).unwrap();
    let execution_started = Instant::now();
    let result = runtime
        .edit(
            request,
            cancellation_with_optional_deadline(
                "XRT_QWEN_IMAGE_EDIT_CUDA_SMOKE_TIMEOUT_SECONDS",
                1_800,
            ),
            Some(elapsed_progress_sink(execution_started)),
        )
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = &result.images[0];
    assert_eq!([image.width, image.height], [16, 16]);
    assert_eq!(image.backend, ImageBackendKind::Cuda);
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    assert_eq!([decoded.width(), decoded.height()], [16, 16]);
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    assert_eq!(
        png_sha256,
        "a7210827c5a229ff94b1b0c15752eec65bd937d24fe31a0dbaa77bd7ccb3230f"
    );
    eprintln!(
        "Qwen Image Edit native CUDA smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s png_sha256={png_sha256} plan={plan:?} arena={:?} timings={:?}",
        resources.allocation_arena().snapshot(),
        image.timings
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, CUDA, and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_edit_2511_q4_cuda_two_image_edit_smoke() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let runtime =
        ImageRuntime::load(bundle, ImageBackendKind::Cuda, Arc::clone(&resources)).unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let request = pinned_two_image_edit_request_with_backend(
        model,
        ImageBackendKind::Cuda,
        ImageOffloadPolicy::Sequential,
    );
    let plan = runtime.plan_edit(&request).unwrap();
    let execution_started = Instant::now();
    let result = runtime
        .edit(
            request,
            cancellation_with_optional_deadline(
                "XRT_QWEN_IMAGE_EDIT_CUDA_MULTI_IMAGE_SMOKE_TIMEOUT_SECONDS",
                1_800,
            ),
            Some(elapsed_progress_sink(execution_started)),
        )
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = &result.images[0];
    assert_eq!([image.width, image.height], [16, 16]);
    assert_eq!(image.backend, ImageBackendKind::Cuda);
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    assert_eq!([decoded.width(), decoded.height()], [16, 16]);
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    eprintln!(
        "Qwen Image Edit two-image CUDA smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s png_sha256={png_sha256} plan={plan:?} arena={:?} timings={:?}",
        resources.allocation_arena().snapshot(),
        image.timings
    );
    assert_eq!(
        png_sha256,
        "5dde8efa3c6f2c3dc6a159956082e5677a88d4e1307e279f653fc6bdf822e7d3"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, CUDA, and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_edit_2511_q4_cuda_three_image_edit_smoke() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let runtime =
        ImageRuntime::load(bundle, ImageBackendKind::Cuda, Arc::clone(&resources)).unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let request = pinned_three_image_edit_request_with_backend(
        model,
        ImageBackendKind::Cuda,
        ImageOffloadPolicy::Sequential,
    );
    let plan = runtime.plan_edit(&request).unwrap();
    let execution_started = Instant::now();
    let result = runtime
        .edit(
            request,
            cancellation_with_optional_deadline(
                "XRT_QWEN_IMAGE_EDIT_CUDA_THREE_IMAGE_SMOKE_TIMEOUT_SECONDS",
                3_600,
            ),
            Some(elapsed_progress_sink(execution_started)),
        )
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = &result.images[0];
    assert_eq!([image.width, image.height], [16, 16]);
    assert_eq!(image.backend, ImageBackendKind::Cuda);
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    assert_eq!([decoded.width(), decoded.height()], [16, 16]);
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    eprintln!(
        "Qwen Image Edit three-image CUDA smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s png_sha256={png_sha256} plan={plan:?} arena={:?} timings={:?}",
        resources.allocation_arena().snapshot(),
        image.timings
    );
    assert_eq!(
        png_sha256,
        "b592ba1d170944f7ca9b41979c1df7d44f0bad4a8a6ee02c1e5d640d128ab31f"
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR and a complete pinned Edit-2511 bundle"]
fn pinned_qwen_image_edit_2511_cpu_loads_and_plans() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let runtime = ImageRuntime::load(
        bundle,
        ImageBackendKind::Cpu,
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap();
    assert_eq!(runtime.capabilities(), [ImageCapability::Edit]);
    assert_eq!(runtime.backend(), ImageBackendKind::Cpu);

    let request = pinned_edit_request(model);
    let plan = runtime.plan_edit(&request).unwrap();
    assert_eq!(plan.request_kind, ImageRequestKind::Edit);
    assert_eq!(plan.backend, ImageBackendKind::Cpu);
    assert_eq!(plan.estimated_device_bytes, 0);
    assert!(plan.estimated_host_bytes > 0);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_edit_2511_cpu_edit_smoke() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let runtime = ImageRuntime::load(
        bundle,
        ImageBackendKind::Cpu,
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();

    let execution_started = Instant::now();
    let progress = elapsed_progress_sink(execution_started);
    let result = runtime
        .edit(
            pinned_edit_request(model),
            cancellation_with_optional_deadline(
                "XRT_QWEN_IMAGE_EDIT_CPU_SMOKE_TIMEOUT_SECONDS",
                7_200,
            ),
            Some(progress),
        )
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = &result.images[0];
    assert_eq!([image.width, image.height], [16, 16]);
    assert_eq!(image.backend, ImageBackendKind::Cpu);
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    assert_eq!([decoded.width(), decoded.height()], [16, 16]);
    eprintln!(
        "qwen-image-edit native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s png_sha256={:x} timings={:?}",
        Sha256::digest(&image.bytes),
        image.timings
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR and performs a bounded real CPU phase probe"]
fn pinned_qwen_image_edit_2511_cpu_phase_probe() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let seconds = std::env::var("XRT_QWEN_IMAGE_EDIT_PROBE_SECONDS")
        .ok()
        .map(|value| value.parse::<u64>().expect("probe seconds must be u64"))
        .unwrap_or(120);
    assert!((1..=600).contains(&seconds));
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let runtime = ImageRuntime::load(
        bundle,
        ImageBackendKind::Cpu,
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap();

    let cancellation = ImageCancellation::new();
    let deadline = cancellation.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(seconds));
        deadline.cancel();
    });
    let started = Instant::now();
    let result = runtime.edit(
        pinned_edit_request(model),
        cancellation,
        Some(elapsed_progress_sink(started)),
    );
    match result {
        Err(ImageError::Cancelled) => {
            eprintln!("Qwen Image Edit CPU phase probe cancelled cleanly after {seconds}s")
        }
        Ok(result) => eprintln!(
            "Qwen Image Edit CPU phase probe completed before cancellation: {:?}",
            result.timings
        ),
        Err(error) => panic!("Qwen Image Edit CPU phase probe failed: {error}"),
    }
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR and probes the real CPU vision tower"]
fn pinned_qwen_image_edit_2511_cpu_vision_phase_probe() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let seconds = std::env::var("XRT_QWEN_IMAGE_EDIT_PROBE_SECONDS")
        .ok()
        .map(|value| value.parse::<u64>().expect("probe seconds must be u64"))
        .unwrap_or(120);
    assert!((1..=600).contains(&seconds));
    let bundle = ImageModelBundle::open(root).unwrap();
    let config = QwenImageBundleConfig::load(&bundle).unwrap();
    let processor = QwenImageEditProcessor::load_with_config(&bundle, &config).unwrap();
    let vision_encoder = QwenImageCpuVisionEncoder::load_with_config(&bundle, &config).unwrap();
    let request = pinned_edit_request(bundle.manifest().id.clone());
    let processed = processor.process_images(&request.images).unwrap();
    eprintln!(
        "Qwen Image Edit source geometry: condition={:?} vae={:?} grids={:?}",
        processed.condition_sizes, processed.vae_sizes, processed.vision.grids
    );

    let started = Instant::now();
    let result = vision_encoder.encode_with_control(&processed.vision, |stage| {
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!("Qwen Image Edit vision stage={stage} elapsed={elapsed:.3}s");
        if elapsed >= seconds as f64 {
            Err(ImageError::Cancelled)
        } else {
            Ok(())
        }
    });
    match result {
        Err(ImageError::Cancelled) => eprintln!(
            "Qwen Image Edit CPU vision probe cancelled cleanly after reaching the {seconds}s bound"
        ),
        Ok(embeddings) => eprintln!(
            "Qwen Image Edit CPU vision probe completed in {:.3}s with {} image tokens",
            started.elapsed().as_secs_f64(),
            embeddings.image_token_counts.iter().sum::<usize>()
        ),
        Err(error) => panic!("Qwen Image Edit CPU vision probe failed: {error}"),
    }
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR and probes the real 1024px CPU VAE encoder"]
fn pinned_qwen_image_edit_2511_cpu_vae_phase_probe() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR must point to the installed bundle directory");
    let seconds = std::env::var("XRT_QWEN_IMAGE_EDIT_PROBE_SECONDS")
        .ok()
        .map(|value| value.parse::<u64>().expect("probe seconds must be u64"))
        .unwrap_or(120);
    assert!((1..=600).contains(&seconds));
    let bundle = ImageModelBundle::open(root).unwrap();
    let config = QwenImageBundleConfig::load(&bundle).unwrap();
    let processor = QwenImageEditProcessor::load_with_config(&bundle, &config).unwrap();
    let request = pinned_edit_request(bundle.manifest().id.clone());
    let mut processed = processor.process_images(&request.images).unwrap();
    let source = processed.vae_sources.remove(0);
    eprintln!(
        "Qwen Image Edit VAE source geometry: {}x{} values={}",
        source.width,
        source.height,
        source.values.len()
    );
    let store = open_vae_safetensors(&bundle, &config.vae).unwrap();
    let weights = load_vae_encoder_f32_weights(&store, &config.vae).unwrap();

    let started = Instant::now();
    let result = qwen_image_vae_encode_f32_with_control(
        &config.vae,
        &weights,
        &source.values,
        1,
        1,
        source.height as usize,
        source.width as usize,
        |stage| {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!("Qwen Image Edit VAE stage={stage} elapsed={elapsed:.3}s");
            if elapsed >= seconds as f64 {
                Err(ImageError::Cancelled)
            } else {
                Ok(())
            }
        },
    );
    match result {
        Err(ImageError::Cancelled) => eprintln!(
            "Qwen Image Edit CPU VAE probe cancelled cleanly after reaching the {seconds}s bound"
        ),
        Ok(latents) => eprintln!(
            "Qwen Image Edit CPU VAE probe completed in {:.3}s with {} latent values",
            started.elapsed().as_secs_f64(),
            latents.len()
        ),
        Err(error) => panic!("Qwen Image Edit CPU VAE probe failed: {error}"),
    }
}

fn elapsed_progress_sink(started: Instant) -> Arc<dyn ImageProgressSink> {
    Arc::new(move |event: &ImageProgressEvent| {
        eprintln!(
            "Qwen Image Edit phase={:?} step={:?}/{:?} elapsed={:.3}s",
            event.phase,
            event.step,
            event.total_steps,
            started.elapsed().as_secs_f64()
        );
    })
}

fn cancellation_with_optional_deadline(variable: &str, max_seconds: u64) -> ImageCancellation {
    let cancellation = ImageCancellation::new();
    if let Some(seconds) = std::env::var(variable)
        .ok()
        .map(|value| value.parse::<u64>().expect("smoke timeout must be u64"))
    {
        assert!((1..=max_seconds).contains(&seconds));
        let deadline = cancellation.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(seconds));
            deadline.cancel();
        });
    }
    cancellation
}

fn pinned_edit_request(model: String) -> ImageEditRequest {
    pinned_edit_request_with_backend(model, ImageBackendKind::Cpu, ImageOffloadPolicy::Cpu)
}

fn pinned_edit_request_with_backend(
    model: String,
    backend: ImageBackendKind,
    offload: ImageOffloadPolicy,
) -> ImageEditRequest {
    let rgba = (0..16 * 16)
        .flat_map(|index| {
            let x = (index % 16) as u8;
            let y = (index / 16) as u8;
            [x.saturating_mul(16), y.saturating_mul(16), 64, 255]
        })
        .collect();
    ImageEditRequest {
        generation: ImageGenerationRequest {
            model,
            prompt: "change the red-orange square to cobalt blue".to_string(),
            width: 16,
            height: 16,
            steps: 2,
            true_cfg_scale: 1.0,
            backend,
            offload,
            ..ImageGenerationRequest::default()
        },
        images: vec![DecodedImage::new_rgba8(16, 16, rgba).unwrap()],
        mask: None,
        strength: 1.0,
    }
}

#[cfg(feature = "cuda")]
fn pinned_two_image_edit_request_with_backend(
    model: String,
    backend: ImageBackendKind,
    offload: ImageOffloadPolicy,
) -> ImageEditRequest {
    let mut request = pinned_edit_request_with_backend(model, backend, offload);
    request.generation.prompt =
        "combine the first red-orange gradient and second green-blue gradient into a cobalt composition"
            .to_string();
    let second_rgba = (0..16 * 16)
        .flat_map(|index| {
            let x = (index % 16) as u8;
            let y = (index / 16) as u8;
            [32, y.saturating_mul(16), x.saturating_mul(16), 255]
        })
        .collect();
    request
        .images
        .push(DecodedImage::new_rgba8(16, 16, second_rgba).unwrap());
    request
}

#[cfg(feature = "cuda")]
fn pinned_three_image_edit_request_with_backend(
    model: String,
    backend: ImageBackendKind,
    offload: ImageOffloadPolicy,
) -> ImageEditRequest {
    let mut request = pinned_two_image_edit_request_with_backend(model, backend, offload);
    request.generation.prompt = "combine the first red-orange, second green-blue, and third magenta-green gradients into a cobalt composition".to_string();
    let third_rgba = (0..16 * 16)
        .flat_map(|index| {
            let x = (index % 16) as u8;
            let y = (index / 16) as u8;
            [x.saturating_mul(16), 32, y.saturating_mul(16), 255]
        })
        .collect();
    request
        .images
        .push(DecodedImage::new_rgba8(16, 16, third_rgba).unwrap());
    request
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the complete pinned Q4_K_M bundle"]
fn pinned_qwen_image_2512_q4_bundle_validates_end_to_end() {
    validate_pinned_generation_bundle("Q4_K_M");
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the complete pinned Q8_0 bundle"]
fn pinned_qwen_image_2512_q8_bundle_validates_end_to_end() {
    validate_pinned_generation_bundle("Q8_0");
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the complete pinned Q6_K bundle"]
fn pinned_qwen_image_2512_q6_bundle_validates_end_to_end() {
    validate_pinned_generation_bundle("Q6_K");
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the complete pinned Q5_K_M bundle"]
fn pinned_qwen_image_2512_q5_bundle_validates_end_to_end() {
    validate_pinned_generation_bundle("Q5_K_M");
}

fn validate_pinned_generation_bundle(expected_quantization: &str) {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let bundle = ImageModelBundle::open(root).unwrap();
    assert_eq!(bundle.manifest().quantization, expected_quantization);
    let config = QwenImageBundleConfig::load(&bundle).unwrap();
    let tokenizer = QwenImagePromptTokenizer::load(
        &bundle,
        config.max_sequence_length,
        config.text_encoder.vocab_size,
    )
    .unwrap();
    assert_eq!(tokenizer.pad_token_id(), 151_643);
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/qwen-image/tokenizer-2512.json");
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_path).unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let expected_ids = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let expected_mask = case["attention_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let actual = tokenizer.encode_batch(&[prompt]).unwrap();
        assert_eq!(actual.input_ids[0], expected_ids);
        assert_eq!(actual.attention_mask[0], expected_mask);
        assert_eq!(actual.retained_lengths[0], expected_ids.len() - 34);
    }
    let padded = tokenizer
        .encode_batch(&["", "A red cube on a blue table."])
        .unwrap();
    assert_eq!(padded.retained_lengths, vec![5, 13]);
    assert!(padded.attention_mask[0][39..]
        .iter()
        .all(|value| *value == 0));
    let transformer = open_transformer_gguf(&bundle, &config.transformer).unwrap();
    assert_eq!(transformer.tensor_infos().len(), 1_933);
    assert_eq!(
        transformer.metadata_string("general.architecture"),
        Some("qwen_image")
    );
    let vae = open_vae_safetensors(&bundle, &config.vae).unwrap();
    assert_eq!(vae.tensor_count(), 194);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned VAE SafeTensors file"]
fn pinned_qwen_image_2512_vae_file_validates_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config =
        QwenImageVaeConfig::from_json_bytes(&fs::read(root.join("vae/config.json")).unwrap())
            .unwrap();
    let store = SafeTensorStore::open_exact(
        &root,
        SafeTensorLayout::single("vae/diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
    validate_vae_safetensors(&store, &config).unwrap();
    assert_eq!(store.tensor_count(), 194);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned text encoder shards"]
fn pinned_qwen_image_2512_cpu_text_component_validates_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let component = root.join("text_encoder");
    let config =
        QwenImageTextConfig::from_json_bytes(&fs::read(component.join("config.json")).unwrap())
            .unwrap();
    let model = HfModelBundle::open_exact(
        &component,
        SafeTensorLayout::indexed(
            "model.safetensors.index.json",
            [
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ],
        ),
    )
    .unwrap();
    let encoder = QwenImageCpuTextEncoder::from_model_bundle(model, config, 1024).unwrap();
    assert_eq!(encoder.hidden_size(), 3584);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned transformer GGUF"]
fn pinned_qwen_image_2512_q4_executor_loads_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let executor = QwenImageGgufTransformer::from_file(file, config, "Q4_K_M").unwrap();
    assert_eq!(executor.gguf().tensor_infos().len(), 1_933);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q8_0 transformer GGUF"]
fn pinned_qwen_image_2512_q8_executor_loads_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q8_0.gguf")).unwrap();
    let executor = QwenImageGgufTransformer::from_file(file, config, "Q8_0").unwrap();
    assert_eq!(executor.gguf().tensor_infos().len(), 1_933);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q6_K transformer GGUF"]
fn pinned_qwen_image_2512_q6_executor_loads_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q6_K.gguf")).unwrap();
    let executor = QwenImageGgufTransformer::from_file(file, config, "Q6_K").unwrap();
    assert_eq!(executor.gguf().tensor_infos().len(), 1_933);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q5_K_M transformer GGUF"]
fn pinned_qwen_image_2512_q5_executor_loads_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q5_K_M.gguf")).unwrap();
    let executor = QwenImageGgufTransformer::from_file(file, config, "Q5_K_M").unwrap();
    assert_eq!(executor.gguf().tensor_infos().len(), 1_933);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned transformer GGUF"]
fn pinned_qwen_image_2512_q4_reports_mixed_tensor_dtypes() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let mut all = BTreeMap::<String, usize>::new();
    let mut matrices = BTreeMap::<String, usize>::new();
    let mut exceptional_matrices = Vec::new();
    for info in file.tensor_infos() {
        *all.entry(format!("{:?}", info.dtype)).or_default() += 1;
        if info.dimensions.len() == 2 {
            *matrices.entry(format!("{:?}", info.dtype)).or_default() += 1;
            if matches!(info.dtype, xrt_core::DType::BF16 | xrt_core::DType::Q8_0) {
                exceptional_matrices.push((
                    info.name.clone(),
                    format!("{:?}", info.dtype),
                    info.dimensions.clone(),
                ));
            }
        }
    }
    eprintln!(
        "qwen-image Q4_K_M dtype distribution: all={all:?} matrices={matrices:?} exceptional={exceptional_matrices:?}"
    );
    assert_eq!(
        all,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("F32".to_string(), 1_087),
            ("Q4_K".to_string(), 560),
            ("Q5_K".to_string(), 20),
            ("Q6_K".to_string(), 258),
            ("Q8_0".to_string(), 2),
        ])
    );
    assert_eq!(
        matrices,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("Q4_K".to_string(), 560),
            ("Q5_K".to_string(), 20),
            ("Q6_K".to_string(), 258),
            ("Q8_0".to_string(), 2),
        ])
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q8_0 transformer GGUF"]
fn pinned_qwen_image_2512_q8_reports_mixed_tensor_dtypes() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q8_0.gguf")).unwrap();
    let mut all = BTreeMap::<String, usize>::new();
    let mut matrices = BTreeMap::<String, usize>::new();
    for info in file.tensor_infos() {
        *all.entry(format!("{:?}", info.dtype)).or_default() += 1;
        if info.dimensions.len() == 2 {
            *matrices.entry(format!("{:?}", info.dtype)).or_default() += 1;
        }
    }
    eprintln!("qwen-image Q8_0 dtype distribution: all={all:?} matrices={matrices:?}");
    assert_eq!(
        all,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("F32".to_string(), 1_087),
            ("Q8_0".to_string(), 840),
        ])
    );
    assert_eq!(
        matrices,
        BTreeMap::from([("BF16".to_string(), 6), ("Q8_0".to_string(), 840)])
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q6_K transformer GGUF"]
fn pinned_qwen_image_2512_q6_reports_mixed_tensor_dtypes() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q6_K.gguf")).unwrap();
    let mut all = BTreeMap::<String, usize>::new();
    let mut matrices = BTreeMap::<String, usize>::new();
    for info in file.tensor_infos() {
        *all.entry(format!("{:?}", info.dtype)).or_default() += 1;
        if info.dimensions.len() == 2 {
            *matrices.entry(format!("{:?}", info.dtype)).or_default() += 1;
        }
    }
    eprintln!("qwen-image Q6_K dtype distribution: all={all:?} matrices={matrices:?}");
    assert_eq!(
        all,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("F32".to_string(), 1_087),
            ("Q6_K".to_string(), 840),
        ])
    );
    assert_eq!(
        matrices,
        BTreeMap::from([("BF16".to_string(), 6), ("Q6_K".to_string(), 840)])
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned Q5_K_M transformer GGUF"]
fn pinned_qwen_image_2512_q5_reports_mixed_tensor_dtypes() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q5_K_M.gguf")).unwrap();
    let mut all = BTreeMap::<String, usize>::new();
    let mut matrices = BTreeMap::<String, usize>::new();
    for info in file.tensor_infos() {
        *all.entry(format!("{:?}", info.dtype)).or_default() += 1;
        if info.dimensions.len() == 2 {
            *matrices.entry(format!("{:?}", info.dtype)).or_default() += 1;
        }
    }
    eprintln!("qwen-image Q5_K_M dtype distribution: all={all:?} matrices={matrices:?}");
    assert_eq!(
        all,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("F32".to_string(), 1_087),
            ("Q5_K".to_string(), 560),
            ("Q6_K".to_string(), 280),
        ])
    );
    assert_eq!(
        matrices,
        BTreeMap::from([
            ("BF16".to_string(), 6),
            ("Q5_K".to_string(), 560),
            ("Q6_K".to_string(), 280),
        ])
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR, CUDA, and enough VRAM for the pinned transformer"]
fn pinned_qwen_image_2512_q4_cuda_transformer_loads_resident_mixed_weights() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let started = Instant::now();
    let transformer =
        QwenImageCudaTransformer::from_file(file, config, "Q4_K_M", Arc::clone(&resources))
            .unwrap();
    assert!(transformer.weight_bytes() > 10 * 1024 * 1024 * 1024);
    let snapshot = resources.allocation_arena().snapshot();
    assert_eq!(
        snapshot.by_class.image_component_weight_bytes,
        transformer.weight_bytes()
    );
    eprintln!(
        "qwen-image CUDA resident load: elapsed={:.3}s weights={} bytes transfers={:?} arena={snapshot:?}",
        started.elapsed().as_secs_f64(),
        transformer.weight_bytes(),
        transformer.transfer_stats()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR, XRT_CPU_FLOAT_ACTIVATION_REFERENCE=1, CUDA, and enough VRAM"]
fn pinned_qwen_image_2512_q4_cuda_transformer_forward_is_resident_and_deterministic() {
    assert!(
        std::env::var("XRT_CPU_FLOAT_ACTIVATION_REFERENCE")
            .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True")),
        "set XRT_CPU_FLOAT_ACTIVATION_REFERENCE=1 so the CPU oracle consumes F32 activations like CUDA"
    );
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let transformer =
        QwenImageCudaTransformer::from_file(file, config, "Q4_K_M", resources).unwrap();
    let packed_latents = (0..64)
        .map(|index| (index as f32 - 31.5) / 64.0)
        .collect::<Vec<_>>();
    let prompt = QwenImagePromptEmbeddings {
        embeddings: (0..3_584)
            .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
            .collect(),
        attention_mask: vec![1],
        retained_lengths: vec![1],
        batch_size: 1,
        sequence_length: 1,
        hidden_size: 3_584,
    };

    let before = transformer.transfer_stats();
    let started = Instant::now();
    let first = transformer
        .forward(&packed_latents, &prompt, &[1.0], 1, 1, 1)
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    let after_first = transformer.transfer_stats();
    let first_transfers = after_first.saturating_sub(before);
    let second = transformer
        .forward(&packed_latents, &prompt, &[1.0], 1, 1, 1)
        .unwrap();
    let second_transfers = transformer.transfer_stats().saturating_sub(after_first);

    assert_eq!(first.len(), 64);
    assert!(first.iter().all(|value| value.is_finite()));
    assert_eq!(first, second);
    for transfers in [first_transfers, second_transfers] {
        assert_eq!(transfers.host_to_device_calls, 8);
        assert_eq!(transfers.device_to_host_calls, 1);
        assert_eq!(transfers.device_to_host_bytes, 64 * 4);
    }
    let encoded = first
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let cuda_sha256 = format!("{:x}", Sha256::digest(&encoded));
    assert_eq!(
        cuda_sha256,
        "c1f495b6603f299926afaaa8e380392317622fa2dcea28c2516029a42ef02372"
    );

    let cpu_file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let cpu_config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let cpu_transformer =
        QwenImageGgufTransformer::from_file(cpu_file, cpu_config, "Q4_K_M").unwrap();
    let cpu_started = Instant::now();
    let cpu = cpu_transformer
        .forward(&packed_latents, &prompt, &[1.0], 1, 1, 1)
        .unwrap();
    let cpu_elapsed = cpu_started.elapsed().as_secs_f64();
    let max_abs = first
        .iter()
        .zip(&cpu)
        .map(|(cuda, cpu)| (cuda - cpu).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = first
        .iter()
        .zip(&cpu)
        .map(|(cuda, cpu)| f64::from((cuda - cpu).abs()))
        .sum::<f64>()
        / first.len() as f64;
    let dot = first
        .iter()
        .zip(&cpu)
        .map(|(cuda, cpu)| f64::from(*cuda) * f64::from(*cpu))
        .sum::<f64>();
    let cuda_norm = first
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let cpu_norm = cpu
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (cuda_norm * cpu_norm);
    eprintln!(
        "qwen-image CUDA full transformer forward: cuda={elapsed:.3}s cpu={cpu_elapsed:.3}s sha256={cuda_sha256} max_abs={max_abs:.8} mean_abs={mean_abs:.8} cosine={cosine:.10} transfers={first_transfers:?}"
    );
    assert!(max_abs < 1e-4, "max_abs={max_abs}");
    assert!(mean_abs < 2e-5, "mean_abs={mean_abs}");
    assert!(cosine > 0.999_999_9, "cosine={cosine}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR, CUDA, and enough VRAM; performance evidence only"]
fn pinned_qwen_image_2512_q4_cuda_512_transformer_benchmark() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(root.join("transformer/config.json")).unwrap(),
    )
    .unwrap();
    let file = GgufFile::open(root.join("transformer/qwen-image-2512-Q4_K_M.gguf")).unwrap();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let transformer =
        QwenImageCudaTransformer::from_file(file, config, "Q4_K_M", Arc::clone(&resources))
            .unwrap();
    let image_sequence = 32 * 32;
    let packed_latents = (0..image_sequence * 64)
        .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
        .collect::<Vec<_>>();
    let prompt = QwenImagePromptEmbeddings {
        embeddings: (0..3_584)
            .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
            .collect(),
        attention_mask: vec![1],
        retained_lengths: vec![1],
        batch_size: 1,
        sequence_length: 1,
        hidden_size: 3_584,
    };
    let before = transformer.transfer_stats();
    let started = Instant::now();
    let output = transformer
        .forward(&packed_latents, &prompt, &[1.0], 1, 32, 32)
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    let transfers = transformer.transfer_stats().saturating_sub(before);
    assert_eq!(output.len(), image_sequence * 64);
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(transfers.host_to_device_calls, 8);
    assert_eq!(transfers.device_to_host_calls, 1);
    eprintln!(
        "qwen-image CUDA 512-equivalent transformer benchmark: elapsed={elapsed:.3}s steps_per_second={:.6} transfers={transfers:?} arena={:?}",
        elapsed.recip(),
        resources.allocation_arena().snapshot()
    );
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the pinned BF16 transformer shards"]
fn pinned_qwen_image_2512_bf16_executor_loads_without_full_bundle_rehash() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let component = root.join("transformer");
    let config = QwenImageTransformerConfig::from_json_bytes(
        &fs::read(component.join("config.json")).unwrap(),
    )
    .unwrap();
    let store = SafeTensorStore::open_exact(
        &component,
        SafeTensorLayout::indexed(
            "diffusion_pytorch_model.safetensors.index.json",
            (1..=9).map(|shard| format!("diffusion_pytorch_model-{shard:05}-of-00009.safetensors")),
        ),
    )
    .unwrap();
    let executor = QwenImageBf16Transformer::from_store(store, config).unwrap();
    assert_eq!(executor.tensor_store().tensor_count(), 1_933);
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_2512_q4_cpu_generation_smoke() {
    let (image, load_seconds, execution_seconds) = run_pinned_cpu_generation("Q4_K_M");
    let pixel_container_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        pixel_container_sha256,
        "9f9792969f9c288946648145d19dac5656f1866b9a85958cb1e79d536e908490"
    );
    eprintln!(
        "qwen-image native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={pixel_container_sha256} timings={:?}",
        image.timings
    );
}

#[test]
#[ignore = "requires the complete pinned Q8_0 bundle, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_2512_q8_cpu_generation_smoke() {
    let (image, load_seconds, execution_seconds) = run_pinned_cpu_generation("Q8_0");
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        png_sha256,
        "f22e1fb7cbb1c61ee598fff09bad1d779bcd0898daa5e4cd687d1c50bde7d8d5"
    );
    eprintln!(
        "qwen-image Q8_0 native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={png_sha256} timings={:?}",
        image.timings
    );
}

#[test]
#[ignore = "requires the complete pinned Q6_K bundle, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_2512_q6_cpu_generation_smoke() {
    let (image, load_seconds, execution_seconds) = run_pinned_cpu_generation("Q6_K");
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        png_sha256,
        "be525556997f75fa50dd4e3d0aa46c15265bc610eec070e744db771492015e13"
    );
    eprintln!(
        "qwen-image Q6_K native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={png_sha256} timings={:?}",
        image.timings
    );
}

#[test]
#[ignore = "requires the complete pinned Q5_K_M bundle, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_2512_q5_cpu_generation_smoke() {
    let (image, load_seconds, execution_seconds) = run_pinned_cpu_generation("Q5_K_M");
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        png_sha256,
        "331c6edd6005f0f03e46b6d64ff1c166b4ed22f8fa531f2ca39812ae8dca5e4e"
    );
    eprintln!(
        "qwen-image Q5_K_M native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={png_sha256} timings={:?}",
        image.timings
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires the complete pinned Q4_K_M bundle, CUDA, and enough host/device memory"]
fn pinned_qwen_image_2512_q4_cuda_generation_smoke() {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let runtime =
        ImageRuntime::load(bundle, ImageBackendKind::Cuda, Arc::clone(&resources)).unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let request = ImageGenerationRequest {
        model,
        prompt: "a".to_string(),
        width: 16,
        height: 16,
        steps: 2,
        true_cfg_scale: 1.0,
        backend: ImageBackendKind::Cuda,
        offload: ImageOffloadPolicy::Sequential,
        ..ImageGenerationRequest::default()
    };
    let plan = runtime.plan_generation(&request).unwrap();
    assert_eq!(plan.backend, ImageBackendKind::Cuda);
    assert_eq!(plan.offload, ImageOffloadPolicy::Sequential);
    let resident_weight_bytes = resources
        .allocation_arena()
        .snapshot()
        .by_class
        .image_component_weight_bytes;
    assert_eq!(resident_weight_bytes, 13_649_426_688);
    assert!(plan.estimated_device_bytes > resident_weight_bytes);

    let execution_started = Instant::now();
    let result = runtime
        .generate(request, ImageCancellation::new(), None)
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = result.images.into_iter().next().unwrap();
    assert_eq!(image.backend, ImageBackendKind::Cuda);
    assert_eq!(image.quantization, "Q4_K_M");
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        png_sha256,
        "82a69a3d50c4502f1166657b8c9df9e6e25848b13f9e00085c29ebc326b1ca71"
    );
    assert_eq!(
        pixel_sha256,
        "428023782ca2d88aea4069f7cfc5eeeb5fc1aa17bd2acc26d0927073cee03df6"
    );
    eprintln!(
        "qwen-image native CUDA smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={png_sha256} timings={:?} arena={:?}",
        image.timings,
        resources.allocation_arena().snapshot()
    );
}

#[test]
#[ignore = "requires the complete pinned BF16 bundle, high RAM, and a long real CPU execution"]
fn pinned_qwen_image_2512_bf16_cpu_generation_smoke() {
    let (image, load_seconds, execution_seconds) = run_pinned_cpu_generation("BF16");
    let png_sha256 = format!("{:x}", Sha256::digest(&image.bytes));
    let decoded = decode_image(&image.bytes, ImageIoLimits::default()).unwrap();
    let pixel_sha256 = format!("{:x}", Sha256::digest(decoded.rgba8()));
    assert_eq!(
        png_sha256,
        "d8f8c6efd3d203b0f4d473bab31e88a21f6e4159f6e622fc61dca3d303f9a965"
    );
    assert_eq!(
        pixel_sha256,
        "346c7b7345fffb3156f5a70552b7a01474fd6fc8d53314fedf4aebab68ff7a0c"
    );
    eprintln!(
        "qwen-image BF16 native CPU smoke: load={load_seconds:.3}s execute={execution_seconds:.3}s pixel_sha256={pixel_sha256} png_sha256={png_sha256} timings={:?}",
        image.timings
    );
}

fn run_pinned_cpu_generation(expected_quantization: &str) -> (ImageResult, f64, f64) {
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let load_started = Instant::now();
    let bundle = ImageModelBundle::open(root).unwrap();
    let model = bundle.manifest().id.clone();
    let runtime = ImageRuntime::load(
        bundle,
        ImageBackendKind::Cpu,
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    let request = ImageGenerationRequest {
        model,
        prompt: "a".to_string(),
        width: 16,
        height: 16,
        steps: 2,
        true_cfg_scale: 1.0,
        backend: ImageBackendKind::Cpu,
        offload: ImageOffloadPolicy::Cpu,
        ..ImageGenerationRequest::default()
    };
    let plan = runtime.plan_generation(&request).unwrap();
    assert_eq!(plan.width, 16);
    assert_eq!(plan.height, 16);
    assert_eq!(plan.estimated_device_bytes, 0);

    let execution_started = Instant::now();
    let result = runtime
        .generate(request, ImageCancellation::new(), None)
        .unwrap();
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    assert_eq!(result.images.len(), 1);
    let image = result.images.into_iter().next().unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.quantization, expected_quantization);
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    (image, load_seconds, execution_seconds)
}

#[test]
#[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and the captured BF16 Diffusers smoke tensors"]
fn pinned_qwen_image_2512_cpu_prompt_matches_real_diffusers_oracle() {
    const PROMPT: &str =
        "A cobalt mechanical keyboard on a walnut desk, precise product photograph.";
    let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
        .map(PathBuf::from)
        .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
    let bundle = ImageModelBundle::open(root).unwrap();
    let config = QwenImageBundleConfig::load(&bundle).unwrap();
    let tokenizer = QwenImagePromptTokenizer::load(
        &bundle,
        config.max_sequence_length,
        config.text_encoder.vocab_size,
    )
    .unwrap();
    let tokens = tokenizer.encode_batch(&[PROMPT]).unwrap();
    let encoder = QwenImageCpuTextEncoder::load_with_config(&bundle, &config).unwrap();
    let actual = encoder.encode_tokens(&tokens).unwrap();
    assert_eq!(actual.shape(), [1, 19, 3_584]);

    let oracle_root = std::env::var_os("XRT_QWEN_IMAGE_REFERENCE_RESULT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
                "../../benchmark-results/image/phase0-2026-07-21/diffusers/\
                 bf16-smoke-512x512-s4-seed424242",
            )
        });
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(oracle_root.join("result.json")).unwrap()).unwrap();
    assert_eq!(report["request"]["prompt"], PROMPT);
    assert_eq!(report["tensors"]["prompt_embeds"]["dtype"], "bfloat16");
    assert_eq!(
        report["tensors"]["prompt_embeds"]["shape"],
        serde_json::json!([1, 19, 3584])
    );
    let encoded = fs::read(oracle_root.join("tensors/prompt_embeds.bin")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&encoded)),
        report["tensors"]["prompt_embeds"]["sha256"]
            .as_str()
            .unwrap()
    );
    let expected = encoded
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(actual.embeddings.len(), expected.len());
    let (max_abs, normalized_rms, cosine) = numerical_metrics(&actual.embeddings, &expected);
    let (max_index, (max_actual, max_expected)) = actual
        .embeddings
        .iter()
        .copied()
        .zip(expected.iter().copied())
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            (left.0 - left.1)
                .abs()
                .total_cmp(&(right.0 - right.1).abs())
        })
        .unwrap();
    eprintln!(
        "real prompt oracle parity: max_abs={max_abs:.9} normalized_rms={normalized_rms:.9} cosine={cosine:.9} max_token={} max_feature={} actual={max_actual} expected={max_expected}",
        max_index / actual.hidden_size,
        max_index % actual.hidden_size,
    );
    let mut worst_token_normalized_rms = 0.0f32;
    let mut worst_token_cosine = 1.0f32;
    for token in 0..actual.sequence_length {
        let start = token * actual.hidden_size;
        let end = start + actual.hidden_size;
        let metrics = numerical_metrics(&actual.embeddings[start..end], &expected[start..end]);
        worst_token_normalized_rms = worst_token_normalized_rms.max(metrics.1);
        worst_token_cosine = worst_token_cosine.min(metrics.2);
        eprintln!(
            "token {token:02}: max_abs={:.9} normalized_rms={:.9} cosine={:.9}",
            metrics.0, metrics.1, metrics.2
        );
    }
    // This is a cross-backend BF16 comparison: the native scalar CPU path
    // and pinned CUDA SDPA oracle share graph semantics but not reduction
    // order. Layer checkpoints bound the drift through all 28 decoder layers.
    assert!(max_abs <= 8.0, "max_abs {max_abs} exceeds 8.0");
    assert!(
        normalized_rms <= 0.025,
        "normalized RMS {normalized_rms} exceeds 0.025"
    );
    assert!(cosine >= 0.999, "cosine {cosine} is below 0.999");
    assert!(
        worst_token_normalized_rms <= 0.065,
        "worst-token normalized RMS {worst_token_normalized_rms} exceeds 0.065"
    );
    assert!(
        worst_token_cosine >= 0.998,
        "worst-token cosine {worst_token_cosine} is below 0.998"
    );
}

fn numerical_metrics(actual: &[f32], expected: &[f32]) -> (f32, f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut expected_squared = 0.0f64;
    let mut actual_squared = 0.0f64;
    let mut dot = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = f64::from(actual) - f64::from(expected);
        max_abs = max_abs.max((actual - expected).abs());
        squared_error += difference * difference;
        expected_squared += f64::from(expected) * f64::from(expected);
        actual_squared += f64::from(actual) * f64::from(actual);
        dot += f64::from(actual) * f64::from(expected);
    }
    let normalized_rms = (squared_error / expected_squared.max(f64::MIN_POSITIVE)).sqrt() as f32;
    let cosine =
        (dot / (actual_squared.sqrt() * expected_squared.sqrt()).max(f64::MIN_POSITIVE)) as f32;
    (max_abs, normalized_rms, cosine)
}
