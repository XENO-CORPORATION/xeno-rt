#![cfg(feature = "image-generation-tests")]

use std::sync::Arc;

use xrt_image::{
    synthetic_bundle_for_tests, DecodedImage, ImageBackendKind, ImageCancellation,
    ImageEditRequest, ImageErrorKind, ImageGenerationRequest, ImageProgressPhase, ImageRuntime,
};
use xrt_runtime::{GpuResourceConfig, GpuResourceManager};

fn runtime() -> ImageRuntime {
    ImageRuntime::load(
        synthetic_bundle_for_tests(),
        ImageBackendKind::Cpu,
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap()
}

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        model: "xrt-image-synthetic-v1".to_string(),
        prompt: "a cobalt keyboard".to_string(),
        width: 32,
        height: 32,
        n: 2,
        steps: 4,
        seed: 42,
        ..ImageGenerationRequest::default()
    }
}

#[test]
fn synthetic_generation_is_ordered_and_deterministic() {
    let first = runtime()
        .generate(request(), ImageCancellation::new(), None)
        .unwrap();
    let second = runtime()
        .generate(request(), ImageCancellation::new(), None)
        .unwrap();
    assert_eq!(first.images.len(), 2);
    assert_eq!(first.images[0].seed, 42);
    assert_eq!(first.images[1].seed, 43);
    assert_eq!(first.images[0].bytes, second.images[0].bytes);
    assert_eq!(first.images[1].bytes, second.images[1].bytes);
    assert_ne!(first.images[0].bytes, first.images[1].bytes);
}

#[test]
fn generation_plan_exposes_ordered_seeds_and_memory_before_execution() {
    let plan = runtime().plan_generation(&request()).unwrap();
    assert_eq!(plan.outputs.len(), 2);
    assert_eq!(plan.outputs[0].seed, 42);
    assert_eq!(plan.outputs[1].seed, 43);
    assert!(plan.estimated_host_bytes > 0);
    assert_eq!(plan.estimated_device_bytes, 0);
}

#[test]
fn synthetic_edit_uses_ordered_source_images() {
    let red = DecodedImage::new_rgba8(32, 32, [255, 0, 0, 255].repeat(32 * 32)).unwrap();
    let blue = DecodedImage::new_rgba8(32, 32, [0, 0, 255, 255].repeat(32 * 32)).unwrap();
    let mut generation = request();
    generation.n = 1;
    let forward = runtime()
        .edit(
            ImageEditRequest {
                generation: generation.clone(),
                images: vec![red.clone(), blue.clone()],
                mask: None,
                strength: 0.75,
            },
            ImageCancellation::new(),
            None,
        )
        .unwrap();
    let reverse = runtime()
        .edit(
            ImageEditRequest {
                generation,
                images: vec![blue, red],
                mask: None,
                strength: 0.75,
            },
            ImageCancellation::new(),
            None,
        )
        .unwrap();
    assert_eq!(forward.images[0].seed, reverse.images[0].seed);
    assert_eq!(forward.images[0].mime_type, "image/png");
    assert_ne!(forward.images[0].bytes, reverse.images[0].bytes);
}

#[test]
fn cancellation_from_progress_stops_between_steps() {
    let cancellation = ImageCancellation::new();
    let callback_cancellation = cancellation.clone();
    let progress = Arc::new(move |event: &xrt_image::ImageProgressEvent| {
        if event.phase == ImageProgressPhase::Denoising && event.step == Some(0) {
            callback_cancellation.cancel();
        }
    });
    let error = runtime()
        .generate(request(), cancellation, Some(progress))
        .unwrap_err();
    assert_eq!(error.kind(), ImageErrorKind::Cancelled);
}

#[test]
fn edit_mask_is_rejected_without_inpaint_capability() {
    let source = DecodedImage::new_rgba8(32, 32, [0, 0, 0, 255].repeat(32 * 32)).unwrap();
    let mut generation = request();
    generation.n = 1;
    let error = runtime()
        .edit(
            ImageEditRequest {
                generation,
                images: vec![source.clone()],
                mask: Some(source),
                strength: 0.5,
            },
            ImageCancellation::new(),
            None,
        )
        .unwrap_err();
    assert_eq!(error.kind(), ImageErrorKind::UnsupportedCapability);
}
