//! Native, headless image generation and generative editing for XENO RT.
//!
//! This crate deliberately owns an image-specific execution contract. It
//! shares formats, kernels, and resource accounting with the text runtime but
//! does not emulate token sessions or KV-cache state.

mod backend;
mod bundle;
mod cancellation;
mod error;
mod image_io;
mod memory;
mod metrics;
pub mod models;
mod pipeline;
mod request;
mod rng;
mod runtime;
pub mod scheduler;

#[cfg(feature = "test-util")]
mod synthetic;

pub use backend::{ImageBackendKind, ImageOffloadPolicy};
pub use bundle::{
    BundleComponent, BundleFile, BundleLicense, BundleLimits, BundleManifest, ComponentFormat,
    ComponentRole, ImageModelBundle, ManifestMode,
};
pub use cancellation::{
    ImageCancellation, ImagePreviewEvent, ImageProgressEvent, ImageProgressPhase, ImageProgressSink,
};
pub use error::{ImageError, ImageErrorKind};
pub use image_io::{decode_image, encode_image, DecodedImage, ImageIoLimits};
pub use memory::{ImageExecutionPlan, ImageRequestKind, PlannedImageOutput};
pub use metrics::{ImageBatchTimings, ImageTimings};
pub use request::{
    ImageBatchResult, ImageCapability, ImageEditRequest, ImageGenerationRequest, ImageOutputFormat,
    ImageQuality, ImageRequestLimits, ImageResizePolicy, ImageResult,
};
pub use rng::{NormalRngV1, IMAGE_RNG_SCHEMA_V1};
pub use runtime::ImageRuntime;

#[cfg(feature = "test-util")]
pub use synthetic::synthetic_bundle_for_tests;
