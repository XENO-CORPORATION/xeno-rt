use serde::{Deserialize, Serialize};

use crate::{
    DecodedImage, ImageBackendKind, ImageBatchTimings, ImageError, ImageOffloadPolicy, ImageTimings,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImageCapability {
    #[serde(rename = "image.generate")]
    Generate,
    #[serde(rename = "image.edit")]
    Edit,
    #[serde(rename = "image.inpaint")]
    Inpaint,
}

impl ImageCapability {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Generate => "image.generate",
            Self::Edit => "image.edit",
            Self::Inpaint => "image.inpaint",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageOutputFormat {
    #[default]
    Png,
    Jpeg,
    Webp,
}

impl ImageOutputFormat {
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    #[default]
    Standard,
    Hd,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResizePolicy {
    #[default]
    Reject,
    RoundDown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub n: usize,
    pub steps: usize,
    pub true_cfg_scale: f32,
    pub seed: u64,
    pub output_format: ImageOutputFormat,
    pub quality: ImageQuality,
    pub backend: ImageBackendKind,
    pub offload: ImageOffloadPolicy,
    pub resize_policy: ImageResizePolicy,
    pub preview_interval: Option<usize>,
}

impl Default for ImageGenerationRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            n: 1,
            steps: 50,
            true_cfg_scale: 4.0,
            seed: 0,
            output_format: ImageOutputFormat::Png,
            quality: ImageQuality::Standard,
            backend: ImageBackendKind::Auto,
            offload: ImageOffloadPolicy::Sequential,
            resize_policy: ImageResizePolicy::Reject,
            preview_interval: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageEditRequest {
    pub generation: ImageGenerationRequest,
    pub images: Vec<DecodedImage>,
    pub mask: Option<DecodedImage>,
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRequestLimits {
    pub max_prompt_bytes: usize,
    pub max_outputs: usize,
    pub max_steps: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub dimension_multiple: u32,
    pub max_source_images: usize,
}

impl Default for ImageRequestLimits {
    fn default() -> Self {
        Self {
            max_prompt_bytes: 32 * 1024,
            max_outputs: 4,
            max_steps: 100,
            max_width: 4096,
            max_height: 4096,
            max_pixels: 16_777_216,
            dimension_multiple: 16,
            max_source_images: 3,
        }
    }
}

impl ImageRequestLimits {
    pub fn validate_generation(&self, request: &ImageGenerationRequest) -> Result<(), ImageError> {
        if request.model.trim().is_empty() {
            return Err(ImageError::InvalidRequest(
                "model must not be empty".to_string(),
            ));
        }
        if request.prompt.is_empty() || request.prompt.len() > self.max_prompt_bytes {
            return Err(ImageError::InvalidRequest(format!(
                "prompt byte length must be between 1 and {}",
                self.max_prompt_bytes
            )));
        }
        if request.n == 0 || request.n > self.max_outputs {
            return Err(ImageError::InvalidRequest(format!(
                "n must be between 1 and {}",
                self.max_outputs
            )));
        }
        if request.steps == 0 || request.steps > self.max_steps {
            return Err(ImageError::InvalidRequest(format!(
                "steps must be between 1 and {}",
                self.max_steps
            )));
        }
        if !request.true_cfg_scale.is_finite() || request.true_cfg_scale < 0.0 {
            return Err(ImageError::InvalidRequest(
                "true_cfg_scale must be finite and non-negative".to_string(),
            ));
        }
        if request.width == 0
            || request.height == 0
            || request.width > self.max_width
            || request.height > self.max_height
        {
            return Err(ImageError::UnsupportedShape(format!(
                "dimensions {}x{} exceed the admitted 1..={} by 1..={} range",
                request.width, request.height, self.max_width, self.max_height
            )));
        }
        let pixels = u64::from(request.width)
            .checked_mul(u64::from(request.height))
            .ok_or_else(|| ImageError::UnsupportedShape("pixel count overflowed".to_string()))?;
        if pixels > self.max_pixels {
            return Err(ImageError::UnsupportedShape(format!(
                "{pixels} decoded pixels exceed the {}-pixel limit",
                self.max_pixels
            )));
        }
        if self.dimension_multiple == 0 {
            return Err(ImageError::Internal(
                "dimension_multiple must be greater than zero".to_string(),
            ));
        }
        if request.resize_policy == ImageResizePolicy::Reject
            && (request.width % self.dimension_multiple != 0
                || request.height % self.dimension_multiple != 0)
        {
            return Err(ImageError::UnsupportedShape(format!(
                "dimensions must be divisible by {}",
                self.dimension_multiple
            )));
        }
        if request.preview_interval == Some(0) {
            return Err(ImageError::InvalidRequest(
                "preview_interval must be greater than zero".to_string(),
            ));
        }
        request
            .seed
            .checked_add((request.n - 1) as u64)
            .ok_or_else(|| {
                ImageError::InvalidRequest("derived output seed overflows u64".to_string())
            })?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResult {
    #[serde(skip)]
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub model: String,
    pub bundle_digest: String,
    pub backend: ImageBackendKind,
    pub quantization: String,
    pub timings: ImageTimings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBatchResult {
    pub images: Vec<ImageResult>,
    pub timings: ImageBatchTimings,
}
