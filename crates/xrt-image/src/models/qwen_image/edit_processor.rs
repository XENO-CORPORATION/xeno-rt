use std::{fs, path::Path};

use image::{imageops::FilterType, ImageBuffer, Rgb, RgbImage};
use serde::Deserialize;
use xrt_tokenizer::Tokenizer;

use crate::{ComponentFormat, ComponentRole, DecodedImage, ImageError, ImageModelBundle};

use super::{
    prompt::tokenizer_root, QwenImageBundleConfig, QwenImageTokenBatch, QwenImageVisionConfig,
    QwenImageVisionInput, QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS,
};

const EDIT_PROMPT_PREFIX: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
const EDIT_PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
const VISION_START: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const VISION_END: &str = "<|vision_end|>";
const CONDITION_TARGET_AREA: u64 = 384 * 384;
const VAE_TARGET_AREA: u64 = 1024 * 1024;
const DIMENSION_ROUNDING: u32 = 32;
const MAX_PROCESSOR_CONFIG_BYTES: u64 = 64 * 1024;
const MAX_EDIT_SOURCE_IMAGES: usize = 3;

/// Source pixels prepared for the still-image Qwen VAE encoder. Values are
/// channel-first NCTHW (with implicit batch/depth equal to one) in `[-1, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageVaeSource {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageEditImageBatch {
    pub vision: QwenImageVisionInput,
    pub vae_sources: Vec<QwenImageVaeSource>,
    pub condition_sizes: Vec<[u32; 2]>,
    pub vae_sizes: Vec<[u32; 2]>,
}

#[derive(Debug)]
pub struct QwenImageEditProcessor {
    tokenizer: Tokenizer,
    image: QwenImageEditImageProcessor,
    max_sequence_length: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl QwenImageEditProcessor {
    pub fn load(bundle: &ImageModelBundle) -> Result<Self, ImageError> {
        let config = QwenImageBundleConfig::load(bundle)?;
        Self::load_with_config(bundle, &config)
    }

    pub fn load_with_config(
        bundle: &ImageModelBundle,
        config: &QwenImageBundleConfig,
    ) -> Result<Self, ImageError> {
        let vision = config.text_encoder.vision.clone().ok_or_else(|| {
            ImageError::MissingComponent(
                "Qwen Image Edit text encoder has no vision configuration".to_string(),
            )
        })?;
        let components = bundle
            .manifest()
            .components
            .iter()
            .filter(|component| component.role == ComponentRole::Processor)
            .collect::<Vec<_>>();
        let [component] = components.as_slice() else {
            return Err(ImageError::MissingComponent(format!(
                "expected exactly one processor component, found {}",
                components.len()
            )));
        };
        if component.format != ComponentFormat::HuggingFaceJson {
            return Err(ImageError::UnsupportedTensor(format!(
                "processor component format `{}` is not huggingface-json",
                component.format.as_str()
            )));
        }
        let root = tokenizer_root(component.files.iter().map(|file| file.path.as_str()))?;
        let preprocessor_configs = component
            .files
            .iter()
            .filter(|file| {
                Path::new(&file.path)
                    .file_name()
                    .is_some_and(|name| name == "preprocessor_config.json")
            })
            .collect::<Vec<_>>();
        let [preprocessor_config] = preprocessor_configs.as_slice() else {
            return Err(ImageError::MissingComponent(format!(
                "processor requires exactly one preprocessor_config.json, found {}",
                preprocessor_configs.len()
            )));
        };
        if Path::new(&preprocessor_config.path)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            != root
        {
            return Err(ImageError::CorruptComponent(
                "processor config and tokenizer files do not share one directory".to_string(),
            ));
        }
        let config_path = bundle.root().join(&preprocessor_config.path);
        let metadata = fs::metadata(&config_path).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to inspect `{}`: {error}",
                config_path.display()
            ))
        })?;
        if metadata.len() == 0 || metadata.len() > MAX_PROCESSOR_CONFIG_BYTES {
            return Err(ImageError::InputLimit(format!(
                "processor config size {} is outside 1..={MAX_PROCESSOR_CONFIG_BYTES}",
                metadata.len()
            )));
        }
        let bytes = fs::read(&config_path).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to read `{}`: {error}",
                config_path.display()
            ))
        })?;
        let processor_config = Qwen2VlProcessorConfig::parse(&bytes, &vision)?;
        let tokenizer = Tokenizer::from_hf_dir(bundle.root().join(root)).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "Qwen Image Edit processor tokenizer failed validation: {error}"
            ))
        })?;
        if tokenizer.vocab_size() > config.text_encoder.vocab_size {
            return Err(ImageError::UnsupportedShape(format!(
                "processor vocabulary {} exceeds text encoder vocabulary {}",
                tokenizer.vocab_size(),
                config.text_encoder.vocab_size
            )));
        }
        let image_token_id =
            required_token_id(&tokenizer, IMAGE_PAD, config.text_encoder.image_token_id)?;
        let vision_start_token_id = required_token_id(
            &tokenizer,
            VISION_START,
            config.text_encoder.vision_start_token_id,
        )?;
        let vision_end_token_id = required_token_id(
            &tokenizer,
            VISION_END,
            config.text_encoder.vision_end_token_id,
        )?;
        Ok(Self {
            tokenizer,
            image: QwenImageEditImageProcessor {
                config: processor_config,
                vision,
            },
            max_sequence_length: config.max_sequence_length,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }

    pub fn process_images(
        &self,
        images: &[DecodedImage],
    ) -> Result<QwenImageEditImageBatch, ImageError> {
        self.image.process(images)
    }

    pub fn tokenize_prompt(
        &self,
        prompt: &str,
        image_token_counts: &[usize],
    ) -> Result<QwenImageTokenBatch, ImageError> {
        let expected_image_tokens = checked_sum(image_token_counts, "edit image token count")?;
        let formatted = format_edit_prompt(prompt, image_token_counts)?;
        let ids = self
            .tokenizer
            .encode_with_options(&formatted, true, true)
            .map_err(|error| {
                ImageError::Execution(format!("edit prompt tokenization failed: {error}"))
            })?;
        let encoded_limit = self
            .max_sequence_length
            .checked_add(expected_image_tokens)
            .and_then(|value| value.checked_add(QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS))
            .ok_or_else(|| {
                ImageError::UnsupportedShape("edit prompt encoded length overflow".to_string())
            })?;
        if ids.len() <= QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS || ids.len() > encoded_limit {
            return Err(ImageError::InputLimit(format!(
                "encoded edit prompt length {} is outside {}..={encoded_limit}",
                ids.len(),
                QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS + 1
            )));
        }
        for (label, actual, expected) in [
            (
                "image",
                ids.iter()
                    .filter(|token| **token == self.image_token_id)
                    .count(),
                expected_image_tokens,
            ),
            (
                "vision-start",
                ids.iter()
                    .filter(|token| **token == self.vision_start_token_id)
                    .count(),
                image_token_counts.len(),
            ),
            (
                "vision-end",
                ids.iter()
                    .filter(|token| **token == self.vision_end_token_id)
                    .count(),
                image_token_counts.len(),
            ),
        ] {
            if actual != expected {
                return Err(ImageError::CorruptComponent(format!(
                    "processor encoded {actual} {label} tokens, expected {expected}"
                )));
            }
        }
        let valid_length = ids.len();
        Ok(QwenImageTokenBatch {
            input_ids: vec![ids],
            attention_mask: vec![vec![1; valid_length]],
            retained_lengths: vec![valid_length - QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS],
            context_extension: expected_image_tokens,
            drop_tokens: QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS,
        })
    }
}

#[derive(Debug)]
struct QwenImageEditImageProcessor {
    config: Qwen2VlProcessorConfig,
    vision: QwenImageVisionConfig,
}

impl QwenImageEditImageProcessor {
    fn process(&self, images: &[DecodedImage]) -> Result<QwenImageEditImageBatch, ImageError> {
        if images.is_empty() || images.len() > MAX_EDIT_SOURCE_IMAGES {
            return Err(ImageError::InvalidRequest(format!(
                "Qwen Image Edit requires 1..={MAX_EDIT_SOURCE_IMAGES} ordered source images"
            )));
        }
        let mut vision_values = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        let mut vae_sources = Vec::with_capacity(images.len());
        let mut condition_sizes = Vec::with_capacity(images.len());
        let mut vae_sizes = Vec::with_capacity(images.len());
        for image in images {
            let rgb = decoded_rgb(image)?;
            let (condition_width, condition_height) =
                calculate_dimensions(CONDITION_TARGET_AREA, image.width(), image.height())?;
            let condition = image::imageops::resize(
                &rgb,
                condition_width,
                condition_height,
                FilterType::Lanczos3,
            );
            let factor = self
                .vision
                .patch_size
                .checked_mul(self.vision.spatial_merge_size)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    ImageError::UnsupportedShape("vision resize factor overflow".to_string())
                })?;
            let (vision_height, vision_width) = smart_resize(
                condition_height,
                condition_width,
                factor,
                self.config.min_pixels,
                self.config.max_pixels,
            )?;
            let condition = image::imageops::resize(
                &condition,
                vision_width,
                vision_height,
                FilterType::CatmullRom,
            );
            let (mut values, grid) = flatten_rgb_patches(
                &condition,
                &self.vision,
                self.config.image_mean,
                self.config.image_std,
            )?;
            vision_values.append(&mut values);
            grids.push(grid);
            condition_sizes.push([vision_width, vision_height]);

            let (vae_width, vae_height) =
                calculate_dimensions(VAE_TARGET_AREA, image.width(), image.height())?;
            let vae_image =
                image::imageops::resize(&rgb, vae_width, vae_height, FilterType::Lanczos3);
            let vae_values = normalize_vae_rgb(&vae_image)?;
            vae_sources.push(QwenImageVaeSource {
                values: vae_values,
                width: vae_width,
                height: vae_height,
            });
            vae_sizes.push([vae_width, vae_height]);
        }
        Ok(QwenImageEditImageBatch {
            vision: QwenImageVisionInput {
                pixel_values: vision_values,
                grids,
            },
            vae_sources,
            condition_sizes,
            vae_sizes,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Qwen2VlProcessorConfig {
    data_format: String,
    do_convert_rgb: bool,
    do_normalize: bool,
    do_rescale: bool,
    do_resize: bool,
    image_mean: [f32; 3],
    image_processor_type: String,
    image_std: [f32; 3],
    max_pixels: u64,
    merge_size: usize,
    min_pixels: u64,
    patch_size: usize,
    processor_class: String,
    resample: u8,
    rescale_factor: f32,
    size: Qwen2VlProcessorSize,
    temporal_patch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Qwen2VlProcessorSize {
    longest_edge: u64,
    shortest_edge: u64,
}

impl Qwen2VlProcessorConfig {
    fn parse(bytes: &[u8], vision: &QwenImageVisionConfig) -> Result<Self, ImageError> {
        let config: Self = serde_json::from_slice(bytes).map_err(|error| {
            ImageError::CorruptComponent(format!("invalid Qwen2-VL preprocessor config: {error}"))
        })?;
        let expected_mean = [0.481_454_66, 0.457_827_5, 0.408_210_73];
        let expected_std = [0.268_629_54, 0.261_302_6, 0.275_777_1];
        if config.data_format != "channels_first"
            || !config.do_convert_rgb
            || !config.do_normalize
            || !config.do_rescale
            || !config.do_resize
            || config.image_processor_type != "Qwen2VLImageProcessorFast"
            || config.processor_class != "Qwen2VLProcessor"
            || config.resample != 3
            || (config.rescale_factor - 1.0 / 255.0).abs() > 1e-9
            || config.patch_size != vision.patch_size
            || config.merge_size != vision.spatial_merge_size
            || config.temporal_patch_size != vision.temporal_patch_size
            || config.min_pixels == 0
            || config.min_pixels > config.max_pixels
            || config.size.shortest_edge != config.min_pixels
            || config.size.longest_edge != config.max_pixels
            || !triplet_close(config.image_mean, expected_mean, 1e-7)
            || !triplet_close(config.image_std, expected_std, 1e-7)
        {
            return Err(ImageError::UnsupportedCapability(
                "processor config differs from the pinned Qwen2-VL image contract".to_string(),
            ));
        }
        Ok(config)
    }
}

fn required_token_id(
    tokenizer: &Tokenizer,
    piece: &str,
    configured: Option<u32>,
) -> Result<u32, ImageError> {
    let configured = configured.ok_or_else(|| {
        ImageError::UnsupportedShape(format!("text config does not declare token `{piece}`"))
    })?;
    let actual = tokenizer.token_id_for_piece(piece).ok_or_else(|| {
        ImageError::CorruptComponent(format!("processor tokenizer has no `{piece}` token"))
    })?;
    if actual != configured {
        return Err(ImageError::UnsupportedShape(format!(
            "processor token `{piece}` is {actual}, text config declares {configured}"
        )));
    }
    Ok(actual)
}

fn format_edit_prompt(prompt: &str, image_token_counts: &[usize]) -> Result<String, ImageError> {
    if image_token_counts.is_empty() || image_token_counts.len() > MAX_EDIT_SOURCE_IMAGES {
        return Err(ImageError::InvalidRequest(format!(
            "edit prompt requires 1..={MAX_EDIT_SOURCE_IMAGES} image token spans"
        )));
    }
    if image_token_counts.contains(&0) {
        return Err(ImageError::InvalidRequest(
            "edit image token spans must be non-empty".to_string(),
        ));
    }
    if [VISION_START, IMAGE_PAD, VISION_END]
        .iter()
        .any(|marker| prompt.contains(marker))
    {
        return Err(ImageError::InvalidRequest(
            "edit prompt must not contain reserved Qwen vision markers".to_string(),
        ));
    }
    let image_tokens = checked_sum(image_token_counts, "edit image token count")?;
    let marker_bytes = image_tokens
        .checked_mul(IMAGE_PAD.len())
        .and_then(|value| {
            value.checked_add(
                image_token_counts
                    .len()
                    .checked_mul(VISION_START.len() + VISION_END.len() + 16)?,
            )
        })
        .ok_or_else(|| ImageError::InputLimit("edit prompt marker size overflow".to_string()))?;
    let capacity = EDIT_PROMPT_PREFIX
        .len()
        .checked_add(EDIT_PROMPT_SUFFIX.len())
        .and_then(|value| value.checked_add(prompt.len()))
        .and_then(|value| value.checked_add(marker_bytes))
        .ok_or_else(|| ImageError::InputLimit("formatted edit prompt overflow".to_string()))?;
    let mut formatted = String::with_capacity(capacity);
    formatted.push_str(EDIT_PROMPT_PREFIX);
    for (index, count) in image_token_counts.iter().copied().enumerate() {
        formatted.push_str("Picture ");
        formatted.push_str(&(index + 1).to_string());
        formatted.push_str(": ");
        formatted.push_str(VISION_START);
        for _ in 0..count {
            formatted.push_str(IMAGE_PAD);
        }
        formatted.push_str(VISION_END);
    }
    formatted.push_str(prompt);
    formatted.push_str(EDIT_PROMPT_SUFFIX);
    Ok(formatted)
}

fn decoded_rgb(image: &DecodedImage) -> Result<RgbImage, ImageError> {
    let capacity = u64::from(image.width())
        .checked_mul(u64::from(image.height()))
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| ImageError::InputLimit("source RGB size overflow".to_string()))?;
    let mut rgb = Vec::with_capacity(capacity);
    for pixel in image.rgba8().chunks_exact(4) {
        // Match PIL `convert("RGB")`: retain stored RGB and discard alpha.
        rgb.extend_from_slice(&pixel[..3]);
    }
    ImageBuffer::<Rgb<u8>, _>::from_raw(image.width(), image.height(), rgb).ok_or_else(|| {
        ImageError::Internal("validated edit source became an invalid RGB image".to_string())
    })
}

fn calculate_dimensions(
    target_area: u64,
    source_width: u32,
    source_height: u32,
) -> Result<(u32, u32), ImageError> {
    if target_area == 0 || source_width == 0 || source_height == 0 {
        return Err(ImageError::UnsupportedShape(
            "edit resize requires positive area and dimensions".to_string(),
        ));
    }
    let ratio = f64::from(source_width) / f64::from(source_height);
    if !ratio.is_finite() || ratio.max(1.0 / ratio) > 200.0 {
        return Err(ImageError::UnsupportedShape(format!(
            "edit source aspect ratio {ratio} exceeds the Qwen processor limit"
        )));
    }
    let width = ((target_area as f64) * ratio).sqrt();
    let height = width / ratio;
    Ok((
        rounded_multiple(width, DIMENSION_ROUNDING)?,
        rounded_multiple(height, DIMENSION_ROUNDING)?,
    ))
}

pub(super) fn qwen_image_edit_output_dimensions(
    source_width: u32,
    source_height: u32,
) -> Result<(u32, u32), ImageError> {
    calculate_dimensions(VAE_TARGET_AREA, source_width, source_height)
}

fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), ImageError> {
    if height == 0 || width == 0 || factor == 0 || min_pixels == 0 || min_pixels > max_pixels {
        return Err(ImageError::UnsupportedShape(
            "invalid Qwen2-VL smart-resize geometry".to_string(),
        ));
    }
    let aspect = f64::from(height.max(width)) / f64::from(height.min(width));
    if aspect > 200.0 {
        return Err(ImageError::UnsupportedShape(format!(
            "absolute image aspect ratio must not exceed 200, found {aspect}"
        )));
    }
    let mut resized_height = u64::from(rounded_multiple(f64::from(height), factor)?);
    let mut resized_width = u64::from(rounded_multiple(f64::from(width), factor)?);
    let pixels = resized_height.checked_mul(resized_width).ok_or_else(|| {
        ImageError::UnsupportedShape("smart-resize pixel count overflow".to_string())
    })?;
    if pixels > max_pixels {
        let beta = ((u64::from(height) * u64::from(width)) as f64 / max_pixels as f64).sqrt();
        resized_height = ((f64::from(height) / beta / f64::from(factor)).floor() as u64)
            .max(u64::from(factor))
            * u64::from(factor);
        resized_width = ((f64::from(width) / beta / f64::from(factor)).floor() as u64)
            .max(u64::from(factor))
            * u64::from(factor);
    } else if pixels < min_pixels {
        let source_pixels = u64::from(height)
            .checked_mul(u64::from(width))
            .ok_or_else(|| {
                ImageError::UnsupportedShape("smart-resize source pixels overflow".to_string())
            })?;
        let beta = (min_pixels as f64 / source_pixels as f64).sqrt();
        resized_height =
            (f64::from(height) * beta / f64::from(factor)).ceil() as u64 * u64::from(factor);
        resized_width =
            (f64::from(width) * beta / f64::from(factor)).ceil() as u64 * u64::from(factor);
    }
    Ok((
        u32::try_from(resized_height).map_err(|_| {
            ImageError::UnsupportedShape("smart-resize height exceeds u32".to_string())
        })?,
        u32::try_from(resized_width).map_err(|_| {
            ImageError::UnsupportedShape("smart-resize width exceeds u32".to_string())
        })?,
    ))
}

fn rounded_multiple(value: f64, factor: u32) -> Result<u32, ImageError> {
    if !value.is_finite() || value <= 0.0 || factor == 0 {
        return Err(ImageError::UnsupportedShape(
            "resize rounding received invalid geometry".to_string(),
        ));
    }
    let units = round_ties_even_positive(value / f64::from(factor));
    let rounded = units
        .checked_mul(u64::from(factor))
        .ok_or_else(|| ImageError::UnsupportedShape("resize dimension overflow".to_string()))?;
    u32::try_from(rounded)
        .map_err(|_| ImageError::UnsupportedShape("resize dimension exceeds u32".to_string()))
}

fn round_ties_even_positive(value: f64) -> u64 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor = floor as u64;
    if fraction > 0.5 || (fraction == 0.5 && floor % 2 == 1) {
        floor + 1
    } else {
        floor
    }
}

fn flatten_rgb_patches(
    image: &RgbImage,
    vision: &QwenImageVisionConfig,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Vec<f32>, [usize; 3]), ImageError> {
    let width = usize::try_from(image.width()).map_err(|_| {
        ImageError::UnsupportedShape("vision image width exceeds usize".to_string())
    })?;
    let height = usize::try_from(image.height()).map_err(|_| {
        ImageError::UnsupportedShape("vision image height exceeds usize".to_string())
    })?;
    let factor = vision
        .patch_size
        .checked_mul(vision.spatial_merge_size)
        .ok_or_else(|| ImageError::UnsupportedShape("vision patch factor overflow".to_string()))?;
    if vision.in_channels != 3
        || width % factor != 0
        || height % factor != 0
        || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ImageError::UnsupportedShape(format!(
            "invalid processor image geometry {width}x{height} for factor {factor}"
        )));
    }
    let grid_height = height / vision.patch_size;
    let grid_width = width / vision.patch_size;
    let rows = checked_product(&[grid_height, grid_width], "vision patch rows")?;
    let row_width = checked_product(
        &[
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ],
        "vision patch row width",
    )?;
    let mut output = Vec::with_capacity(checked_product(
        &[rows, row_width],
        "vision processor output",
    )?);
    for group_height in 0..grid_height / vision.spatial_merge_size {
        for group_width in 0..grid_width / vision.spatial_merge_size {
            for local_height in 0..vision.spatial_merge_size {
                for local_width in 0..vision.spatial_merge_size {
                    for channel in 0..vision.in_channels {
                        for _temporal in 0..vision.temporal_patch_size {
                            for patch_height in 0..vision.patch_size {
                                for patch_width in 0..vision.patch_size {
                                    let y = (group_height * vision.spatial_merge_size
                                        + local_height)
                                        * vision.patch_size
                                        + patch_height;
                                    let x = (group_width * vision.spatial_merge_size + local_width)
                                        * vision.patch_size
                                        + patch_width;
                                    let pixel = image.get_pixel(x as u32, y as u32).0[channel];
                                    output.push(
                                        (f32::from(pixel) / 255.0 - mean[channel]) / std[channel],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((output, [1, grid_height, grid_width]))
}

fn normalize_vae_rgb(image: &RgbImage) -> Result<Vec<f32>, ImageError> {
    let pixels = usize::try_from(image.width())
        .ok()
        .and_then(|width| {
            usize::try_from(image.height())
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| ImageError::UnsupportedShape("VAE image size overflow".to_string()))?;
    let mut output = Vec::with_capacity(checked_product(&[3, pixels], "VAE source pixels")?);
    for channel in 0..3 {
        for pixel in image.pixels() {
            output.push(f32::from(pixel.0[channel]) / 127.5 - 1.0);
        }
    }
    Ok(output)
}

fn checked_sum(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
    })
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
    })
}

fn triplet_close(left: [f32; 3], right: [f32; 3], tolerance: f32) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vision() -> QwenImageVisionConfig {
        QwenImageVisionConfig {
            depth: 2,
            fullatt_block_indexes: vec![1],
            hidden_act: "silu".to_string(),
            hidden_size: 8,
            in_channels: 3,
            intermediate_size: 16,
            num_heads: 2,
            out_hidden_size: 6,
            patch_size: 1,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 4,
        }
    }

    fn pinned_vision() -> QwenImageVisionConfig {
        QwenImageVisionConfig {
            depth: 1,
            fullatt_block_indexes: vec![0],
            hidden_act: "silu".to_string(),
            hidden_size: 8,
            in_channels: 3,
            intermediate_size: 16,
            num_heads: 2,
            out_hidden_size: 6,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 112,
        }
    }

    fn pinned_processor_config(vision: &QwenImageVisionConfig) -> Qwen2VlProcessorConfig {
        Qwen2VlProcessorConfig::parse(
            br#"{
                "data_format":"channels_first",
                "do_convert_rgb":true,
                "do_normalize":true,
                "do_rescale":true,
                "do_resize":true,
                "image_mean":[0.48145466,0.4578275,0.40821073],
                "image_processor_type":"Qwen2VLImageProcessorFast",
                "image_std":[0.26862954,0.26130258,0.27577711],
                "max_pixels":12845056,
                "merge_size":2,
                "min_pixels":3136,
                "patch_size":14,
                "processor_class":"Qwen2VLProcessor",
                "resample":3,
                "rescale_factor":0.00392156862745098,
                "size":{"longest_edge":12845056,"shortest_edge":3136},
                "temporal_patch_size":2
            }"#,
            vision,
        )
        .unwrap()
    }

    #[test]
    fn official_area_and_smart_resize_rules_are_deterministic() {
        assert_eq!(
            calculate_dimensions(CONDITION_TARGET_AREA, 16, 9).unwrap(),
            (512, 288)
        );
        assert_eq!(
            calculate_dimensions(VAE_TARGET_AREA, 16, 9).unwrap(),
            (1376, 768)
        );
        assert_eq!(
            smart_resize(288, 512, 28, 3136, 12_845_056).unwrap(),
            (280, 504)
        );
        assert_eq!(round_ties_even_positive(2.5), 2);
        assert_eq!(round_ties_even_positive(3.5), 4);
    }

    #[test]
    fn edit_source_alpha_is_discarded_without_compositing_rgb() {
        let source = DecodedImage::new_rgba8(2, 1, vec![255, 0, 0, 0, 0, 255, 0, 128]).unwrap();

        let rgb = decoded_rgb(&source).unwrap();

        assert_eq!(rgb.into_raw(), vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn patch_rows_follow_qwen_merge_channel_temporal_order() {
        let image = RgbImage::from_raw(2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        let (values, grid) =
            flatten_rgb_patches(&image, &tiny_vision(), [0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(grid, [1, 2, 2]);
        assert_eq!(values.len(), 24);
        let bytes = values
            .into_iter()
            .map(|value| (value * 255.0).round() as u8)
            .collect::<Vec<_>>();
        assert_eq!(
            bytes,
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12]
        );
    }

    #[test]
    fn edit_template_preserves_order_and_expands_each_image_span() {
        let formatted = format_edit_prompt("replace the sky", &[2, 3]).unwrap();
        assert!(formatted.starts_with(EDIT_PROMPT_PREFIX));
        assert!(formatted.ends_with(EDIT_PROMPT_SUFFIX));
        assert!(formatted.contains(
            "Picture 1: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>Picture 2: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>replace the sky"
        ));
        assert_eq!(formatted.matches(IMAGE_PAD).count(), 5);
    }

    #[test]
    fn edit_template_accepts_three_visual_spans_larger_than_text_context() {
        let formatted = format_edit_prompt("combine them", &[196, 196, 196]).unwrap();
        assert_eq!(formatted.matches(IMAGE_PAD).count(), 588);
        assert!(formatted.contains("Picture 3:"));
    }

    #[test]
    fn edit_template_rejects_user_supplied_vision_markers() {
        let error = format_edit_prompt("literal <|image_pad|>", &[1]).unwrap_err();
        assert_eq!(error.kind(), crate::ImageErrorKind::InvalidRequest);
    }

    #[test]
    fn image_preprocessing_tracks_pinned_diffusers_and_transformers_samples() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/edit-processor-diffusers-0.39.json"
        ))
        .unwrap();
        let mut rgba = Vec::with_capacity(16 * 9 * 4);
        for y in 0..9u8 {
            for x in 0..16u8 {
                rgba.extend_from_slice(&[
                    x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)),
                    x.wrapping_mul(7)
                        .wrapping_add(y.wrapping_mul(29))
                        .wrapping_add(31),
                    x.wrapping_mul(3)
                        .wrapping_add(y.wrapping_mul(5))
                        .wrapping_add(127),
                    0,
                ]);
            }
        }
        let source = DecodedImage::new_rgba8(16, 9, rgba).unwrap();
        let vision = pinned_vision();
        let processor = QwenImageEditImageProcessor {
            config: pinned_processor_config(&vision),
            vision,
        };
        let output = processor.process(&[source]).unwrap();
        assert_eq!(output.condition_sizes, [[504, 280]]);
        assert_eq!(output.vision.grids, [[1, 20, 36]]);
        assert_eq!(output.vae_sizes, [[1376, 768]]);
        assert_eq!(output.vision.pixel_values.len(), 720 * 1176);
        assert_eq!(output.vae_sources[0].values.len(), 3 * 768 * 1376);

        for (index, expected) in fixture["vision_sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(fixture["vision_samples"].as_array().unwrap())
        {
            let index = index.as_u64().unwrap() as usize;
            let expected = expected.as_f64().unwrap() as f32;
            let actual = output.vision.pixel_values[index];
            assert!(
                (actual - expected).abs() <= 0.035,
                "vision sample {index}: actual={actual}, expected={expected}"
            );
        }
        for (index, expected) in fixture["vae_sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(fixture["vae_samples"].as_array().unwrap())
        {
            let index = index.as_u64().unwrap() as usize;
            let expected = expected.as_f64().unwrap() as f32;
            let actual = output.vae_sources[0].values[index];
            assert!(
                (actual - expected).abs() <= 0.025,
                "VAE sample {index}: actual={actual}, expected={expected}"
            );
        }
    }
}
