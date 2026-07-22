use std::io::Cursor;

use image::{
    codecs::jpeg::JpegEncoder, metadata::Orientation, DynamicImage, ImageBuffer, ImageDecoder,
    ImageFormat, ImageReader, Rgba,
};

use crate::{ImageError, ImageOutputFormat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

impl DecodedImage {
    pub fn new_rgba8(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, ImageError> {
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| ImageError::InputLimit("RGBA byte length overflowed".to_string()))?;
        if width == 0 || height == 0 || rgba8.len() != expected {
            return Err(ImageError::Codec(format!(
                "RGBA buffer has {} bytes, expected {expected} for {width}x{height}",
                rgba8.len()
            )));
        }
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }

    fn to_dynamic(&self) -> Result<DynamicImage, ImageError> {
        ImageBuffer::<Rgba<u8>, _>::from_raw(self.width, self.height, self.rgba8.clone())
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| ImageError::Internal("validated RGBA buffer became invalid".to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageIoLimits {
    pub max_encoded_bytes: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub max_output_bytes: usize,
}

impl Default for ImageIoLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 32 * 1024 * 1024,
            max_width: 4096,
            max_height: 4096,
            max_pixels: 16_777_216,
            max_output_bytes: 64 * 1024 * 1024,
        }
    }
}

pub fn decode_image(bytes: &[u8], limits: ImageIoLimits) -> Result<DecodedImage, ImageError> {
    if bytes.is_empty() || bytes.len() > limits.max_encoded_bytes {
        return Err(ImageError::InputLimit(format!(
            "encoded image size must be between 1 and {} bytes",
            limits.max_encoded_bytes
        )));
    }
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| ImageError::Codec(error.to_string()))?;
    let format = reader
        .format()
        .ok_or_else(|| ImageError::Codec("unknown image format".to_string()))?;
    if !matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP
    ) {
        return Err(ImageError::Codec(format!(
            "unsupported input image format {format:?}"
        )));
    }
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .map_err(|error| ImageError::Codec(error.to_string()))?;
    if decoder
        .icc_profile()
        .map_err(|error| ImageError::Codec(format!("invalid embedded ICC profile: {error}")))?
        .is_some()
    {
        return Err(ImageError::Codec(
            "embedded ICC color profiles are unsupported; convert the image to sRGB before input"
                .to_string(),
        ));
    }
    let orientation = decoder.orientation().map_err(|error| {
        ImageError::Codec(format!("invalid image orientation metadata: {error}"))
    })?;
    let (encoded_width, encoded_height) = decoder.dimensions();
    let (width, height) = if matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    ) {
        (encoded_height, encoded_width)
    } else {
        (encoded_width, encoded_height)
    };
    validate_dimensions(width, height, limits)?;

    let mut decoded = DynamicImage::from_decoder(decoder)
        .map_err(|error| ImageError::Codec(error.to_string()))?;
    decoded.apply_orientation(orientation);
    let decoded = decoded.to_rgba8();
    DecodedImage::new_rgba8(width, height, decoded.into_raw())
}

pub fn encode_image(
    image: &DecodedImage,
    format: ImageOutputFormat,
    quality: u8,
    max_output_bytes: usize,
) -> Result<Vec<u8>, ImageError> {
    let dynamic = image.to_dynamic()?;
    let mut output = Cursor::new(Vec::new());
    match format {
        ImageOutputFormat::Png => dynamic
            .write_to(&mut output, ImageFormat::Png)
            .map_err(|error| ImageError::Codec(error.to_string()))?,
        ImageOutputFormat::Webp => dynamic
            .write_to(&mut output, ImageFormat::WebP)
            .map_err(|error| ImageError::Codec(error.to_string()))?,
        ImageOutputFormat::Jpeg => {
            let rgb = dynamic.to_rgb8();
            JpegEncoder::new_with_quality(&mut output, quality)
                .encode_image(&rgb)
                .map_err(|error| ImageError::Codec(error.to_string()))?;
        }
    }
    let output = output.into_inner();
    if output.len() > max_output_bytes {
        return Err(ImageError::InputLimit(format!(
            "encoded output is {} bytes, above the {max_output_bytes}-byte limit",
            output.len()
        )));
    }
    Ok(output)
}

fn validate_dimensions(width: u32, height: u32, limits: ImageIoLimits) -> Result<(), ImageError> {
    if width == 0 || height == 0 || width > limits.max_width || height > limits.max_height {
        return Err(ImageError::InputLimit(format!(
            "decoded dimensions {width}x{height} exceed the configured bounds"
        )));
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| ImageError::InputLimit("decoded pixel count overflowed".to_string()))?;
    if pixels > limits.max_pixels {
        return Err(ImageError::InputLimit(format!(
            "decoded image has {pixels} pixels, above the {}-pixel limit",
            limits.max_pixels
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_png_with_limits() {
        let image = DecodedImage::new_rgba8(2, 1, vec![1, 2, 3, 255, 4, 5, 6, 255]).unwrap();
        let encoded = encode_image(&image, ImageOutputFormat::Png, 90, 1024 * 1024).unwrap();
        let decoded = decode_image(&encoded, ImageIoLimits::default()).unwrap();
        assert_eq!(decoded, image);
    }

    #[test]
    fn rejects_encoded_byte_limit_before_decode() {
        let error = decode_image(
            &[0; 8],
            ImageIoLimits {
                max_encoded_bytes: 4,
                ..ImageIoLimits::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), crate::ImageErrorKind::InputLimit);
    }

    #[test]
    fn jpeg_exif_orientation_is_applied_before_dimension_validation() {
        let source = DecodedImage::new_rgba8(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255]).unwrap();
        let jpeg = encode_image(&source, ImageOutputFormat::Jpeg, 100, 1024 * 1024).unwrap();
        let jpeg = with_exif_orientation(jpeg, 6);
        let decoded = decode_image(
            &jpeg,
            ImageIoLimits {
                max_width: 1,
                max_height: 2,
                ..ImageIoLimits::default()
            },
        )
        .unwrap();
        assert_eq!([decoded.width(), decoded.height()], [1, 2]);
    }

    #[test]
    fn embedded_icc_profiles_are_rejected_instead_of_silently_stripped() {
        use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};

        let mut encoded = Vec::new();
        let mut encoder = PngEncoder::new(&mut encoded);
        encoder.set_icc_profile(vec![0; 128]).unwrap();
        encoder
            .write_image(&[1, 2, 3, 255], 1, 1, ExtendedColorType::Rgba8)
            .unwrap();
        let error = decode_image(&encoded, ImageIoLimits::default()).unwrap_err();
        assert_eq!(error.kind(), crate::ImageErrorKind::Codec);
        assert!(error
            .to_string()
            .contains("convert the image to sRGB before input"));
    }

    fn with_exif_orientation(mut jpeg: Vec<u8>, orientation: u16) -> Vec<u8> {
        assert_eq!(&jpeg[..2], [0xff, 0xd8]);
        let mut payload = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
        payload.extend_from_slice(&orientation.to_le_bytes());
        payload.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let segment_length = u16::try_from(payload.len() + 2).unwrap();
        let mut oriented = Vec::with_capacity(jpeg.len() + payload.len() + 4);
        oriented.extend_from_slice(&jpeg[..2]);
        oriented.extend_from_slice(&[0xff, 0xe1]);
        oriented.extend_from_slice(&segment_length.to_be_bytes());
        oriented.extend_from_slice(&payload);
        oriented.append(&mut jpeg.split_off(2));
        oriented
    }
}
