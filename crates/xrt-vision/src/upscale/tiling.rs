use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use ndarray::Array4;

use crate::VisionError;

/// Convert an RGB image slice into an NCHW f32 tensor with values normalized to [0, 1].
pub(super) fn image_to_tensor(
    img: &DynamicImage,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<Array4<f32>, VisionError> {
    let rgb = img.to_rgb8();
    let mut tensor = Array4::<f32>::zeros((1, 3, crop_h as usize, crop_w as usize));

    for y in 0..crop_h {
        for x in 0..crop_w {
            let px = rgb.get_pixel(crop_x + x, crop_y + y);
            tensor[[0, 0, y as usize, x as usize]] = px[0] as f32 / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = px[1] as f32 / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = px[2] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

/// Convert an NCHW f32 tensor output back into an RGB image buffer.
pub(super) fn tensor_to_rgb_image(
    tensor: &Array4<f32>,
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, VisionError> {
    let shape = tensor.shape();
    let (_, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    if c < 3 {
        return Err(VisionError::Inference(format!(
            "expected at least 3 channels in output tensor, got {c}"
        )));
    }

    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let r = (tensor[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (tensor[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (tensor[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(img)
}

/// If the original image had an alpha channel, upscale the alpha channel (via bilinear/bicubic)
/// and merge it with the upscaled RGB result.
pub(super) fn merge_alpha_if_present(
    original: &DynamicImage,
    upscaled_rgb: ImageBuffer<Rgb<u8>, Vec<u8>>,
    target_width: u32,
    target_height: u32,
) -> DynamicImage {
    if original.color().has_alpha() {
        let rgba_orig = original.to_rgba8();
        let mut alpha_plane = ImageBuffer::<image::Luma<u8>, Vec<u8>>::new(original.width(), original.height());
        for (x, y, pixel) in rgba_orig.enumerate_pixels() {
            alpha_plane.put_pixel(x, y, image::Luma([pixel[3]]));
        }

        let upscaled_alpha = image::imageops::resize(
            &alpha_plane,
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        );

        let mut final_rgba = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(target_width, target_height);
        for y in 0..target_height {
            for x in 0..target_width {
                let rgb_px = upscaled_rgb.get_pixel(x, y);
                let a_px = upscaled_alpha.get_pixel(x, y);
                final_rgba.put_pixel(x, y, Rgba([rgb_px[0], rgb_px[1], rgb_px[2], a_px[0]]));
            }
        }
        DynamicImage::ImageRgba8(final_rgba)
    } else {
        DynamicImage::ImageRgb8(upscaled_rgb)
    }
}
