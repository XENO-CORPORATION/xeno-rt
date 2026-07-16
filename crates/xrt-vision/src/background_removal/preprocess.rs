use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn image_to_tensor(image: &DynamicImage, target_size: (u32, u32)) -> Array4<f32> {
    let (target_width, target_height) = target_size;
    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                tensor[[0, channel, y, x]] = (value - MEAN[channel]) / STD[channel];
            }
        }
    }
    tensor
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage};

    use super::*;

    #[test]
    fn image_to_tensor_has_birefnet_layout_and_finite_values() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 3, image::Rgb([64, 128, 192])));
        let tensor = image_to_tensor(&image, (8, 6));
        assert_eq!(tensor.shape(), &[1, 3, 6, 8]);
        assert!(tensor.iter().all(|value| value.is_finite()));
    }
}
