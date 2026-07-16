use image::{DynamicImage, RgbaImage};
use ndarray::Array2;
use rayon::prelude::*;

pub(super) fn apply_mask(
    original: &DynamicImage,
    mask: &Array2<f32>,
    original_width: u32,
    original_height: u32,
    _threshold: f32,
) -> DynamicImage {
    let resized_mask = resize_mask(mask, original_width as usize, original_height as usize);
    let rgba = original.to_rgba8();
    let refined_mask = guided_filter(&rgba, &resized_mask, 15, 0.01);
    DynamicImage::ImageRgba8(apply_mask_to_image(&rgba, &refined_mask))
}

fn guided_filter(guide: &RgbaImage, mask: &Array2<f32>, radius: i32, epsilon: f32) -> Array2<f32> {
    let width = guide.width() as usize;
    let height = guide.height() as usize;
    let mut guide_gray = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let pixel = guide.get_pixel(x as u32, y as u32);
            guide_gray[[y, x]] =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
                    / 255.0;
        }
    }

    let mean_guide = box_filter(&guide_gray, radius);
    let mean_mask = box_filter(mask, radius);
    let guide_mask = &guide_gray * mask;
    let guide_squared = &guide_gray * &guide_gray;
    let mean_guide_mask = box_filter(&guide_mask, radius);
    let mean_guide_squared = box_filter(&guide_squared, radius);
    let covariance = mean_guide_mask - &mean_guide * &mean_mask;
    let variance = mean_guide_squared - &mean_guide * &mean_guide;
    let coefficient_a = covariance / variance.mapv(|value| value + epsilon);
    let coefficient_b = mean_mask - &coefficient_a * &mean_guide;
    let mean_a = box_filter(&coefficient_a, radius);
    let mean_b = box_filter(&coefficient_b, radius);
    (mean_a * guide_gray + mean_b).mapv(|value| value.clamp(0.0, 1.0))
}

fn box_filter(input: &Array2<f32>, radius: i32) -> Array2<f32> {
    let (height, width) = input.dim();
    let mut integral = Array2::<f64>::zeros((height + 1, width + 1));
    for y in 0..height {
        for x in 0..width {
            integral[[y + 1, x + 1]] =
                input[[y, x]] as f64 + integral[[y, x + 1]] + integral[[y + 1, x]]
                    - integral[[y, x]];
        }
    }

    let mut output = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let y0 = (y as i32 - radius).max(0) as usize;
            let y1 = ((y as i32 + radius + 1) as usize).min(height);
            let x0 = (x as i32 - radius).max(0) as usize;
            let x1 = ((x as i32 + radius + 1) as usize).min(width);
            let area = ((y1 - y0) * (x1 - x0)) as f64;
            let sum =
                integral[[y1, x1]] - integral[[y0, x1]] - integral[[y1, x0]] + integral[[y0, x0]];
            output[[y, x]] = (sum / area) as f32;
        }
    }
    output
}

fn resize_mask(mask: &Array2<f32>, target_width: usize, target_height: usize) -> Array2<f32> {
    let (source_height, source_width) = mask.dim();
    if source_width == target_width && source_height == target_height {
        return mask.clone();
    }

    let mut resized = Array2::<f32>::zeros((target_height, target_width));
    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;
    resized
        .as_slice_mut()
        .expect("zeroed ndarray is contiguous")
        .par_chunks_mut(target_width)
        .enumerate()
        .for_each(|(y, row)| {
            let source_y = y as f32 * scale_y;
            let y0 = (source_y.floor() as usize).min(source_height - 1);
            let y1 = (y0 + 1).min(source_height - 1);
            let y_fraction = source_y - y0 as f32;
            for (x, pixel) in row.iter_mut().enumerate() {
                let source_x = x as f32 * scale_x;
                let x0 = (source_x.floor() as usize).min(source_width - 1);
                let x1 = (x0 + 1).min(source_width - 1);
                let x_fraction = source_x - x0 as f32;
                let top = mask[[y0, x0]] * (1.0 - x_fraction) + mask[[y0, x1]] * x_fraction;
                let bottom = mask[[y1, x0]] * (1.0 - x_fraction) + mask[[y1, x1]] * x_fraction;
                *pixel = top * (1.0 - y_fraction) + bottom * y_fraction;
            }
        });
    resized
}

fn apply_mask_to_image(image: &RgbaImage, mask: &Array2<f32>) -> RgbaImage {
    let width = image.width();
    let height = image.height();
    let input = image.as_raw();
    let mut output = vec![0; (width * height * 4) as usize];
    output
        .par_chunks_mut((width * 4) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let input_row = y * width as usize * 4;
            for x in 0..width as usize {
                let pixel = x * 4;
                let source = input_row + pixel;
                row[pixel] = input[source];
                row[pixel + 1] = input[source + 1];
                row[pixel + 2] = input[source + 2];
                row[pixel + 3] = (mask[[y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });
    RgbaImage::from_raw(width, height, output).expect("RGBA output size matches dimensions")
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};

    use super::*;

    #[test]
    fn apply_mask_preserves_dimensions_and_rgb() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 3, Rgba([10, 20, 30, 255])));
        let mask = Array2::from_elem((2, 2), 1.0);
        let output = apply_mask(&image, &mask, 4, 3, 0.1).to_rgba8();
        assert_eq!(output.dimensions(), (4, 3));
        assert!(output.pixels().all(|pixel| pixel.0 == [10, 20, 30, 255]));
    }
}
