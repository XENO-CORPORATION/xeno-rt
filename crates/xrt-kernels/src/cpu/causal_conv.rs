//! Frame-major causal neural-audio convolution with SIMD channel reduction.
//! Packing retains FP32 weights and isolates Torch's ordinary/transpose layouts.
use rayon::prelude::*;
use xrt_core::{Result, XrtError};

pub struct PackedCausalConv1d {
    weight: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    transpose: bool,
    left: usize,
}
impl PackedCausalConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: &[f32],
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        transpose: bool,
        packed_budget: usize,
    ) -> Result<Self> {
        if input_channels == 0
            || output_channels == 0
            || kernel == 0
            || stride == 0
            || dilation == 0
            || groups == 0
            || input_channels % groups != 0
            || output_channels % groups != 0
            || (transpose && (groups != 1 || dilation != 1))
        {
            return Err(XrtError::Shape(
                "invalid causal convolution geometry".into(),
            ));
        }
        let count = output_channels
            .checked_mul(input_channels / groups)
            .and_then(|n| n.checked_mul(kernel))
            .ok_or_else(|| XrtError::Shape("causal convolution weight size overflow".into()))?;
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| XrtError::Shape("causal convolution byte size overflow".into()))?;
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| XrtError::Shape("causal convolution kernel overflow".into()))?;
        if effective < stride || weight.len() != count || bytes > packed_budget {
            return Err(XrtError::Shape(
                "causal convolution storage/budget mismatch".into(),
            ));
        }
        if weight.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "non-finite causal convolution weight".into(),
            ));
        }
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(count)
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        packed.resize(count, 0.0);
        let per_group = input_channels / groups;
        for o in 0..output_channels {
            for k in 0..kernel {
                for i in 0..per_group {
                    let source = if transpose {
                        (i * output_channels + o) * kernel + k
                    } else {
                        (o * per_group + i) * kernel + k
                    };
                    packed[(o * kernel + k) * per_group + i] = weight[source];
                }
            }
        }
        Ok(Self {
            weight: packed,
            input_channels,
            output_channels,
            kernel,
            stride,
            dilation,
            groups,
            transpose,
            left: effective - stride,
        })
    }
    pub fn output_frames(&self, input_frames: usize) -> Result<usize> {
        if input_frames == 0 {
            return Err(XrtError::Shape("empty causal convolution sequence".into()));
        }
        // Bound the largest address expression before entering the parallel loop.
        input_frames
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.left))
            .ok_or_else(|| XrtError::Shape("causal convolution time overflow".into()))?;
        Ok(if self.transpose {
            input_frames * self.stride
        } else {
            input_frames.div_ceil(self.stride)
        })
    }
    pub fn apply(
        &self,
        input: &[f32],
        input_frames: usize,
        bias: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let frames = self.output_frames(input_frames)?;
        let input_len = input_frames
            .checked_mul(self.input_channels)
            .ok_or_else(|| XrtError::Shape("causal convolution input overflow".into()))?;
        let output_len = frames
            .checked_mul(self.output_channels)
            .ok_or_else(|| XrtError::Shape("causal convolution output overflow".into()))?;
        if input.len() != input_len
            || output.len() != output_len
            || bias.len() != self.output_channels
        {
            return Err(XrtError::Shape(
                "causal convolution activation dimensions".into(),
            ));
        }
        if input.iter().chain(bias).any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "non-finite causal convolution input".into(),
            ));
        }
        let per_group = self.input_channels / self.groups;
        output
            .par_chunks_mut(self.output_channels)
            .enumerate()
            .for_each(|(time, row)| {
                for (o, value) in row.iter_mut().enumerate() {
                    let mut sum = bias[o];
                    let offset = (o / (self.output_channels / self.groups)) * per_group;
                    for k in 0..self.kernel {
                        let source = if self.transpose {
                            if time < k || (time - k) % self.stride != 0 {
                                continue;
                            }
                            (time - k) / self.stride
                        } else {
                            let padded = time * self.stride + k * self.dilation;
                            if padded < self.left {
                                continue;
                            }
                            padded - self.left
                        };
                        if source >= input_frames {
                            continue;
                        }
                        let first = source * self.input_channels + offset;
                        let weight_first = (o * self.kernel + k) * per_group;
                        sum += super::dot(
                            &input[first..first + per_group],
                            &self.weight[weight_first..weight_first + per_group],
                        );
                    }
                    *value = sum;
                }
            });
        if output.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "causal convolution result overflow".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_bad_geometry_budget_and_activation_without_output_mutation() {
        assert!(PackedCausalConv1d::new(&[1.0; 6], 2, 1, 3, 1, 1, 1, false, 8).is_err());
        assert!(PackedCausalConv1d::new(&[1.0; 6], 2, 1, 3, 1, 1, 2, true, 24).is_err());
        let conv = PackedCausalConv1d::new(&[1.0; 6], 2, 1, 3, 1, 1, 1, false, 24).unwrap();
        let mut out = [7.0; 2];
        assert!(conv.apply(&[f32::NAN; 4], 2, &[0.0], &mut out).is_err());
        assert_eq!(out, [7.0; 2]);
        assert!(conv.output_frames(usize::MAX).is_err());
    }
}
