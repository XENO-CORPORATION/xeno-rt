//! Native tensor-only causal DAC decoding. CPU/F32 reference precision.
//! No pickle, Python process, or external media engine is used at inference.
use super::backbone::rmsnorm;
use crate::control::InferenceControl;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::path::Path;
use xrt_core::{Result, XrtError};
use xrt_kernels::cpu::{linear_f32_bytes, softmax_inplace};
use xrt_safetensors::{SafeTensorDType, SafeTensorLayout, SafeTensorStore};

#[derive(Clone)]
pub struct Sequence {
    pub frames: usize,
    pub channels: usize,
    pub values: Vec<f32>,
}
impl Sequence {
    fn new(frames: usize, channels: usize, budget: usize) -> Result<Self> {
        let len = frames
            .checked_mul(channels)
            .ok_or_else(|| XrtError::Shape("codec activation size overflow".into()))?;
        let bytes = len
            .checked_mul(4)
            .ok_or_else(|| XrtError::Shape("codec activation bytes overflow".into()))?;
        if bytes > budget {
            return Err(XrtError::Runtime(format!(
                "codec activation needs {bytes} bytes, limit {budget}"
            )));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        values.resize(len, 0.0);
        Ok(Self {
            frames,
            channels,
            values,
        })
    }
    fn row(&self, frame: usize) -> &[f32] {
        &self.values[frame * self.channels..][..self.channels]
    }
}
pub struct Codec {
    store: SafeTensorStore,
    pub working_bytes: usize,
    encoder: bool,
}
pub struct Decoded {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub clamped_residual_codes: usize,
}
pub struct EncodedReference {
    pub frames: Vec<Vec<u32>>,
    pub original_samples: usize,
    pub padded_samples: usize,
    pub sample_rate: u32,
}
impl Codec {
    pub fn open(root: &Path, working_bytes: usize) -> Result<Self> {
        if working_bytes < 1024 * 1024 {
            return Err(XrtError::Runtime("codec working limit below 1 MiB".into()));
        }
        let p = root.join("codec.json");
        if std::fs::metadata(&p)?.len() > 64 * 1024 {
            return Err(XrtError::InvalidMetadata(
                "codec metadata exceeds bounds".into(),
            ));
        }
        let info: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)
            .map_err(|e| XrtError::InvalidMetadata(e.to_string()))?;
        if info["family"] != "fish-dac-drama3"
            || info["schema"] != 1
            || info["sample_rate"] != 44100
            || info["frame_samples"] != 2048
            || info["precision"] != "f32"
            || info["source_sha256"]
                != "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
        {
            return Err(XrtError::InvalidMetadata(
                "unsupported native codec identity/configuration".into(),
            ));
        }
        let expected = info["converted_sha256"]
            .as_str()
            .ok_or_else(|| XrtError::InvalidMetadata("codec checksum missing".into()))?;
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(XrtError::InvalidMetadata("invalid codec checksum".into()));
        }
        let encoder = info["encoder"].as_bool().unwrap_or(false);
        // The metadata beside a file cannot vouch for its own provenance.
        // Admit only conversions whose actual trained tensors were qualified.
        validate_conversion(expected, encoder)?;
        let weights = root.join("model.safetensors");
        if std::fs::metadata(&weights)?.len() > 2 * 1024 * 1024 * 1024 {
            return Err(XrtError::Runtime(
                "converted codec exceeds 2 GiB bound".into(),
            ));
        }
        use std::io::Read;
        let mut file = std::fs::File::open(weights)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        if format!("{:x}", hash.finalize()) != expected.to_ascii_lowercase() {
            return Err(XrtError::InvalidTensor(
                "converted codec SHA-256 mismatch".into(),
            ));
        }
        Ok(Self {
            store: SafeTensorStore::open_exact(
                root,
                SafeTensorLayout::single("model.safetensors"),
            )?,
            working_bytes,
            encoder,
        })
    }
    fn tensor(&self, name: &str) -> Result<xrt_safetensors::SafeTensorView<'_>> {
        let v = self.store.require_tensor(name)?;
        if v.info.dtype != SafeTensorDType::F32 {
            return Err(XrtError::Unsupported(format!(
                "codec tensor dtype for {name}"
            )));
        }
        Ok(v)
    }
    fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let v = self.tensor(name)?;
        if v.data.len() > self.working_bytes {
            return Err(XrtError::Runtime(
                "codec parameter expansion exceeds limit".into(),
            ));
        }
        let out: Vec<_> = v
            .data
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        check(&out)?;
        Ok(out)
    }
    fn linear(&self, input: &Sequence, name: &str) -> Result<Sequence> {
        let w = self.tensor(&format!("{name}.weight"))?;
        if w.info.shape.len() != 2 || w.info.shape[1] != input.channels {
            return Err(XrtError::Shape(format!("codec linear geometry for {name}")));
        }
        let mut out = Sequence::new(input.frames, w.info.shape[0], self.working_bytes)?;
        let b = if self.store.tensor_info(&format!("{name}.bias")).is_some() {
            Some(self.vector(&format!("{name}.bias"))?)
        } else {
            None
        };
        linear_f32_bytes(
            &input.values,
            input.frames,
            input.channels,
            w.data,
            out.channels,
            b.as_deref(),
            &mut out.values,
        )?;
        check(&out.values)?;
        Ok(out)
    }
    fn conv(
        &self,
        input: &Sequence,
        name: &str,
        stride: usize,
        dilation: usize,
        groups: usize,
        transpose: bool,
        control: &InferenceControl,
    ) -> Result<Sequence> {
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        let w = self.tensor(&format!("{name}.weight"))?;
        let shape = &w.info.shape;
        if shape.len() != 3
            || stride == 0
            || dilation == 0
            || groups == 0
            || input.channels % groups != 0
        {
            return Err(XrtError::Shape(format!(
                "codec convolution geometry for {name}"
            )));
        }
        let channels = if transpose { shape[1] } else { shape[0] };
        let kernel = shape[2];
        if kernel == 0
            || channels == 0
            || channels % groups != 0
            || (!transpose && shape[1] != input.channels / groups)
            || (transpose && (shape[0] != input.channels || groups != 1 || dilation != 1))
        {
            return Err(XrtError::Shape(format!(
                "codec convolution shape for {name}"
            )));
        }
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| XrtError::Shape("codec kernel overflow".into()))?;
        if effective < stride {
            return Err(XrtError::Shape("codec kernel smaller than stride".into()));
        }
        let frames = if transpose {
            input
                .frames
                .checked_mul(stride)
                .ok_or_else(|| XrtError::Shape("codec transpose length overflow".into()))?
        } else {
            input.frames.div_ceil(stride)
        };
        let mut out = Sequence::new(frames, channels, self.working_bytes)?;
        let weight = self.vector(&format!("{name}.weight"))?;
        let bias = self.vector(&format!("{name}.bias"))?;
        if bias.len() != channels {
            return Err(XrtError::Shape("codec bias dimensions".into()));
        }
        let packed = xrt_kernels::cpu::causal_conv::PackedCausalConv1d::new(
            &weight,
            input.channels,
            channels,
            kernel,
            stride,
            dilation,
            groups,
            transpose,
            self.working_bytes.min(128 * 1024 * 1024),
        )?;
        packed.apply(&input.values, input.frames, &bias, &mut out.values)?;
        check(&out.values)?;
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        Ok(out)
    }
    fn snake(&self, input: &Sequence, name: &str) -> Result<Sequence> {
        let alpha = self.vector(&format!("{name}.alpha"))?;
        if alpha.len() != input.channels {
            return Err(XrtError::Shape("codec Snake alpha width".into()));
        }
        let mut out = input.clone();
        for row in out.values.chunks_exact_mut(input.channels) {
            for (x, a) in row.iter_mut().zip(&alpha) {
                *x += (*a * *x).sin().powi(2) / (*a + 1e-9);
            }
        }
        check(&out.values)?;
        Ok(out)
    }
    fn residual(
        &self,
        input: &Sequence,
        prefix: &str,
        dilation: usize,
        control: &InferenceControl,
    ) -> Result<Sequence> {
        let out = self.snake(input, &format!("{prefix}.block.0"))?;
        let out = self.conv(
            &out,
            &format!("{prefix}.block.1.conv"),
            1,
            dilation,
            1,
            false,
            control,
        )?;
        let out = self.snake(&out, &format!("{prefix}.block.2"))?;
        let mut out = self.conv(
            &out,
            &format!("{prefix}.block.3.conv"),
            1,
            1,
            1,
            false,
            control,
        )?;
        add_sequence(&mut out, input)?;
        Ok(out)
    }
    fn convnext(
        &self,
        input: &Sequence,
        prefix: &str,
        control: &InferenceControl,
    ) -> Result<Sequence> {
        let mut out = self.conv(
            input,
            &format!("{prefix}.dwconv.conv"),
            1,
            1,
            input.channels,
            false,
            control,
        )?;
        let weights = self.vector(&format!("{prefix}.norm.weight"))?;
        let bias = self.vector(&format!("{prefix}.norm.bias"))?;
        if weights.len() != out.channels || bias.len() != out.channels {
            return Err(XrtError::Shape("codec LayerNorm geometry".into()));
        }
        for row in out.values.chunks_exact_mut(out.channels) {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            for ((x, w), b) in row.iter_mut().zip(&weights).zip(&bias) {
                *x = (*x - mean) / (variance + 1e-6).sqrt() * w + b;
            }
        }
        let mut out = self.linear(&out, &format!("{prefix}.pwconv1"))?;
        for x in &mut out.values {
            *x = 0.5 * *x * (1.0 + libm::erff(*x / std::f32::consts::SQRT_2));
        }
        let mut out = self.linear(&out, &format!("{prefix}.pwconv2"))?;
        let gamma = self.vector(&format!("{prefix}.gamma"))?;
        if gamma.len() != out.channels {
            return Err(XrtError::Shape("codec layer-scale geometry".into()));
        }
        for row in out.values.chunks_exact_mut(out.channels) {
            for (x, g) in row.iter_mut().zip(&gamma) {
                *x *= g;
            }
        }
        add_sequence(&mut out, input)?;
        Ok(out)
    }
    fn transformer(
        &self,
        input: &Sequence,
        module: &str,
        layers: usize,
        window: usize,
        control: &InferenceControl,
    ) -> Result<Sequence> {
        let mut current = input.clone();
        let width = 1024;
        let heads = 16;
        let dim = 64;
        if current.channels != width || current.frames > 4096 {
            return Err(XrtError::Shape(
                "codec post-transformer dimensions/context".into(),
            ));
        }
        for layer in 0..layers {
            control
                .check()
                .map_err(|e| XrtError::Runtime(e.to_string()))?;
            let prefix = format!("{module}.layers.{layer}");
            let n = self.vector(&format!("{prefix}.attention_norm.weight"))?;
            let mut normalized = current.clone();
            for row in normalized.values.chunks_exact_mut(width) {
                row.copy_from_slice(&rmsnorm(row, &n, 1e-5, false)?);
            }
            let packed = self.linear(&normalized, &format!("{prefix}.attention.wqkv"))?;
            if packed.channels != 3 * width {
                return Err(XrtError::Shape("codec QKV packing".into()));
            }
            let mut q = Sequence::new(input.frames, width, self.working_bytes)?;
            let mut k = q.clone();
            let mut v = q.clone();
            for time in 0..input.frames {
                q.values[time * width..][..width].copy_from_slice(&packed.row(time)[..width]);
                k.values[time * width..][..width]
                    .copy_from_slice(&packed.row(time)[width..2 * width]);
                v.values[time * width..][..width].copy_from_slice(&packed.row(time)[2 * width..]);
            }
            for seq in [&mut q, &mut k] {
                for (time, row) in seq.values.chunks_exact_mut(width).enumerate() {
                    for head in row.chunks_exact_mut(dim) {
                        for pair in 0..dim / 2 {
                            let angle =
                                time as f32 / 10000.0f32.powf((2 * pair) as f32 / dim as f32);
                            let (s, c) = angle.sin_cos();
                            let a = head[2 * pair];
                            let b = head[2 * pair + 1];
                            head[2 * pair] = a * c - b * s;
                            head[2 * pair + 1] = b * c + a * s;
                        }
                    }
                }
            }
            let mut attended = Sequence::new(input.frames, width, self.working_bytes)?;
            attended
                .values
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(time, row)| {
                    let start = (time + 1).saturating_sub(window);
                    for head in 0..heads {
                        let query = &q.row(time)[head * dim..][..dim];
                        let mut scores: Vec<_> = (start..=time)
                            .map(|t| {
                                query
                                    .iter()
                                    .zip(&k.row(t)[head * dim..][..dim])
                                    .map(|(a, b)| a * b)
                                    .sum::<f32>()
                                    / (dim as f32).sqrt()
                            })
                            .collect();
                        softmax_inplace(&mut scores);
                        for (n, p) in scores.iter().enumerate() {
                            for d in 0..dim {
                                row[head * dim + d] +=
                                    p * v.values[(start + n) * width + head * dim + d];
                            }
                        }
                    }
                });
            let mut attention = self.linear(&attended, &format!("{prefix}.attention.wo"))?;
            scale_channels(
                &mut attention,
                &self.vector(&format!("{prefix}.attention_layer_scale.gamma"))?,
            )?;
            add_sequence(&mut current, &attention)?;
            let n = self.vector(&format!("{prefix}.ffn_norm.weight"))?;
            let mut normalized = current.clone();
            for row in normalized.values.chunks_exact_mut(width) {
                row.copy_from_slice(&rmsnorm(row, &n, 1e-5, false)?);
            }
            let p = format!("{prefix}.feed_forward");
            let mut gate = self.linear(&normalized, &format!("{p}.w1"))?;
            let up = self.linear(&normalized, &format!("{p}.w3"))?;
            if gate.values.len() != up.values.len() {
                return Err(XrtError::Shape("codec gated MLP shape".into()));
            }
            for (x, u) in gate.values.iter_mut().zip(up.values) {
                *x = *x / (1.0 + (-*x).exp()) * u;
            }
            let mut ffn = self.linear(&gate, &format!("{p}.w2"))?;
            scale_channels(
                &mut ffn,
                &self.vector(&format!("{prefix}.ffn_layer_scale.gamma"))?,
            )?;
            add_sequence(&mut current, &ffn)?;
        }
        let norm = self.vector(&format!("{module}.norm.weight"))?;
        for row in current.values.chunks_exact_mut(width) {
            row.copy_from_slice(&rmsnorm(row, &norm, 1e-5, false)?);
        }
        Ok(current)
    }
    pub fn has_encoder(&self) -> bool {
        self.encoder
    }
    pub fn mapped_bytes(&self) -> Result<u64> {
        self.store.mapped_bytes()
    }
    pub fn decoding_workspace_bytes(frames: usize) -> Result<usize> {
        if frames == 0 || frames > 1024 {
            return Err(XrtError::Shape("codec frame bound is 1..1024".into()));
        }
        frames
            .checked_mul(2048 * 96 * 4 * 12)
            .and_then(|n| n.checked_add(256 * 1024 * 1024))
            .ok_or_else(|| XrtError::Shape("codec aggregate workspace overflow".into()))
    }
    /// Native reference conditioning. The caller supplies mono 44.1 kHz PCM;
    /// general media decoding/resampling remains outside the neural codec.
    pub fn encode(
        &self,
        samples: &[f32],
        sample_rate: u32,
        control: &InferenceControl,
    ) -> Result<EncodedReference> {
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        if !self.encoder {
            return Err(XrtError::Unsupported(
                "native codec bundle has no reference encoder".into(),
            ));
        }
        if sample_rate != 44100 || samples.is_empty() || samples.len() > 30 * 44100 {
            return Err(XrtError::Shape(
                "reference encoder requires non-empty mono 44.1 kHz PCM of at most 30 seconds"
                    .into(),
            ));
        }
        check(samples)?;
        let padded = samples.len().div_ceil(2048) * 2048;
        let required = padded
            .checked_mul(64 * 4 * 4)
            .and_then(|n| n.checked_add(256 * 1024 * 1024))
            .ok_or_else(|| XrtError::Shape("reference encoder workspace overflow".into()))?;
        if required > self.working_bytes {
            return Err(XrtError::Runtime(format!(
                "reference encoder workspace needs {required} bytes, budget {}",
                self.working_bytes
            )));
        }
        let mut seq = Sequence::new(padded, 1, self.working_bytes)?;
        seq.values[..samples.len()].copy_from_slice(samples);
        seq = self.conv(&seq, "encoder.block.0.conv", 1, 1, 1, false, control)?;
        for (i, stride) in [2, 4, 8, 8].into_iter().enumerate() {
            let p = format!("encoder.block.{}.block", i + 1);
            for (j, dilation) in [1, 3, 9].into_iter().enumerate() {
                seq = self.residual(&seq, &format!("{p}.{j}"), dilation, control)?;
            }
            seq = self.snake(&seq, &format!("{p}.3"))?;
            seq = self.conv(&seq, &format!("{p}.4.conv"), stride, 1, 1, false, control)?;
            if i == 3 {
                seq = self.transformer(&seq, &format!("{p}.5"), 4, 512, control)?;
            }
        }
        seq = self.snake(&seq, "encoder.block.5")?;
        seq = self.conv(&seq, "encoder.block.6.conv", 1, 1, 1, false, control)?;
        for i in 0..2 {
            seq = self.conv(
                &seq,
                &format!("quantizer.downsample.{i}.0.conv"),
                2,
                1,
                1,
                false,
                control,
            )?;
            seq = self.convnext(&seq, &format!("quantizer.downsample.{i}.1"), control)?;
        }
        let mut residual = self.transformer(&seq, "quantizer.pre_module", 8, 128, control)?;
        if residual.frames != padded / 2048 {
            return Err(XrtError::Shape(
                "native reference frame count mismatch".into(),
            ));
        }
        let mut frames = vec![vec![0u32; 10]; residual.frames];
        // Each codebook consumes the previous residual; iterate codebooks, not frames.
        #[allow(clippy::needless_range_loop)]
        for book in 0..10 {
            control
                .check()
                .map_err(|e| XrtError::Runtime(e.to_string()))?;
            let p = if book == 0 {
                "quantizer.semantic_quantizer.quantizers.0".to_owned()
            } else {
                format!("quantizer.quantizer.quantizers.{}", book - 1)
            };
            let projected =
                self.conv(&residual, &format!("{p}.in_proj"), 1, 1, 1, false, control)?;
            let raw = self.vector(&format!("{p}.codebook.weight"))?;
            let count = if book == 0 { 4096 } else { 1024 };
            if projected.channels != 8 || raw.len() != count * 8 {
                return Err(XrtError::Shape("reference VQ codebook geometry".into()));
            }
            let table = normalize_codebook(&raw, 8)?;
            let mut embedded = Sequence::new(residual.frames, 8, self.working_bytes)?;
            for (time, row) in projected.values.chunks_exact(8).enumerate() {
                let id = nearest_normalized_code(row, &table, 8)?;
                frames[time][book] = id as u32;
                embedded.values[time * 8..][..8].copy_from_slice(&raw[id * 8..][..8]);
            }
            let decoded =
                self.conv(&embedded, &format!("{p}.out_proj"), 1, 1, 1, false, control)?;
            if residual.values.len() != decoded.values.len() {
                return Err(XrtError::Shape("reference VQ residual dimensions".into()));
            }
            for (r, q) in residual.values.iter_mut().zip(decoded.values) {
                *r -= q;
            }
            check(&residual.values)?;
        }
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        Ok(EncodedReference {
            frames,
            original_samples: samples.len(),
            padded_samples: padded,
            sample_rate,
        })
    }
    pub fn decode(&self, frames: &[Vec<u32>], control: &InferenceControl) -> Result<Decoded> {
        if frames.is_empty()
            || frames.len() > 1024
            || frames
                .iter()
                .any(|f| f.len() != 10 || f.iter().any(|c| *c > 4095))
        {
            return Err(XrtError::Shape(
                "codec requires 1..1024 frames of 10 bounded code ids".into(),
            ));
        }
        // Conservative aggregate workspace: up to twelve simultaneous activation
        // buffers at the largest decoder stage plus bounded parameter expansion.
        // Mapped model weights are a separate residency reservation.
        let required = Self::decoding_workspace_bytes(frames.len())?;
        if required > self.working_bytes {
            return Err(XrtError::Runtime(format!(
                "codec aggregate workspace needs {required} bytes, budget {}",
                self.working_bytes
            )));
        }
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        let mut latent = Sequence::new(frames.len(), 1024, self.working_bytes)?;
        let mut clamped = 0;
        for book in 0..10 {
            let prefix = if book == 0 {
                "quantizer.semantic_quantizer.quantizers.0".to_owned()
            } else {
                format!("quantizer.quantizer.quantizers.{}", book - 1)
            };
            let cb = self.tensor(&format!("{prefix}.codebook.weight"))?;
            let shape = &cb.info.shape;
            if shape.len() != 2 || shape[1] != 8 || shape[0] != if book == 0 { 4096 } else { 1024 }
            {
                return Err(XrtError::Shape("codec codebook shape".into()));
            }
            let mut embedded = Sequence::new(frames.len(), 8, self.working_bytes)?;
            for (i, f) in frames.iter().enumerate() {
                let id = (f[book] as usize).min(shape[0] - 1);
                if id != f[book] as usize {
                    clamped += 1;
                }
                for d in 0..8 {
                    let off = (id * 8 + d) * 4;
                    embedded.values[i * 8 + d] =
                        f32::from_le_bytes(cb.data[off..off + 4].try_into().unwrap());
                }
            }
            let projected = self.conv(
                &embedded,
                &format!("{prefix}.out_proj"),
                1,
                1,
                1,
                false,
                control,
            )?;
            add_sequence(&mut latent, &projected)?;
        }
        let mut seq = self.transformer(&latent, "quantizer.post_module", 8, 128, control)?;
        for i in 0..2 {
            seq = self.conv(
                &seq,
                &format!("quantizer.upsample.{i}.0.conv"),
                2,
                1,
                1,
                true,
                control,
            )?;
            seq = self.convnext(&seq, &format!("quantizer.upsample.{i}.1"), control)?;
        }
        seq = self.conv(&seq, "decoder.model.0.conv", 1, 1, 1, false, control)?;
        for (i, stride) in [8, 8, 4, 2].into_iter().enumerate() {
            let prefix = format!("decoder.model.{}.block", i + 1);
            seq = self.snake(&seq, &format!("{prefix}.0"))?;
            seq = self.conv(
                &seq,
                &format!("{prefix}.1.conv"),
                stride,
                1,
                1,
                true,
                control,
            )?;
            for (j, dilation) in [1, 3, 9].into_iter().enumerate() {
                seq = self.residual(&seq, &format!("{prefix}.{}", j + 2), dilation, control)?;
            }
        }
        seq = self.snake(&seq, "decoder.model.5")?;
        seq = self.conv(&seq, "decoder.model.6.conv", 1, 1, 1, false, control)?;
        if seq.channels != 1 || seq.frames != frames.len() * 2048 {
            return Err(XrtError::Shape(
                "codec decoded sample count mismatch".into(),
            ));
        }
        for x in &mut seq.values {
            *x = x.tanh();
        }
        check(&seq.values)?;
        Ok(Decoded {
            samples: seq.values,
            sample_rate: 44100,
            clamped_residual_codes: clamped,
        })
    }
}
fn validate_conversion(digest: &str, encoder: bool) -> Result<()> {
    let qualified = if encoder {
        "fb52f9bce7c2235c46626044155ce8a6cdcd74a8507dfaee8cd56f570ac19ff9"
    } else {
        "30b123894028df60c880e668e91ffa49c16d19faa1077b75018d97893f82d464"
    };
    if !digest.eq_ignore_ascii_case(qualified) {
        return Err(XrtError::InvalidMetadata(
            "native codec conversion has no pinned numerical qualification".into(),
        ));
    }
    Ok(())
}
fn check(values: &[f32]) -> Result<()> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(XrtError::InvalidTensor(
            "non-finite native codec values".into(),
        ))
    } else {
        Ok(())
    }
}
fn normalize_codebook(table: &[f32], width: usize) -> Result<Vec<f32>> {
    if width == 0 || table.is_empty() || table.len() % width != 0 {
        return Err(XrtError::Shape("VQ normalization dimensions".into()));
    }
    check(table)?;
    let mut out = table.to_vec();
    for row in out.chunks_exact_mut(width) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        if !norm.is_finite() {
            return Err(XrtError::InvalidTensor("VQ normalization overflow".into()));
        }
        for v in row {
            *v /= norm;
        }
    }
    Ok(out)
}
fn nearest_normalized_code(input: &[f32], table: &[f32], width: usize) -> Result<usize> {
    if input.len() != width || width == 0 || table.is_empty() || table.len() % width != 0 {
        return Err(XrtError::Shape("VQ lookup dimensions".into()));
    }
    let normalized = normalize_codebook(input, width)?;
    let input_norm: f32 = normalized.iter().map(|v| v * v).sum();
    let mut best = (0, f32::INFINITY);
    for (i, code) in table.chunks_exact(width).enumerate() {
        let distance = input_norm - 2.0 * xrt_kernels::cpu::dot(&normalized, code)
            + code.iter().map(|v| v * v).sum::<f32>();
        if !distance.is_finite() {
            return Err(XrtError::InvalidTensor("VQ distance overflow".into()));
        }
        if distance < best.1 {
            best = (i, distance);
        }
    }
    Ok(best.0)
}

fn add_sequence(target: &mut Sequence, source: &Sequence) -> Result<()> {
    if target.frames != source.frames || target.channels != source.channels {
        return Err(XrtError::Shape("codec residual sequence shape".into()));
    }
    for (x, y) in target.values.iter_mut().zip(&source.values) {
        *x += y;
    }
    check(&target.values)
}
fn scale_channels(seq: &mut Sequence, scale: &[f32]) -> Result<()> {
    if seq.channels != scale.len() {
        return Err(XrtError::Shape("codec scale channels".into()));
    }
    for row in seq.values.chunks_exact_mut(seq.channels) {
        for (x, s) in row.iter_mut().zip(scale) {
            *x *= s;
        }
    }
    check(&seq.values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_self_declared_codec_checksum_cannot_establish_provenance() {
        assert!(validate_conversion(&"a".repeat(64), false).is_err());
        assert!(validate_conversion(
            "fb52f9bce7c2235c46626044155ce8a6cdcd74a8507dfaee8cd56f570ac19ff9",
            true
        )
        .is_ok());
        assert!(validate_conversion(
            "fb52f9bce7c2235c46626044155ce8a6cdcd74a8507dfaee8cd56f570ac19ff9",
            false
        )
        .is_err());
    }
    #[test]
    fn cosine_vq_is_scale_invariant_and_ties_select_first() {
        let raw = [2.0, 0.0, 0.0, 3.0, 2.0, 0.0];
        let table = normalize_codebook(&raw, 2).unwrap();
        assert_eq!(nearest_normalized_code(&[5.0, 0.0], &table, 2).unwrap(), 0);
        assert_eq!(nearest_normalized_code(&[0.0, 0.5], &table, 2).unwrap(), 1);
        assert_eq!(nearest_normalized_code(&[0.0, 0.0], &table, 2).unwrap(), 0);
        assert!(nearest_normalized_code(&[f32::NAN, 0.0], &table, 2).is_err());
        assert!(normalize_codebook(&[f32::MAX; 2], 2).is_err());
    }
}
