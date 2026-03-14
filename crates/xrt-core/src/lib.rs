use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::io;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, XrtError>;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl DType {
    pub fn from_ggml_type_id(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            30 => Ok(Self::BF16),
            _ => Err(XrtError::Unsupported(format!(
                "unsupported GGML tensor type id {value}"
            ))),
        }
    }

    pub fn ggml_type_id(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 30,
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q8_0 | Self::Q4_0 | Self::Q4_K | Self::Q5_K | Self::Q6_K
        )
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            _ => None,
        }
    }

    pub fn storage_len(self, shape: &[usize]) -> Result<usize> {
        if shape.is_empty() {
            return Ok(0);
        }

        let first_dim = shape[0];
        let block = self.block_size();
        if first_dim % block != 0 {
            return Err(XrtError::Shape(format!(
                "tensor first dimension {first_dim} is not divisible by block size {block} for {self:?}"
            )));
        }

        let rows = if shape.len() == 1 {
            Ok(1usize)
        } else {
            shape[1..]
                .iter()
                .try_fold(1usize, |acc, dim| checked_mul(acc, *dim, "tensor rows"))
        }?;
        let blocks_per_row = first_dim / block;
        let row_bytes = checked_mul(blocks_per_row, self.block_bytes(), "row bytes")?;
        checked_mul(rows, row_bytes, "tensor bytes")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda { ordinal: usize },
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub shape: &'a [usize],
    pub stride: &'a [usize],
    pub dtype: DType,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(
        shape: &'a [usize],
        stride: &'a [usize],
        dtype: DType,
        data: &'a [u8],
    ) -> Result<Self> {
        if shape.len() != stride.len() {
            return Err(XrtError::Shape(format!(
                "shape rank {} does not match stride rank {}",
                shape.len(),
                stride.len()
            )));
        }

        let expected = dtype.storage_len(shape)?;
        if data.len() < expected {
            return Err(XrtError::InvalidTensor(format!(
                "tensor data is truncated: expected at least {expected} bytes, found {}",
                data.len()
            )));
        }

        Ok(Self {
            shape,
            stride,
            dtype,
            data,
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for (dim, stride) in self.shape.iter().zip(self.stride.iter()) {
            if *stride != expected {
                return false;
            }
            expected = expected.saturating_mul(*dim);
        }
        true
    }
}

pub trait KvCache {
    fn layers(&self) -> usize;
    fn width(&self) -> usize;
    fn len(&self, layer: usize) -> usize;
    fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()>;
    fn key(&self, layer: usize, position: usize) -> Option<&[f32]>;
    fn value(&self, layer: usize, position: usize) -> Option<&[f32]>;
    fn clear(&mut self);

    /// Truncate all layers to `new_len` positions.
    /// Used by speculative decoding to roll back rejected draft tokens.
    fn truncate(&mut self, new_len: usize);

    /// Append `count` (key, value) pairs to the given layer in one call.
    /// `keys` and `values` are concatenated vectors of length `count * width`.
    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        let w = self.width();
        for i in 0..count {
            self.append(layer, &keys[i * w..(i + 1) * w], &values[i * w..(i + 1) * w])?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum XrtError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("utf8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    #[error("shape error: {0}")]
    Shape(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("cuda error: {0}")]
    Cuda(String),
}

pub fn checked_mul(lhs: usize, rhs: usize, what: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        XrtError::InvalidFormat(format!("overflow while computing {what}: {lhs} * {rhs}"))
    })
}

pub fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(XrtError::InvalidFormat(format!(
            "alignment must be a power of two, got {alignment}"
        )));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|aligned| aligned & !mask)
        .ok_or_else(|| XrtError::InvalidFormat(format!("overflow while aligning {value}")))
}

pub fn decode_f16(bytes: &[u8]) -> Result<f32> {
    if bytes.len() < 2 {
        return Err(XrtError::InvalidTensor(
            "not enough bytes for f16 element".to_string(),
        ));
    }
    Ok(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
}

pub fn decode_bf16(bytes: &[u8]) -> Result<f32> {
    if bytes.len() < 2 {
        return Err(XrtError::InvalidTensor(
            "not enough bytes for bf16 element".to_string(),
        ));
    }
    Ok(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
}
