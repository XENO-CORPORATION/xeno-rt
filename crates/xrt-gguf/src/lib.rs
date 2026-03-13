use memmap2::{Mmap, MmapOptions};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
};
use xrt_core::{align_up, checked_mul, DType, Result, TensorView, XrtError};

const GGUF_MAGIC: u32 = 0x4647_5547;
const GGUF_VERSION_MAX: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const MAX_STRING_LEN: usize = 1 << 30;

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MetadataType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    UInt64,
    Int64,
    Float64,
}

impl MetadataType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(XrtError::InvalidMetadata(format!(
                "unknown metadata type tag {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetadataArray {
    pub element_type: MetadataType,
    pub values: Vec<MetadataValue>,
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_strings(&self) -> Option<Vec<&str>> {
        self.values.iter().map(MetadataValue::as_str).collect()
    }

    pub fn as_u32_vec(&self) -> Option<Vec<u32>> {
        self.values.iter().map(MetadataValue::to_u32).collect()
    }

    pub fn as_i32_vec(&self) -> Option<Vec<i32>> {
        self.values.iter().map(MetadataValue::to_i32).collect()
    }

    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        self.values.iter().map(MetadataValue::to_f32).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Self::UInt8(value) => Some(*value as u32),
            Self::UInt16(value) => Some(*value as u32),
            Self::UInt32(value) => Some(*value),
            Self::UInt64(value) => u32::try_from(*value).ok(),
            Self::Int8(value) if *value >= 0 => Some(*value as u32),
            Self::Int16(value) if *value >= 0 => Some(*value as u32),
            Self::Int32(value) if *value >= 0 => Some(*value as u32),
            Self::Int64(value) if *value >= 0 => u32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn to_i32(&self) -> Option<i32> {
        match self {
            Self::UInt8(value) => Some(*value as i32),
            Self::UInt16(value) => Some(*value as i32),
            Self::UInt32(value) => i32::try_from(*value).ok(),
            Self::UInt64(value) => i32::try_from(*value).ok(),
            Self::Int8(value) => Some(*value as i32),
            Self::Int16(value) => Some(*value as i32),
            Self::Int32(value) => Some(*value),
            Self::Int64(value) => i32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Self::UInt8(value) => Some(*value as u64),
            Self::UInt16(value) => Some(*value as u64),
            Self::UInt32(value) => Some(*value as u64),
            Self::UInt64(value) => Some(*value),
            Self::Int8(value) if *value >= 0 => Some(*value as u64),
            Self::Int16(value) if *value >= 0 => Some(*value as u64),
            Self::Int32(value) if *value >= 0 => Some(*value as u64),
            Self::Int64(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    pub fn to_usize(&self) -> Option<usize> {
        self.to_u64().and_then(|value| usize::try_from(value).ok())
    }

    pub fn to_f32(&self) -> Option<f32> {
        match self {
            Self::UInt8(value) => Some(*value as f32),
            Self::UInt16(value) => Some(*value as f32),
            Self::UInt32(value) => Some(*value as f32),
            Self::UInt64(value) => Some(*value as f32),
            Self::Int8(value) => Some(*value as f32),
            Self::Int16(value) => Some(*value as f32),
            Self::Int32(value) => Some(*value as f32),
            Self::Int64(value) => Some(*value as f32),
            Self::Float32(value) => Some(*value),
            Self::Float64(value) => Some(*value as f32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn row_len(&self) -> usize {
        self.dimensions.first().copied().unwrap_or_default()
    }

    pub fn rows(&self) -> usize {
        if self.dimensions.len() <= 1 {
            1
        } else {
            self.dimensions[1..].iter().copied().product()
        }
    }

    pub fn numel(&self) -> usize {
        self.dimensions.iter().copied().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
    header: GgufHeader,
    metadata: BTreeMap<String, MetadataValue>,
    tensor_infos: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    data_offset: usize,
    alignment: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mut cursor = Cursor::new(&mmap);

        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(XrtError::InvalidFormat(format!(
                "invalid GGUF magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
            )));
        }

        let version = cursor.read_u32()?;
        if version == 0 {
            return Err(XrtError::InvalidFormat(
                "GGUF version 0 is invalid".to_string(),
            ));
        }
        if version & 0x0000_ffff == 0 {
            return Err(XrtError::InvalidFormat(
                "GGUF version bytes look endian-swapped".to_string(),
            ));
        }
        if version > GGUF_VERSION_MAX {
            return Err(XrtError::Unsupported(format!(
                "GGUF version {version} is newer than supported version {GGUF_VERSION_MAX}"
            )));
        }

        let tensor_count = if version == 1 {
            cursor.read_u32()? as u64
        } else {
            cursor.read_u64()?
        };
        let metadata_kv_count = if version == 1 {
            cursor.read_u32()? as u64
        } else {
            cursor.read_u64()?
        };

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = cursor.read_string()?;
            if metadata.contains_key(&key) {
                return Err(XrtError::InvalidMetadata(format!(
                    "duplicate metadata key {key}"
                )));
            }
            let value = cursor.read_metadata_value()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::to_usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        if !alignment.is_power_of_two() {
            return Err(XrtError::InvalidMetadata(format!(
                "general.alignment must be a power of two, got {alignment}"
            )));
        }

        let tensor_cap = usize::try_from(tensor_count).map_err(|_| {
            XrtError::InvalidFormat(format!("tensor count {tensor_count} does not fit in usize"))
        })?;
        let mut tensor_infos = Vec::with_capacity(tensor_cap);
        let mut tensor_index = HashMap::with_capacity(tensor_cap);

        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            if tensor_index.contains_key(&name) {
                return Err(XrtError::InvalidTensor(format!(
                    "duplicate tensor name {name}"
                )));
            }

            let rank = cursor.read_u32()? as usize;
            if rank == 0 {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor {name} has zero rank"
                )));
            }

            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dim = usize::try_from(cursor.read_u64()?).map_err(|_| {
                    XrtError::InvalidTensor(format!(
                        "tensor {name} has a dimension that does not fit in usize"
                    ))
                })?;
                if dim == 0 {
                    return Err(XrtError::InvalidTensor(format!(
                        "tensor {name} contains a zero-sized dimension"
                    )));
                }
                dimensions.push(dim);
            }

            let dtype = DType::from_ggml_type_id(cursor.read_i32()?)?;
            let offset = usize::try_from(cursor.read_u64()?).map_err(|_| {
                XrtError::InvalidTensor(format!("tensor {name} offset does not fit in usize"))
            })?;
            if offset % alignment != 0 {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor {name} offset {offset} is not aligned to {alignment}"
                )));
            }

            let nbytes = dtype.storage_len(&dimensions)?;
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for dim in &dimensions {
                strides.push(stride);
                stride = checked_mul(stride, *dim, "tensor strides")?;
            }

            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                strides,
                dtype,
                offset,
                nbytes,
            };
            tensor_index.insert(name, tensor_infos.len());
            tensor_infos.push(info);
        }

        let data_offset = align_up(cursor.position(), alignment)?;
        if data_offset > mmap.len() {
            return Err(XrtError::InvalidFormat(format!(
                "GGUF data section starts at {data_offset}, past EOF {}",
                mmap.len()
            )));
        }

        for info in &tensor_infos {
            let start = data_offset.checked_add(info.offset).ok_or_else(|| {
                XrtError::InvalidTensor(format!("tensor {} offset overflow", info.name))
            })?;
            let end = start.checked_add(info.nbytes).ok_or_else(|| {
                XrtError::InvalidTensor(format!("tensor {} size overflow", info.name))
            })?;
            if end > mmap.len() {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor {} extends past EOF: [{}..{}) exceeds {}",
                    info.name,
                    start,
                    end,
                    mmap.len()
                )));
            }
        }

        Ok(Self {
            path,
            mmap,
            header: GgufHeader {
                version,
                tensor_count,
                metadata_kv_count,
            },
            metadata,
            tensor_infos,
            tensor_index,
            data_offset,
            alignment,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    pub fn require_metadata(&self, key: &str) -> Result<&MetadataValue> {
        self.metadata.get(key).ok_or_else(|| {
            XrtError::InvalidMetadata(format!("missing required metadata key {key}"))
        })
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(MetadataValue::as_str)
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(MetadataValue::as_bool)
    }

    pub fn metadata_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(MetadataValue::to_usize)
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(MetadataValue::to_f32)
    }

    pub fn metadata_array(&self, key: &str) -> Option<&MetadataArray> {
        self.metadata.get(key).and_then(MetadataValue::as_array)
    }

    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensor_infos
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_infos.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index
            .get(name)
            .and_then(|index| self.tensor_infos.get(*index))
    }

    pub fn require_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor_info(name)
            .ok_or_else(|| XrtError::InvalidTensor(format!("missing tensor {name}")))
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.require_tensor(name)?;
        let start = self
            .data_offset
            .checked_add(info.offset)
            .ok_or_else(|| XrtError::InvalidTensor(format!("tensor {name} offset overflow")))?;
        let end = start
            .checked_add(info.nbytes)
            .ok_or_else(|| XrtError::InvalidTensor(format!("tensor {name} size overflow")))?;
        Ok(&self.mmap[start..end])
    }

    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self.require_tensor(name)?;
        TensorView::new(
            &info.dimensions,
            &info.strides,
            info.dtype,
            self.tensor_data(name)?,
        )
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read_exact(&mut self, nbytes: usize) -> Result<&'a [u8]> {
        let end = self.position.checked_add(nbytes).ok_or_else(|| {
            XrtError::InvalidFormat(format!(
                "overflow while reading {nbytes} bytes at offset {}",
                self.position
            ))
        })?;
        if end > self.bytes.len() {
            return Err(XrtError::InvalidFormat(format!(
                "unexpected EOF while reading {nbytes} bytes at offset {}",
                self.position
            )));
        }
        let out = &self.bytes[self.position..end];
        self.position = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_i8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let length = usize::try_from(self.read_u64()?).map_err(|_| {
            XrtError::InvalidFormat("string length does not fit in usize".to_string())
        })?;
        if length > MAX_STRING_LEN {
            return Err(XrtError::InvalidFormat(format!(
                "string length {length} exceeds maximum {MAX_STRING_LEN}"
            )));
        }
        let bytes = self.read_exact(length)?;
        Ok(std::str::from_utf8(bytes)?.to_string())
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue> {
        let value_type = MetadataType::from_u32(self.read_u32()?)?;
        self.read_metadata_value_of(value_type)
    }

    fn read_metadata_value_of(&mut self, value_type: MetadataType) -> Result<MetadataValue> {
        Ok(match value_type {
            MetadataType::UInt8 => MetadataValue::UInt8(self.read_u8()?),
            MetadataType::Int8 => MetadataValue::Int8(self.read_i8()?),
            MetadataType::UInt16 => MetadataValue::UInt16(self.read_u16()?),
            MetadataType::Int16 => MetadataValue::Int16(self.read_i16()?),
            MetadataType::UInt32 => MetadataValue::UInt32(self.read_u32()?),
            MetadataType::Int32 => MetadataValue::Int32(self.read_i32()?),
            MetadataType::Float32 => MetadataValue::Float32(self.read_f32()?),
            MetadataType::Bool => MetadataValue::Bool(self.read_bool()?),
            MetadataType::String => MetadataValue::String(self.read_string()?),
            MetadataType::UInt64 => MetadataValue::UInt64(self.read_u64()?),
            MetadataType::Int64 => MetadataValue::Int64(self.read_i64()?),
            MetadataType::Float64 => MetadataValue::Float64(self.read_f64()?),
            MetadataType::Array => {
                let element_type = MetadataType::from_u32(self.read_u32()?)?;
                if element_type == MetadataType::Array {
                    return Err(XrtError::InvalidMetadata(
                        "nested metadata arrays are not supported".to_string(),
                    ));
                }
                let len = usize::try_from(self.read_u64()?).map_err(|_| {
                    XrtError::InvalidMetadata(
                        "metadata array length does not fit in usize".to_string(),
                    )
                })?;
                let reserve = checked_mul(len, 1, "metadata array length")?;
                let mut values = Vec::with_capacity(reserve);
                for _ in 0..len {
                    values.push(self.read_metadata_value_of(element_type)?);
                }
                MetadataValue::Array(MetadataArray {
                    element_type,
                    values,
                })
            }
        })
    }
}
