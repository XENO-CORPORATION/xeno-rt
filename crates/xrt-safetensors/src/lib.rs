use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    path::{Component, Path, PathBuf},
};
use xrt_core::{Result, XrtError};

const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const INDEX_FILE_NAME: &str = "model.safetensors.index.json";
const SINGLE_FILE_NAME: &str = "model.safetensors";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SafeTensorDType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    Other(String),
}

impl SafeTensorDType {
    fn from_debug_name(name: String) -> Self {
        match name.as_str() {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            _ => Self::Other(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub byte_len: usize,
    pub shard_file: String,
    shard_index: usize,
    data_start: usize,
    data_end: usize,
}

impl SafeTensorInfo {
    pub fn numel(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{}` element count overflows for shape {:?}",
                    self.name, self.shape
                ))
            })
        })
    }
}

#[derive(Debug)]
struct SafeTensorShard {
    file_name: String,
    path: PathBuf,
    mmap: Mmap,
    tensors: BTreeMap<String, SafeTensorInfo>,
}

impl SafeTensorShard {
    fn open(path: PathBuf, file_name: String, shard_index: usize) -> Result<Self> {
        let file = File::open(&path)?;
        // The mapping is read-only and remains owned by this shard for every returned view.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let (header_len, metadata) = SafeTensors::read_metadata(&mmap).map_err(|err| {
            XrtError::InvalidFormat(format!(
                "failed to parse SafeTensors shard `{}`: {err}",
                path.display()
            ))
        })?;
        let data_base = std::mem::size_of::<u64>()
            .checked_add(header_len)
            .ok_or_else(|| {
                XrtError::InvalidFormat(format!(
                    "SafeTensors shard `{}` header offset overflows",
                    path.display()
                ))
            })?;
        let mut tensors = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let data_start = data_base.checked_add(info.data_offsets.0).ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` start offset overflows in `{}`",
                    path.display()
                ))
            })?;
            let data_end = data_base.checked_add(info.data_offsets.1).ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` end offset overflows in `{}`",
                    path.display()
                ))
            })?;
            if data_start > data_end || data_end > mmap.len() {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` has invalid absolute data range {data_start}..{data_end} in {}-byte shard `{}`",
                    mmap.len(),
                    path.display()
                )));
            }
            let tensor = SafeTensorInfo {
                name: name.clone(),
                dtype: SafeTensorDType::from_debug_name(format!("{:?}", info.dtype)),
                shape: info.shape.clone(),
                byte_len: data_end - data_start,
                shard_file: file_name.clone(),
                shard_index,
                data_start,
                data_end,
            };
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(XrtError::InvalidTensor(format!(
                    "duplicate tensor `{name}` in shard `{}`",
                    path.display()
                )));
            }
        }
        Ok(Self {
            file_name,
            path,
            mmap,
            tensors,
        })
    }

    fn tensor_data(&self, info: &SafeTensorInfo) -> Result<&[u8]> {
        self.mmap
            .get(info.data_start..info.data_end)
            .ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{}` data range is outside shard `{}`",
                    info.name,
                    self.path.display()
                ))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfQuantizationMethod {
    Awq,
    Gptq,
    CompressedTensors,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HfQuantizationConfig {
    pub method: HfQuantizationMethod,
    pub bits: Option<u32>,
    pub group_size: Option<i64>,
    pub zero_point: Option<bool>,
    pub desc_act: Option<bool>,
    pub format: Option<String>,
    pub raw: Value,
}

impl HfQuantizationConfig {
    fn from_value(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            XrtError::InvalidMetadata("quantization_config must be a JSON object".to_string())
        })?;
        let method_name = object
            .get("quant_method")
            .or_else(|| object.get("quantization_method"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "quantization_config is missing string quant_method".to_string(),
                )
            })?
            .trim()
            .to_ascii_lowercase();
        let method = match method_name.as_str() {
            "awq" => HfQuantizationMethod::Awq,
            "gptq" | "gptq_v2" => HfQuantizationMethod::Gptq,
            "compressed-tensors" | "compressed_tensors" => HfQuantizationMethod::CompressedTensors,
            _ => HfQuantizationMethod::Other(method_name),
        };
        let bits = optional_u32_alias(object, "bits", "w_bit")?;
        let group_size = optional_i64_alias(object, "group_size", "q_group_size")?;
        if bits == Some(0) {
            return Err(XrtError::InvalidMetadata(
                "quantization_config bits must be greater than zero".to_string(),
            ));
        }
        if group_size.is_some_and(|size| size == 0 || size < -1) {
            return Err(XrtError::InvalidMetadata(format!(
                "quantization_config group_size must be -1 or positive, got {}",
                group_size.expect("group size was checked above")
            )));
        }
        let explicit_zero_point = optional_bool(object, "zero_point")?;
        let symmetric = optional_bool(object, "sym")?;
        if let (Some(zero_point), Some(sym)) = (explicit_zero_point, symmetric) {
            if zero_point == sym {
                return Err(XrtError::InvalidMetadata(format!(
                    "quantization_config zero_point={zero_point} conflicts with sym={sym}"
                )));
            }
        }
        let zero_point = explicit_zero_point.or_else(|| symmetric.map(|sym| !sym));
        let desc_act = optional_bool(object, "desc_act")?;
        let format = object
            .get("format")
            .or_else(|| object.get("version"))
            .and_then(Value::as_str)
            .map(|value| value.trim().to_ascii_lowercase());
        Ok(Self {
            method,
            bits,
            group_size,
            zero_point,
            desc_act,
            format,
            raw: value.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HfModelConfig {
    pub model_name: Option<String>,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    pub dtype: Option<String>,
    pub vocab_size: usize,
    pub bos_token_ids: Vec<u32>,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub quantization: Option<HfQuantizationConfig>,
    pub raw: Value,
}

impl HfModelConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: Value = serde_json::from_slice(bytes).map_err(|err| {
            XrtError::InvalidMetadata(format!("failed to parse Hugging Face config.json: {err}"))
        })?;
        let object = raw.as_object().ok_or_else(|| {
            XrtError::InvalidMetadata("Hugging Face config.json must be an object".to_string())
        })?;
        let model_type = required_string(object, "model_type")?;
        let hidden_size = required_usize(object, "hidden_size")?;
        let intermediate_size = required_usize(object, "intermediate_size")?;
        let max_position_embeddings = required_usize(object, "max_position_embeddings")?;
        let num_attention_heads = required_usize(object, "num_attention_heads")?;
        let num_hidden_layers = required_usize(object, "num_hidden_layers")?;
        let num_key_value_heads =
            optional_usize(object, "num_key_value_heads")?.unwrap_or(num_attention_heads);
        let rms_norm_eps = object
            .get("rms_norm_eps")
            .or_else(|| object.get("layer_norm_eps"))
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "config.json is missing numeric rms_norm_eps/layer_norm_eps".to_string(),
                )
            })? as f32;
        if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(XrtError::InvalidMetadata(format!(
                "config.json rms_norm_eps must be finite and positive, got {rms_norm_eps}"
            )));
        }
        let rope_theta = object
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(XrtError::InvalidMetadata(format!(
                "config.json rope_theta must be finite and positive, got {rope_theta}"
            )));
        }
        let vocab_size = required_usize(object, "vocab_size")?;
        for (field, value) in [
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("max_position_embeddings", max_position_embeddings),
            ("num_attention_heads", num_attention_heads),
            ("num_hidden_layers", num_hidden_layers),
            ("num_key_value_heads", num_key_value_heads),
            ("vocab_size", vocab_size),
        ] {
            if value == 0 {
                return Err(XrtError::InvalidMetadata(format!(
                    "config.json field {field} must be greater than zero"
                )));
            }
        }
        if num_key_value_heads > num_attention_heads
            || num_attention_heads % num_key_value_heads != 0
        {
            return Err(XrtError::InvalidMetadata(format!(
                "config.json attention heads {num_attention_heads} are incompatible with {num_key_value_heads} KV heads"
            )));
        }
        let head_dim = optional_usize(object, "head_dim")?;
        if head_dim.is_none() && hidden_size % num_attention_heads != 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "config.json hidden_size {hidden_size} is not divisible by {num_attention_heads} attention heads"
            )));
        }
        let architectures = object
            .get("architectures")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let quantization = object
            .get("quantization_config")
            .map(HfQuantizationConfig::from_value)
            .transpose()?;
        Ok(Self {
            model_name: object
                .get("_name_or_path")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            architectures,
            model_type,
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            sliding_window: optional_usize(object, "sliding_window")?,
            use_sliding_window: optional_bool(object, "use_sliding_window")?.unwrap_or(false),
            tie_word_embeddings: optional_bool(object, "tie_word_embeddings")?.unwrap_or(false),
            hidden_act: object
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            dtype: object
                .get("dtype")
                .or_else(|| object.get("torch_dtype"))
                .and_then(Value::as_str)
                .map(|value| value.trim().to_ascii_lowercase()),
            vocab_size,
            bos_token_ids: token_ids(object.get("bos_token_id"), "bos_token_id")?,
            eos_token_ids: token_ids(object.get("eos_token_id"), "eos_token_id")?,
            pad_token_id: optional_u32(object, "pad_token_id")?,
            quantization,
            raw,
        })
    }
}

#[derive(Debug)]
pub struct HfModelBundle {
    root: PathBuf,
    config: HfModelConfig,
    shards: Vec<SafeTensorShard>,
    tensors: BTreeMap<String, SafeTensorInfo>,
    declared_total_size: Option<u64>,
}

impl HfModelBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref())?;
        if !root.is_dir() {
            return Err(XrtError::InvalidFormat(format!(
                "SafeTensors model path must be a directory, got `{}`",
                root.display()
            )));
        }
        let config_bytes = read_bounded_json(&root.join("config.json"), MAX_CONFIG_BYTES)?;
        let config = HfModelConfig::from_json_bytes(&config_bytes)?;
        let index_path = root.join(INDEX_FILE_NAME);
        let (shard_files, weight_map, declared_total_size) = if index_path.is_file() {
            let index_bytes = read_bounded_json(&index_path, MAX_INDEX_BYTES)?;
            let index: SafeTensorIndex = serde_json::from_slice(&index_bytes).map_err(|err| {
                XrtError::InvalidMetadata(format!(
                    "failed to parse `{}`: {err}",
                    index_path.display()
                ))
            })?;
            if index.weight_map.is_empty() {
                return Err(XrtError::InvalidMetadata(format!(
                    "`{}` has an empty weight_map",
                    index_path.display()
                )));
            }
            let shard_files = index
                .weight_map
                .values()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            (
                shard_files,
                Some(index.weight_map),
                index.metadata.total_size,
            )
        } else {
            let single = root.join(SINGLE_FILE_NAME);
            let shard_files = if single.is_file() {
                vec![SINGLE_FILE_NAME.to_string()]
            } else {
                let mut candidates = fs::read_dir(&root)?
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| {
                        let path = entry.path();
                        (path.is_file()
                            && path
                                .extension()
                                .and_then(|extension| extension.to_str())
                                .is_some_and(|extension| {
                                    extension.eq_ignore_ascii_case("safetensors")
                                }))
                        .then(|| entry.file_name().to_string_lossy().into_owned())
                    })
                    .collect::<Vec<_>>();
                candidates.sort();
                if candidates.len() != 1 {
                    return Err(XrtError::InvalidMetadata(format!(
                        "SafeTensors model directory `{}` has {} shard files but no `{INDEX_FILE_NAME}`",
                        root.display(),
                        candidates.len()
                    )));
                }
                candidates
            };
            (shard_files, None, None)
        };

        let mut shards = Vec::with_capacity(shard_files.len());
        let mut shard_by_name = HashMap::new();
        for file_name in shard_files {
            let path = resolve_contained_shard(&root, &file_name)?;
            let shard_index = shards.len();
            let shard = SafeTensorShard::open(path, file_name.clone(), shard_index)?;
            shard_by_name.insert(file_name, shard_index);
            shards.push(shard);
        }

        let mut tensors = BTreeMap::new();
        if let Some(weight_map) = weight_map {
            for (tensor_name, shard_file) in &weight_map {
                let shard_index = shard_by_name.get(shard_file).copied().ok_or_else(|| {
                    XrtError::InvalidMetadata(format!(
                        "weight_map tensor `{tensor_name}` references unopened shard `{shard_file}`"
                    ))
                })?;
                let info = shards[shard_index]
                    .tensors
                    .get(tensor_name)
                    .cloned()
                    .ok_or_else(|| {
                        XrtError::InvalidTensor(format!(
                            "weight_map tensor `{tensor_name}` is missing from shard `{shard_file}`"
                        ))
                    })?;
                if tensors.insert(tensor_name.clone(), info).is_some() {
                    return Err(XrtError::InvalidTensor(format!(
                        "weight_map declares duplicate tensor `{tensor_name}`"
                    )));
                }
            }
            for shard in &shards {
                for tensor_name in shard.tensors.keys() {
                    match weight_map.get(tensor_name) {
                        Some(expected) if expected == &shard.file_name => {}
                        Some(expected) => {
                            return Err(XrtError::InvalidTensor(format!(
                                "tensor `{tensor_name}` is stored in `{}` but weight_map declares `{expected}`",
                                shard.file_name
                            )))
                        }
                        None => {
                            return Err(XrtError::InvalidTensor(format!(
                                "tensor `{tensor_name}` in `{}` is absent from weight_map",
                                shard.file_name
                            )))
                        }
                    }
                }
            }
        } else {
            for shard in &shards {
                for (name, info) in &shard.tensors {
                    if tensors.insert(name.clone(), info.clone()).is_some() {
                        return Err(XrtError::InvalidTensor(format!(
                            "duplicate tensor `{name}` across SafeTensors shards"
                        )));
                    }
                }
            }
        }

        let actual_total_size = tensors.values().try_fold(0u64, |total, info| {
            total.checked_add(info.byte_len as u64).ok_or_else(|| {
                XrtError::InvalidTensor(
                    "SafeTensors aggregate tensor byte count overflows u64".to_string(),
                )
            })
        })?;
        if let Some(expected) = declared_total_size {
            if expected != actual_total_size {
                return Err(XrtError::InvalidMetadata(format!(
                    "SafeTensors index declares total_size {expected}, but mapped tensor payloads total {actual_total_size} bytes"
                )));
            }
        }

        Ok(Self {
            root,
            config,
            shards,
            tensors,
            declared_total_size,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &HfModelConfig {
        &self.config
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn declared_total_size(&self) -> Option<u64> {
        self.declared_total_size
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_info(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.tensors.get(name)
    }

    pub fn require_tensor(&self, name: &str) -> Result<HfTensorView<'_>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            XrtError::InvalidTensor(format!("missing SafeTensors tensor `{name}`"))
        })?;
        let shard = self.shards.get(info.shard_index).ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "tensor `{name}` references missing shard index {}",
                info.shard_index
            ))
        })?;
        Ok(HfTensorView {
            info,
            data: shard.tensor_data(info)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HfTensorView<'a> {
    pub info: &'a SafeTensorInfo,
    pub data: &'a [u8],
}

#[derive(Debug, Deserialize)]
struct SafeTensorIndex {
    #[serde(default)]
    metadata: SafeTensorIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct SafeTensorIndexMetadata {
    total_size: Option<u64>,
}

fn read_bounded_json(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let length = fs::metadata(path)?.len();
    if length > max_bytes {
        return Err(XrtError::InvalidMetadata(format!(
            "JSON file `{}` is {length} bytes, above the {max_bytes}-byte limit",
            path.display()
        )));
    }
    Ok(fs::read(path)?)
}

fn resolve_contained_shard(root: &Path, file_name: &str) -> Result<PathBuf> {
    let relative = Path::new(file_name);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(XrtError::InvalidMetadata(format!(
            "unsafe SafeTensors shard path `{file_name}`"
        )));
    }
    if !relative
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
    {
        return Err(XrtError::InvalidMetadata(format!(
            "SafeTensors shard path `{file_name}` must end in .safetensors"
        )));
    }
    let path = fs::canonicalize(root.join(relative))?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(XrtError::InvalidMetadata(format!(
            "SafeTensors shard `{}` escapes model directory `{}`",
            path.display(),
            root.display()
        )));
    }
    Ok(path)
}

fn required_string(object: &serde_json::Map<String, Value>, key: &str) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| XrtError::InvalidMetadata(format!("config.json is missing string {key}")))
}

fn required_usize(object: &serde_json::Map<String, Value>, key: &str) -> Result<usize> {
    optional_usize(object, key)?
        .ok_or_else(|| XrtError::InvalidMetadata(format!("config.json is missing integer {key}")))
}

fn optional_usize(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<usize>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let value = value.as_u64().ok_or_else(|| {
        XrtError::InvalidMetadata(format!(
            "config.json field {key} must be an unsigned integer"
        ))
    })?;
    usize::try_from(value).map(Some).map_err(|_| {
        XrtError::InvalidMetadata(format!("config.json field {key} exceeds usize: {value}"))
    })
}

fn optional_u32(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<u32>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let value = value.as_u64().ok_or_else(|| {
        XrtError::InvalidMetadata(format!(
            "config.json field {key} must be an unsigned integer"
        ))
    })?;
    u32::try_from(value).map(Some).map_err(|_| {
        XrtError::InvalidMetadata(format!("config.json field {key} exceeds u32: {value}"))
    })
}

fn optional_bool(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<bool>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| XrtError::InvalidMetadata(format!("JSON field {key} must be boolean")))
}

fn optional_u32_alias(
    object: &serde_json::Map<String, Value>,
    primary: &str,
    alias: &str,
) -> Result<Option<u32>> {
    match (optional_u32(object, primary)?, optional_u32(object, alias)?) {
        (Some(left), Some(right)) if left != right => Err(XrtError::InvalidMetadata(format!(
            "quantization_config {primary}={left} conflicts with {alias}={right}"
        ))),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn optional_i64_alias(
    object: &serde_json::Map<String, Value>,
    primary: &str,
    alias: &str,
) -> Result<Option<i64>> {
    let parse = |key: &str| -> Result<Option<i64>> {
        let Some(value) = object.get(key) else {
            return Ok(None);
        };
        value.as_i64().map(Some).ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "quantization_config field {key} must be an integer"
            ))
        })
    };
    match (parse(primary)?, parse(alias)?) {
        (Some(left), Some(right)) if left != right => Err(XrtError::InvalidMetadata(format!(
            "quantization_config {primary}={left} conflicts with {alias}={right}"
        ))),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn token_ids(value: Option<&Value>, field: &str) -> Result<Vec<u32>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = match value {
        Value::Number(_) => vec![value],
        Value::Array(values) => values.iter().collect(),
        _ => {
            return Err(XrtError::InvalidMetadata(format!(
                "config.json field {field} must be an integer or integer array"
            )))
        }
    };
    values
        .into_iter()
        .map(|value| {
            let value = value.as_u64().ok_or_else(|| {
                XrtError::InvalidMetadata(format!(
                    "config.json field {field} contains a non-integer value"
                ))
            })?;
            u32::try_from(value).map_err(|_| {
                XrtError::InvalidMetadata(format!(
                    "config.json field {field} contains value above u32: {value}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};

    fn config_json(quantization_config: &str) -> String {
        format!(
            r#"{{
                "_name_or_path": "synthetic/qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 4,
                "intermediate_size": 8,
                "max_position_embeddings": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true,
                "torch_dtype": "bfloat16",
                "vocab_size": 8,
                "bos_token_id": 6,
                "eos_token_id": [6, 7]
                {quantization_config}
            }}"#
        )
    }

    fn write_tensor(path: &Path, name: &str, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![values.len()], &bytes).unwrap();
        safetensors::serialize_to_file([(name, view)], &None, path).unwrap();
    }

    #[test]
    fn parses_awq_aliases_and_normalizes_symmetry() {
        let json = config_json(
            r#", "quantization_config": {
                "quant_method": "awq",
                "w_bit": 4,
                "q_group_size": 128,
                "sym": false,
                "version": "GEMM"
            }"#,
        );
        let config = HfModelConfig::from_json_bytes(json.as_bytes()).unwrap();
        let quant = config.quantization.unwrap();
        assert_eq!(quant.method, HfQuantizationMethod::Awq);
        assert_eq!(quant.bits, Some(4));
        assert_eq!(quant.group_size, Some(128));
        assert_eq!(quant.zero_point, Some(true));
        assert_eq!(quant.format.as_deref(), Some("gemm"));
        assert_eq!(config.eos_token_ids, vec![6, 7]);
    }

    #[test]
    fn rejects_conflicting_quantization_aliases() {
        let json = config_json(
            r#", "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "w_bit": 8
            }"#,
        );
        let error = HfModelConfig::from_json_bytes(json.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("bits=4 conflicts with w_bit=8"));
    }

    #[test]
    fn opens_and_validates_a_sharded_weight_index() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("config.json"), config_json("")).unwrap();
        write_tensor(
            &directory.path().join("model-00001-of-00002.safetensors"),
            "model.embed_tokens.weight",
            &[1.0, 2.0],
        );
        write_tensor(
            &directory.path().join("model-00002-of-00002.safetensors"),
            "model.norm.weight",
            &[3.0, 4.0],
        );
        fs::write(
            directory.path().join(INDEX_FILE_NAME),
            r#"{
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

        let bundle = HfModelBundle::open(directory.path()).unwrap();
        assert_eq!(bundle.shard_count(), 2);
        assert_eq!(bundle.tensor_count(), 2);
        assert_eq!(bundle.declared_total_size(), Some(16));
        let tensor = bundle.require_tensor("model.norm.weight").unwrap();
        assert_eq!(tensor.info.dtype, SafeTensorDType::F32);
        assert_eq!(tensor.info.shape, vec![2]);
        assert_eq!(tensor.data.len(), 8);
        assert_eq!(&tensor.data[..4], &3.0f32.to_le_bytes());
    }

    #[test]
    fn rejects_index_paths_that_escape_the_model_directory() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("config.json"), config_json("")).unwrap();
        fs::write(
            directory.path().join(INDEX_FILE_NAME),
            r#"{
                "weight_map": {
                    "model.embed_tokens.weight": "../outside.safetensors"
                }
            }"#,
        )
        .unwrap();
        let error = HfModelBundle::open(directory.path()).unwrap_err();
        assert!(error.to_string().contains("unsafe SafeTensors shard path"));
    }
}
