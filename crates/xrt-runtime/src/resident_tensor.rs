use std::collections::{BTreeMap, BTreeSet};
use xrt_core::{DType, Result, XrtError};
use xrt_cuda::GptqZeroEncoding;
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_safetensors::{
    HfModelBundle, HfQuantizationConfig, HfQuantizationMethod, SafeTensorDType, SafeTensorInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentTensorStorage {
    Dense,
    AwqGemm4 {
        group_size: usize,
    },
    GptqGemm4 {
        group_size: usize,
    },
    GptqExplicitGemm4 {
        group_size: usize,
        zero_encoding: GptqZeroEncoding,
    },
    CompressedTensorsW4A16 {
        group_size: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: DType,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
    pub numel: usize,
    pub byte_len: usize,
    pub storage: ResidentTensorStorage,
}

impl ResidentTensorInfo {
    fn from_gguf(info: &TensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            dimensions: info.dimensions.clone(),
            dtype: info.dtype,
            rank: info.dimensions.len(),
            rows: info.rows(),
            cols: info.row_len(),
            numel: info.numel(),
            byte_len: info.nbytes,
            storage: ResidentTensorStorage::Dense,
        }
    }
}

pub(crate) struct ResidentAwqGemm4Data<'a> {
    pub qweight: &'a [u8],
    pub qzeros: &'a [u8],
    pub scales: &'a [u8],
    pub scale_dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

pub(crate) struct ResidentGptqGemm4Data<'a> {
    pub qweight: &'a [u8],
    pub qzeros: &'a [u8],
    pub scales: &'a [u8],
    pub scale_dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

pub(crate) struct ResidentGptqExplicitGemm4Data<'a> {
    pub qweight: &'a [u8],
    pub qzeros: &'a [u8],
    pub scales: &'a [u8],
    pub scale_dtype: DType,
    pub group_indices: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub zero_encoding: GptqZeroEncoding,
}

pub(crate) struct ResidentCompressedTensorsW4A16Data<'a> {
    pub weight_packed: &'a [u8],
    pub scales: &'a [u8],
    pub scale_dtype: DType,
    pub group_indices: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

pub(crate) trait ResidentTensorSource: Send + Sync {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo>;

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]>;

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo>;

    fn awq_gemm4_data<'a>(&'a self, _name: &str) -> Result<Option<ResidentAwqGemm4Data<'a>>> {
        Ok(None)
    }

    fn gptq_gemm4_data<'a>(&'a self, _name: &str) -> Result<Option<ResidentGptqGemm4Data<'a>>> {
        Ok(None)
    }

    fn gptq_explicit_gemm4_data<'a>(
        &'a self,
        _name: &str,
    ) -> Result<Option<ResidentGptqExplicitGemm4Data<'a>>> {
        Ok(None)
    }

    fn compressed_tensors_w4a16_data<'a>(
        &'a self,
        _name: &str,
    ) -> Result<Option<ResidentCompressedTensorsW4A16Data<'a>>> {
        Ok(None)
    }

    fn require_tensor(&self, name: &str) -> Result<ResidentTensorInfo> {
        self.tensor_info(name)
            .ok_or_else(|| XrtError::InvalidTensor(format!("missing tensor `{name}`")))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GgufResidentTensorSource<'a> {
    gguf: &'a GgufFile,
}

impl<'a> GgufResidentTensorSource<'a> {
    pub(crate) fn new(gguf: &'a GgufFile) -> Self {
        Self { gguf }
    }
}

impl ResidentTensorSource for GgufResidentTensorSource<'_> {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo> {
        self.gguf
            .tensor_info(name)
            .map(ResidentTensorInfo::from_gguf)
    }

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        self.gguf.tensor_data(name)
    }

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo> {
        self.gguf
            .tensor_infos()
            .iter()
            .map(ResidentTensorInfo::from_gguf)
            .collect()
    }
}

#[derive(Debug, Clone)]
enum HfTensorMapping {
    Dense(String),
    AwqGemm4 {
        qweight: String,
        qzeros: String,
        scales: String,
        group_size: usize,
    },
    GptqGemm4 {
        qweight: String,
        qzeros: String,
        scales: String,
        g_idx: String,
        group_size: usize,
    },
    GptqExplicitGemm4 {
        qweight: String,
        qzeros: String,
        scales: String,
        g_idx: String,
        group_size: usize,
        zero_encoding: GptqZeroEncoding,
    },
    CompressedTensorsW4A16 {
        weight_packed: String,
        weight_scale: String,
        weight_shape: String,
        weight_g_idx: String,
        group_size: usize,
    },
}

impl HfTensorMapping {
    fn actual_names(&self) -> Vec<&str> {
        match self {
            Self::Dense(name) => vec![name],
            Self::AwqGemm4 {
                qweight,
                qzeros,
                scales,
                ..
            } => vec![qweight, qzeros, scales],
            Self::GptqGemm4 {
                qweight,
                qzeros,
                scales,
                g_idx,
                ..
            } => vec![qweight, qzeros, scales, g_idx],
            Self::GptqExplicitGemm4 {
                qweight,
                qzeros,
                scales,
                g_idx,
                ..
            } => vec![qweight, qzeros, scales, g_idx],
            Self::CompressedTensorsW4A16 {
                weight_packed,
                weight_scale,
                weight_shape,
                weight_g_idx,
                ..
            } => vec![weight_packed, weight_scale, weight_shape, weight_g_idx],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HfPackedLinearFormat {
    AwqGemm4 {
        configured_group_size: i64,
    },
    GptqGemm4 {
        configured_group_size: i64,
    },
    GptqExplicitGemm4 {
        configured_group_size: i64,
        zero_encoding: GptqZeroEncoding,
    },
    CompressedTensorsW4A16 {
        group_size: usize,
    },
}

pub(crate) struct HfQwen2ResidentTensorSource<'a> {
    bundle: &'a HfModelBundle,
    infos: BTreeMap<String, ResidentTensorInfo>,
    mappings: BTreeMap<String, HfTensorMapping>,
}

impl<'a> HfQwen2ResidentTensorSource<'a> {
    pub(crate) fn new(bundle: &'a HfModelBundle) -> Result<Self> {
        if !bundle.config().model_type.eq_ignore_ascii_case("qwen2") {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors resident source currently supports Qwen2, found `{}`",
                bundle.config().model_type
            )));
        }
        let packed_format = supported_packed_linear_format(bundle)?;

        let mut mappings = BTreeMap::new();
        let mut ignored_actual_names = BTreeSet::new();
        add_required_dense_mapping(
            bundle,
            &mut mappings,
            "token_embd.weight",
            "model.embed_tokens.weight",
        )?;
        add_required_dense_mapping(
            bundle,
            &mut mappings,
            "output_norm.weight",
            "model.norm.weight",
        )?;
        if has_hf_linear(bundle, "lm_head") {
            add_required_linear_mapping(
                bundle,
                &mut mappings,
                "output.weight",
                "lm_head",
                packed_format,
            )?;
        } else if !bundle.config().tie_word_embeddings {
            return Err(XrtError::InvalidTensor(
                "untied Qwen2 SafeTensors model is missing `lm_head.weight` or a supported packed `lm_head` tensor group"
                    .to_string(),
            ));
        }

        for layer in 0..bundle.config().num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for (canonical_suffix, hf_suffix) in [
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ] {
                add_required_dense_mapping(
                    bundle,
                    &mut mappings,
                    &format!("blk.{layer}.{canonical_suffix}"),
                    &format!("{prefix}.{hf_suffix}"),
                )?;
            }
            for (canonical_suffix, hf_base) in [
                ("attn_q.weight", "self_attn.q_proj"),
                ("attn_k.weight", "self_attn.k_proj"),
                ("attn_v.weight", "self_attn.v_proj"),
                ("attn_output.weight", "self_attn.o_proj"),
                ("ffn_gate.weight", "mlp.gate_proj"),
                ("ffn_up.weight", "mlp.up_proj"),
                ("ffn_down.weight", "mlp.down_proj"),
            ] {
                add_required_linear_mapping(
                    bundle,
                    &mut mappings,
                    &format!("blk.{layer}.{canonical_suffix}"),
                    &format!("{prefix}.{hf_base}"),
                    packed_format,
                )?;
            }
            for (canonical_suffix, hf_suffix) in [
                ("attn_q.bias", "self_attn.q_proj.bias"),
                ("attn_k.bias", "self_attn.k_proj.bias"),
                ("attn_v.bias", "self_attn.v_proj.bias"),
            ] {
                add_optional_dense_mapping(
                    bundle,
                    &mut mappings,
                    &format!("blk.{layer}.{canonical_suffix}"),
                    &format!("{prefix}.{hf_suffix}"),
                )?;
            }
            if matches!(
                packed_format,
                Some(
                    HfPackedLinearFormat::GptqGemm4 { .. }
                        | HfPackedLinearFormat::GptqExplicitGemm4 { .. }
                )
            ) {
                for (hf_suffix, expected_len) in [
                    ("self_attn.o_proj.bias", bundle.config().hidden_size),
                    ("mlp.gate_proj.bias", bundle.config().intermediate_size),
                    ("mlp.up_proj.bias", bundle.config().intermediate_size),
                    ("mlp.down_proj.bias", bundle.config().hidden_size),
                ] {
                    let name = format!("{prefix}.{hf_suffix}");
                    if validate_optional_zero_gptq_bias(bundle, &name, expected_len)? {
                        ignored_actual_names.insert(name);
                    }
                }
            }
        }

        let mapped_actual_names = mappings
            .values()
            .flat_map(HfTensorMapping::actual_names)
            .map(ToOwned::to_owned)
            .chain(ignored_actual_names)
            .collect::<BTreeSet<_>>();
        let unmapped = bundle
            .tensor_names()
            .filter(|name| !mapped_actual_names.contains(*name))
            .take(8)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if !unmapped.is_empty() {
            return Err(XrtError::Unsupported(format!(
                "Qwen2 SafeTensors bundle contains unsupported tensors: {}",
                unmapped.join(", ")
            )));
        }

        let infos = mappings
            .iter()
            .map(|(canonical, mapping)| {
                let info = match mapping {
                    HfTensorMapping::Dense(actual) => {
                        let info = bundle.tensor_info(actual).ok_or_else(|| {
                            XrtError::InvalidTensor(format!(
                                "mapped SafeTensors tensor `{actual}` disappeared"
                            ))
                        })?;
                        normalize_hf_tensor(canonical, info)?
                    }
                    HfTensorMapping::AwqGemm4 {
                        qweight,
                        qzeros,
                        scales,
                        group_size,
                    } => normalize_hf_awq_gemm4_matrix(
                        bundle,
                        canonical,
                        qweight,
                        qzeros,
                        scales,
                        *group_size,
                    )?,
                    HfTensorMapping::GptqGemm4 {
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        group_size,
                    } => normalize_hf_gptq_gemm4_matrix(
                        bundle,
                        canonical,
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        *group_size,
                        None,
                    )?,
                    HfTensorMapping::GptqExplicitGemm4 {
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        group_size,
                        zero_encoding,
                    } => normalize_hf_gptq_gemm4_matrix(
                        bundle,
                        canonical,
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        *group_size,
                        Some(*zero_encoding),
                    )?,
                    HfTensorMapping::CompressedTensorsW4A16 {
                        weight_packed,
                        weight_scale,
                        weight_shape,
                        weight_g_idx,
                        group_size,
                    } => normalize_hf_compressed_tensors_w4a16_matrix(
                        bundle,
                        canonical,
                        weight_packed,
                        weight_scale,
                        weight_shape,
                        weight_g_idx,
                        *group_size,
                    )?,
                };
                Ok((canonical.clone(), info))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self {
            bundle,
            infos,
            mappings,
        })
    }
}

impl ResidentTensorSource for HfQwen2ResidentTensorSource<'_> {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo> {
        self.infos.get(name).cloned()
    }

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        let mapping = self
            .mappings
            .get(name)
            .ok_or_else(|| XrtError::InvalidTensor(format!("missing canonical tensor `{name}`")))?;
        match mapping {
            HfTensorMapping::Dense(actual) => Ok(self.bundle.require_tensor(actual)?.data),
            HfTensorMapping::AwqGemm4 { .. }
            | HfTensorMapping::GptqGemm4 { .. }
            | HfTensorMapping::GptqExplicitGemm4 { .. }
            | HfTensorMapping::CompressedTensorsW4A16 { .. } => Err(
                XrtError::InvalidTensor(format!(
                    "canonical tensor `{name}` uses grouped packed storage, not a single dense payload"
                )),
            ),
        }
    }

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo> {
        self.infos.values().cloned().collect()
    }

    fn awq_gemm4_data<'a>(&'a self, name: &str) -> Result<Option<ResidentAwqGemm4Data<'a>>> {
        let Some(HfTensorMapping::AwqGemm4 {
            qweight,
            qzeros,
            scales,
            group_size,
        }) = self.mappings.get(name)
        else {
            return Ok(None);
        };
        let info = self.require_tensor(name)?;
        let scale_info = self.bundle.require_tensor(scales)?;
        Ok(Some(ResidentAwqGemm4Data {
            qweight: self.bundle.require_tensor(qweight)?.data,
            qzeros: self.bundle.require_tensor(qzeros)?.data,
            scales: scale_info.data,
            scale_dtype: safe_float_dtype(scales, &scale_info.info.dtype)?,
            rows: info.rows,
            cols: info.cols,
            group_size: *group_size,
        }))
    }

    fn gptq_gemm4_data<'a>(&'a self, name: &str) -> Result<Option<ResidentGptqGemm4Data<'a>>> {
        let Some(HfTensorMapping::GptqGemm4 {
            qweight,
            qzeros,
            scales,
            group_size,
            ..
        }) = self.mappings.get(name)
        else {
            return Ok(None);
        };
        let info = self.require_tensor(name)?;
        let scale_info = self.bundle.require_tensor(scales)?;
        Ok(Some(ResidentGptqGemm4Data {
            qweight: self.bundle.require_tensor(qweight)?.data,
            qzeros: self.bundle.require_tensor(qzeros)?.data,
            scales: scale_info.data,
            scale_dtype: safe_float_dtype(scales, &scale_info.info.dtype)?,
            rows: info.rows,
            cols: info.cols,
            group_size: *group_size,
        }))
    }

    fn gptq_explicit_gemm4_data<'a>(
        &'a self,
        name: &str,
    ) -> Result<Option<ResidentGptqExplicitGemm4Data<'a>>> {
        let Some(HfTensorMapping::GptqExplicitGemm4 {
            qweight,
            qzeros,
            scales,
            g_idx,
            group_size,
            zero_encoding,
        }) = self.mappings.get(name)
        else {
            return Ok(None);
        };
        let info = self.require_tensor(name)?;
        let scale_info = self.bundle.require_tensor(scales)?;
        Ok(Some(ResidentGptqExplicitGemm4Data {
            qweight: self.bundle.require_tensor(qweight)?.data,
            qzeros: self.bundle.require_tensor(qzeros)?.data,
            scales: scale_info.data,
            scale_dtype: safe_float_dtype(scales, &scale_info.info.dtype)?,
            group_indices: self.bundle.require_tensor(g_idx)?.data,
            rows: info.rows,
            cols: info.cols,
            group_size: *group_size,
            zero_encoding: *zero_encoding,
        }))
    }

    fn compressed_tensors_w4a16_data<'a>(
        &'a self,
        name: &str,
    ) -> Result<Option<ResidentCompressedTensorsW4A16Data<'a>>> {
        let Some(HfTensorMapping::CompressedTensorsW4A16 {
            weight_packed,
            weight_scale,
            weight_g_idx,
            group_size,
            ..
        }) = self.mappings.get(name)
        else {
            return Ok(None);
        };
        let info = self.require_tensor(name)?;
        let scale_info = self.bundle.require_tensor(weight_scale)?;
        Ok(Some(ResidentCompressedTensorsW4A16Data {
            weight_packed: self.bundle.require_tensor(weight_packed)?.data,
            scales: scale_info.data,
            scale_dtype: safe_float_dtype(weight_scale, &scale_info.info.dtype)?,
            group_indices: self.bundle.require_tensor(weight_g_idx)?.data,
            rows: info.rows,
            cols: info.cols,
            group_size: *group_size,
        }))
    }
}

fn validate_optional_zero_gptq_bias(
    bundle: &HfModelBundle,
    name: &str,
    expected_len: usize,
) -> Result<bool> {
    let Some(info) = bundle.tensor_info(name) else {
        return Ok(false);
    };
    if info.shape != [expected_len] {
        return Err(XrtError::InvalidTensor(format!(
            "GPTQ auxiliary zero bias `{name}` has shape {:?}, expected [{expected_len}]",
            info.shape
        )));
    }
    let dtype = safe_float_dtype(name, &info.dtype)?;
    validate_tensor_bytes(
        info,
        match dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => unreachable!("GPTQ auxiliary bias dtype was validated above"),
        },
    )?;
    let data = bundle.require_tensor(name)?.data;
    if data.iter().any(|byte| *byte != 0) {
        return Err(XrtError::Unsupported(format!(
            "GPTQ auxiliary bias `{name}` is nonzero; CUDA decode does not apply O/FFN linear biases"
        )));
    }
    Ok(true)
}

fn supported_packed_linear_format(bundle: &HfModelBundle) -> Result<Option<HfPackedLinearFormat>> {
    let Some(quantization) = bundle.config().quantization.as_ref() else {
        return Ok(None);
    };
    match &quantization.method {
        HfQuantizationMethod::Awq => {
            if quantization.bits != Some(4) {
                return Err(XrtError::Unsupported(format!(
                    "AutoAWQ GEMM requires explicit 4-bit weights, found bits={:?}",
                    quantization.bits
                )));
            }
            let group_size = supported_group_size(quantization.group_size, "AutoAWQ GEMM")?;
            if quantization.zero_point != Some(true) {
                return Err(XrtError::Unsupported(format!(
                    "AutoAWQ GEMM requires explicit zero_point=true, found {:?}",
                    quantization.zero_point
                )));
            }
            if quantization.desc_act == Some(true) {
                return Err(XrtError::Unsupported(
                    "AutoAWQ GEMM does not support desc_act=true".to_string(),
                ));
            }
            if quantization.format.as_deref() != Some("gemm") {
                return Err(XrtError::Unsupported(format!(
                    "AutoAWQ CUDA decode currently supports version/format `gemm`, found {:?}",
                    quantization.format
                )));
            }
            Ok(Some(HfPackedLinearFormat::AwqGemm4 {
                configured_group_size: group_size,
            }))
        }
        HfQuantizationMethod::Gptq => supported_gptq_gemm4(quantization).map(Some),
        HfQuantizationMethod::CompressedTensors => {
            supported_compressed_tensors_w4a16(quantization).map(Some)
        }
        method => Err(XrtError::Unsupported(format!(
            "SafeTensors Qwen2 CUDA decode supports dense tensors, AutoAWQ GEMM, GPTQ v1/v2 GEMM4, or compressed-tensors W4A16, found {method:?}"
        ))),
    }
}

fn supported_gptq_gemm4(quantization: &HfQuantizationConfig) -> Result<HfPackedLinearFormat> {
    if quantization.bits != Some(4) {
        return Err(XrtError::Unsupported(format!(
            "GPTQ GEMM requires explicit 4-bit weights, found bits={:?}",
            quantization.bits
        )));
    }
    let group_size = supported_group_size(quantization.group_size, "GPTQ GEMM")?;
    let raw = quantization.raw.as_object().ok_or_else(|| {
        XrtError::InvalidMetadata("GPTQ quantization config must be an object".to_string())
    })?;
    if raw.get("dynamic").is_some_and(|value| !value.is_null()) {
        return Err(XrtError::Unsupported(
            "GPTQ CUDA decode does not support per-module dynamic quantization overrides"
                .to_string(),
        ));
    }
    if let Some(pack_dtype) = raw.get("pack_dtype").filter(|value| !value.is_null()) {
        if !pack_dtype
            .as_str()
            .is_some_and(|value| value.eq_ignore_ascii_case("int32"))
        {
            return Err(XrtError::Unsupported(format!(
                "GPTQ CUDA decode requires pack_dtype `int32`, found {pack_dtype}"
            )));
        }
    }

    let zero_encoding = gptq_zero_encoding(quantization)?;
    match zero_encoding {
        GptqZeroEncoding::V1MinusOne if quantization.zero_point != Some(false) => {
            return Err(XrtError::Unsupported(format!(
                "GPTQ v1 GEMM requires explicit symmetric quantization, found zero_point={:?}",
                quantization.zero_point
            )));
        }
        GptqZeroEncoding::V2Direct if quantization.zero_point.is_none() => {
            return Err(XrtError::Unsupported(
                "GPTQ v2 GEMM requires an explicit `sym` or `zero_point` declaration".to_string(),
            ));
        }
        _ => {}
    }
    let desc_act = quantization.desc_act.ok_or_else(|| {
        XrtError::Unsupported("GPTQ GEMM requires an explicit desc_act declaration".to_string())
    })?;
    let exllama_version = raw
        .get("exllama_config")
        .filter(|value| !value.is_null())
        .and_then(|value| value.get("version"))
        .and_then(|value| value.as_u64());
    if exllama_version.is_some_and(|version| !matches!(version, 1 | 2)) {
        return Err(XrtError::Unsupported(format!(
            "GPTQ CUDA decode supports exllama_config.version 1 or 2, found {exllama_version:?}"
        )));
    }

    if zero_encoding == GptqZeroEncoding::V1MinusOne && !desc_act {
        if exllama_version != Some(1) {
            return Err(XrtError::Unsupported(format!(
                "standard GPTQ v1 CUDA decode requires exllama_config.version=1, found {exllama_version:?}"
            )));
        }
        Ok(HfPackedLinearFormat::GptqGemm4 {
            configured_group_size: group_size,
        })
    } else {
        Ok(HfPackedLinearFormat::GptqExplicitGemm4 {
            configured_group_size: group_size,
            zero_encoding,
        })
    }
}

fn gptq_zero_encoding(quantization: &HfQuantizationConfig) -> Result<GptqZeroEncoding> {
    let raw = quantization.raw.as_object().ok_or_else(|| {
        XrtError::InvalidMetadata("GPTQ quantization config must be an object".to_string())
    })?;
    let normalized_string = |key: &str| -> Result<Option<String>> {
        let Some(value) = raw.get(key) else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        value
            .as_str()
            .map(|value| Some(value.trim().to_ascii_lowercase()))
            .ok_or_else(|| {
                XrtError::InvalidMetadata(format!(
                    "GPTQ quantization field `{key}` must be a string or null"
                ))
            })
    };
    let quant_method = normalized_string("quant_method")?;
    let checkpoint_format = normalized_string("checkpoint_format")?;
    let format = normalized_string("format")?;
    for (key, value) in [
        ("quant_method", quant_method.as_deref()),
        ("checkpoint_format", checkpoint_format.as_deref()),
        ("format", format.as_deref()),
    ] {
        if let Some(value) = value {
            if !matches!(value, "gptq" | "gptq_v2") {
                return Err(XrtError::Unsupported(format!(
                    "GPTQ CUDA decode does not support {key} `{value}`"
                )));
            }
        }
    }
    let metadata_v2 = raw
        .get("meta")
        .filter(|value| !value.is_null())
        .and_then(|value| value.get("v2"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let is_v2 = metadata_v2
        || quant_method.as_deref() == Some("gptq_v2")
        || checkpoint_format.as_deref() == Some("gptq_v2")
        || format.as_deref() == Some("gptq_v2");
    Ok(if is_v2 {
        GptqZeroEncoding::V2Direct
    } else {
        GptqZeroEncoding::V1MinusOne
    })
}

fn supported_compressed_tensors_w4a16(
    quantization: &HfQuantizationConfig,
) -> Result<HfPackedLinearFormat> {
    if quantization.format.as_deref() != Some("pack-quantized") {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors CUDA decode requires format `pack-quantized`, found {:?}",
            quantization.format
        )));
    }
    let raw = quantization.raw.as_object().ok_or_else(|| {
        XrtError::InvalidMetadata(
            "compressed-tensors quantization config must be an object".to_string(),
        )
    })?;
    if raw
        .get("quantization_status")
        .and_then(|value| value.as_str())
        != Some("compressed")
    {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors CUDA decode requires quantization_status `compressed`, found {:?}",
            raw.get("quantization_status")
        )));
    }
    if raw
        .get("kv_cache_scheme")
        .is_some_and(|value| !value.is_null())
    {
        return Err(XrtError::Unsupported(
            "compressed-tensors W4A16 does not support a quantized kv_cache_scheme".to_string(),
        ));
    }

    let config_groups = raw
        .get("config_groups")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            XrtError::InvalidMetadata(
                "compressed-tensors config is missing object config_groups".to_string(),
            )
        })?;
    if config_groups.len() != 1 {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires one uniform config group, found {}",
            config_groups.len()
        )));
    }
    let (group_name, scheme) = config_groups
        .iter()
        .next()
        .expect("one compressed-tensors config group was checked above");
    let scheme = scheme.as_object().ok_or_else(|| {
        XrtError::InvalidMetadata(format!(
            "compressed-tensors config group `{group_name}` must be an object"
        ))
    })?;
    if scheme
        .get("input_activations")
        .is_some_and(|value| !value.is_null())
        || scheme
            .get("output_activations")
            .is_some_and(|value| !value.is_null())
    {
        return Err(XrtError::Unsupported(
            "compressed-tensors W4A16 requires unquantized activations".to_string(),
        ));
    }
    let targets = scheme
        .get("targets")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "compressed-tensors config group `{group_name}` is missing array targets"
            ))
        })?;
    if targets.len() != 1 || targets[0].as_str() != Some("Linear") {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires exactly target `Linear`, found {targets:?}"
        )));
    }
    let weights = scheme
        .get("weights")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "compressed-tensors config group `{group_name}` is missing object weights"
            ))
        })?;
    if weights.get("num_bits").and_then(|value| value.as_u64()) != Some(4)
        || weights.get("type").and_then(|value| value.as_str()) != Some("int")
    {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires 4-bit INT weights, found num_bits={:?}, type={:?}",
            weights.get("num_bits"),
            weights.get("type")
        )));
    }
    if weights.get("symmetric").and_then(|value| value.as_bool()) != Some(true) {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires symmetric weights, found {:?}",
            weights.get("symmetric")
        )));
    }
    if weights.get("strategy").and_then(|value| value.as_str()) != Some("group") {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires group strategy, found {:?}",
            weights.get("strategy")
        )));
    }
    if weights.get("dynamic").and_then(|value| value.as_bool()) != Some(false) {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires static weights, found dynamic={:?}",
            weights.get("dynamic")
        )));
    }
    if weights.get("actorder").and_then(|value| value.as_str()) != Some("group") {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors W4A16 requires actorder `group`, found {:?}",
            weights.get("actorder")
        )));
    }
    if weights
        .get("block_structure")
        .is_some_and(|value| !value.is_null())
    {
        return Err(XrtError::Unsupported(
            "compressed-tensors W4A16 does not support block_structure".to_string(),
        ));
    }
    if let Some(sparsity) = raw.get("sparsity_config") {
        let sparsity = sparsity.as_object().ok_or_else(|| {
            XrtError::InvalidMetadata(
                "compressed-tensors sparsity_config must be an object".to_string(),
            )
        })?;
        let dense = sparsity.get("format").and_then(|value| value.as_str()) == Some("dense");
        let no_targets = sparsity
            .get("targets")
            .and_then(|value| value.as_array())
            .is_some_and(|targets| targets.is_empty());
        if !dense || !no_targets {
            return Err(XrtError::Unsupported(
                "compressed-tensors W4A16 currently requires dense nonsparse storage".to_string(),
            ));
        }
    }

    let group_size = weights
        .get("group_size")
        .and_then(|value| value.as_i64())
        .ok_or_else(|| {
            XrtError::InvalidMetadata(
                "compressed-tensors W4A16 weights are missing integer group_size".to_string(),
            )
        })?;
    let group_size = supported_group_size(Some(group_size), "compressed-tensors W4A16")?;
    if group_size == -1 {
        return Err(XrtError::Unsupported(
            "compressed-tensors W4A16 full-width group_size=-1 is not wired".to_string(),
        ));
    }
    Ok(HfPackedLinearFormat::CompressedTensorsW4A16 {
        group_size: usize::try_from(group_size).map_err(|_| {
            XrtError::InvalidMetadata(format!(
                "compressed-tensors W4A16 group size {group_size} exceeds usize"
            ))
        })?,
    })
}

fn supported_group_size(group_size: Option<i64>, format_name: &str) -> Result<i64> {
    let group_size = group_size.ok_or_else(|| {
        XrtError::InvalidMetadata(format!(
            "{format_name} requires an explicit group_size/q_group_size"
        ))
    })?;
    if !matches!(group_size, -1 | 32 | 64 | 128) {
        return Err(XrtError::Unsupported(format!(
            "{format_name} group size {group_size} is unsupported; expected -1, 32, 64, or 128"
        )));
    }
    Ok(group_size)
}

fn has_hf_linear(bundle: &HfModelBundle, base: &str) -> bool {
    [
        "weight",
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
        "weight_packed",
        "weight_scale",
        "weight_shape",
        "weight_g_idx",
    ]
    .into_iter()
    .any(|suffix| bundle.tensor_info(&format!("{base}.{suffix}")).is_some())
}

fn add_required_linear_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, HfTensorMapping>,
    canonical: &str,
    base: &str,
    packed_format: Option<HfPackedLinearFormat>,
) -> Result<()> {
    let weight = format!("{base}.weight");
    let qweight = format!("{base}.qweight");
    let qzeros = format!("{base}.qzeros");
    let scales = format!("{base}.scales");
    let g_idx = format!("{base}.g_idx");
    let weight_packed = format!("{base}.weight_packed");
    let weight_scale = format!("{base}.weight_scale");
    let weight_shape = format!("{base}.weight_shape");
    let weight_g_idx = format!("{base}.weight_g_idx");
    let has_weight = bundle.tensor_info(&weight).is_some();
    let legacy_component_presence = [
        bundle.tensor_info(&qweight).is_some(),
        bundle.tensor_info(&qzeros).is_some(),
        bundle.tensor_info(&scales).is_some(),
        bundle.tensor_info(&g_idx).is_some(),
    ];
    let compressed_component_presence = [
        bundle.tensor_info(&weight_packed).is_some(),
        bundle.tensor_info(&weight_scale).is_some(),
        bundle.tensor_info(&weight_shape).is_some(),
        bundle.tensor_info(&weight_g_idx).is_some(),
    ];
    let has_legacy_components = legacy_component_presence.iter().any(|present| *present);
    let has_compressed_components = compressed_component_presence.iter().any(|present| *present);
    if has_weight && (has_legacy_components || has_compressed_components) {
        return Err(XrtError::InvalidTensor(format!(
            "SafeTensors linear `{base}` mixes dense `.weight` and packed components"
        )));
    }
    if has_legacy_components && has_compressed_components {
        return Err(XrtError::InvalidTensor(format!(
            "SafeTensors linear `{base}` mixes AWQ/GPTQ and compressed-tensors packed components"
        )));
    }
    if has_weight {
        return insert_mapping(mappings, canonical, HfTensorMapping::Dense(weight));
    }

    let packed_format = packed_format.ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "Qwen2 SafeTensors model is missing dense tensor `{weight}` for `{canonical}`"
        ))
    })?;
    match packed_format {
        HfPackedLinearFormat::AwqGemm4 {
            configured_group_size,
        } => {
            if has_compressed_components {
                return Err(XrtError::InvalidTensor(format!(
                    "AutoAWQ linear `{base}` contains compressed-tensors components"
                )));
            }
            if legacy_component_presence[..3]
                .iter()
                .any(|present| !*present)
            {
                return Err(XrtError::InvalidTensor(format!(
                    "AutoAWQ linear `{base}` requires `.qweight`, `.qzeros`, and `.scales`"
                )));
            }
            if legacy_component_presence[3] {
                return Err(XrtError::Unsupported(format!(
                    "AutoAWQ linear `{base}` unexpectedly contains GPTQ `.g_idx` storage"
                )));
            }
            let qweight_info = bundle.tensor_info(&qweight).ok_or_else(|| {
                XrtError::InvalidTensor(format!("AutoAWQ linear `{base}` is missing `{qweight}`"))
            })?;
            let cols = match qweight_info.shape.as_slice() {
                [cols, _] => *cols,
                shape => {
                    return Err(XrtError::InvalidTensor(format!(
                        "AutoAWQ qweight `{qweight}` must be rank 2, found shape {shape:?}"
                    )))
                }
            };
            let group_size = resolved_group_size(configured_group_size, cols, "AutoAWQ")?;
            insert_mapping(
                mappings,
                canonical,
                HfTensorMapping::AwqGemm4 {
                    qweight,
                    qzeros,
                    scales,
                    group_size,
                },
            )
        }
        HfPackedLinearFormat::GptqGemm4 {
            configured_group_size,
        }
        | HfPackedLinearFormat::GptqExplicitGemm4 {
            configured_group_size,
            ..
        } => {
            if has_compressed_components {
                return Err(XrtError::InvalidTensor(format!(
                    "GPTQ linear `{base}` contains compressed-tensors components"
                )));
            }
            if legacy_component_presence.iter().any(|present| !*present) {
                return Err(XrtError::InvalidTensor(format!(
                    "GPTQ linear `{base}` requires `.qweight`, `.qzeros`, `.scales`, and `.g_idx`"
                )));
            }
            let qweight_info = bundle.tensor_info(&qweight).ok_or_else(|| {
                XrtError::InvalidTensor(format!("GPTQ linear `{base}` is missing `{qweight}`"))
            })?;
            let packed_cols = match qweight_info.shape.as_slice() {
                [packed_cols, _] => *packed_cols,
                shape => {
                    return Err(XrtError::InvalidTensor(format!(
                        "GPTQ qweight `{qweight}` must be rank 2, found shape {shape:?}"
                    )))
                }
            };
            let cols = packed_cols.checked_mul(8).ok_or_else(|| {
                XrtError::InvalidTensor(format!("GPTQ qweight `{qweight}` input width overflows"))
            })?;
            let group_size = resolved_group_size(configured_group_size, cols, "GPTQ")?;
            let mapping = match packed_format {
                HfPackedLinearFormat::GptqGemm4 { .. } => HfTensorMapping::GptqGemm4 {
                    qweight,
                    qzeros,
                    scales,
                    g_idx,
                    group_size,
                },
                HfPackedLinearFormat::GptqExplicitGemm4 { zero_encoding, .. } => {
                    HfTensorMapping::GptqExplicitGemm4 {
                        qweight,
                        qzeros,
                        scales,
                        g_idx,
                        group_size,
                        zero_encoding,
                    }
                }
                _ => unreachable!("GPTQ branch received a non-GPTQ format"),
            };
            insert_mapping(mappings, canonical, mapping)
        }
        HfPackedLinearFormat::CompressedTensorsW4A16 { group_size } => {
            if has_legacy_components {
                return Err(XrtError::InvalidTensor(format!(
                    "compressed-tensors linear `{base}` contains AWQ/GPTQ components"
                )));
            }
            if compressed_component_presence
                .iter()
                .any(|present| !*present)
            {
                return Err(XrtError::InvalidTensor(format!(
                    "compressed-tensors linear `{base}` requires `.weight_packed`, `.weight_scale`, `.weight_shape`, and `.weight_g_idx`"
                )));
            }
            insert_mapping(
                mappings,
                canonical,
                HfTensorMapping::CompressedTensorsW4A16 {
                    weight_packed,
                    weight_scale,
                    weight_shape,
                    weight_g_idx,
                    group_size,
                },
            )
        }
    }
}

fn resolved_group_size(
    configured_group_size: i64,
    cols: usize,
    format_name: &str,
) -> Result<usize> {
    if configured_group_size == -1 {
        Ok(cols)
    } else {
        usize::try_from(configured_group_size).map_err(|_| {
            XrtError::InvalidMetadata(format!(
                "{format_name} group size {configured_group_size} exceeds usize"
            ))
        })
    }
}

fn add_required_dense_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, HfTensorMapping>,
    canonical: &str,
    actual: &str,
) -> Result<()> {
    if bundle.tensor_info(actual).is_none() {
        return Err(XrtError::InvalidTensor(format!(
            "Qwen2 SafeTensors model is missing required tensor `{actual}` for `{canonical}`"
        )));
    }
    insert_mapping(
        mappings,
        canonical,
        HfTensorMapping::Dense(actual.to_string()),
    )
}

fn add_optional_dense_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, HfTensorMapping>,
    canonical: &str,
    actual: &str,
) -> Result<()> {
    if bundle.tensor_info(actual).is_some() {
        insert_mapping(
            mappings,
            canonical,
            HfTensorMapping::Dense(actual.to_string()),
        )?;
    }
    Ok(())
}

fn insert_mapping(
    mappings: &mut BTreeMap<String, HfTensorMapping>,
    canonical: &str,
    mapping: HfTensorMapping,
) -> Result<()> {
    if mappings.insert(canonical.to_string(), mapping).is_some() {
        return Err(XrtError::InvalidTensor(format!(
            "duplicate canonical tensor mapping `{canonical}`"
        )));
    }
    Ok(())
}

fn safe_float_dtype(name: &str, dtype: &SafeTensorDType) -> Result<DType> {
    match dtype {
        SafeTensorDType::F32 => Ok(DType::F32),
        SafeTensorDType::F16 => Ok(DType::F16),
        SafeTensorDType::Bf16 => Ok(DType::BF16),
        dtype => Err(XrtError::Unsupported(format!(
            "SafeTensors tensor `{name}` has unsupported float dtype {dtype:?}; expected F32, F16, or BF16"
        ))),
    }
}

fn validate_tensor_bytes(info: &SafeTensorInfo, element_bytes: usize) -> Result<()> {
    let expected = info.numel()?.checked_mul(element_bytes).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "SafeTensors tensor `{}` byte length overflows",
            info.name
        ))
    })?;
    if info.byte_len != expected {
        return Err(XrtError::InvalidTensor(format!(
            "SafeTensors tensor `{}` has {} bytes, expected {expected} for {:?} shape {:?}",
            info.name, info.byte_len, info.dtype, info.shape
        )));
    }
    Ok(())
}

fn normalize_hf_awq_gemm4_matrix(
    bundle: &HfModelBundle,
    canonical: &str,
    qweight_name: &str,
    qzeros_name: &str,
    scales_name: &str,
    group_size: usize,
) -> Result<ResidentTensorInfo> {
    let qweight = bundle.tensor_info(qweight_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!("missing AutoAWQ tensor `{qweight_name}`"))
    })?;
    let qzeros = bundle.tensor_info(qzeros_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!("missing AutoAWQ tensor `{qzeros_name}`"))
    })?;
    let scales = bundle.tensor_info(scales_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!("missing AutoAWQ tensor `{scales_name}`"))
    })?;
    if qweight.dtype != SafeTensorDType::I32 || qzeros.dtype != SafeTensorDType::I32 {
        return Err(XrtError::Unsupported(format!(
            "AutoAWQ tensors `{qweight_name}` and `{qzeros_name}` must use I32 storage"
        )));
    }
    let scale_dtype = safe_float_dtype(scales_name, &scales.dtype)?;
    let (cols, packed_rows) = match qweight.shape.as_slice() {
        [cols, packed_rows] => (*cols, *packed_rows),
        shape => {
            return Err(XrtError::InvalidTensor(format!(
            "AutoAWQ qweight `{qweight_name}` must have shape [input, output/8], found {shape:?}"
        )))
        }
    };
    let rows = packed_rows.checked_mul(8).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "AutoAWQ qweight `{qweight_name}` output width overflows"
        ))
    })?;
    if cols == 0 || rows == 0 || group_size == 0 || cols % group_size != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "AutoAWQ matrix `{canonical}` has incompatible rows={rows}, cols={cols}, group_size={group_size}"
        )));
    }
    let groups = cols / group_size;
    if qzeros.shape != [groups, packed_rows] {
        return Err(XrtError::InvalidTensor(format!(
            "AutoAWQ qzeros `{qzeros_name}` has shape {:?}, expected [{groups}, {packed_rows}]",
            qzeros.shape
        )));
    }
    if scales.shape != [groups, rows] {
        return Err(XrtError::InvalidTensor(format!(
            "AutoAWQ scales `{scales_name}` has shape {:?}, expected [{groups}, {rows}]",
            scales.shape
        )));
    }
    validate_tensor_bytes(qweight, 4)?;
    validate_tensor_bytes(qzeros, 4)?;
    validate_tensor_bytes(
        scales,
        match scale_dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => unreachable!("AutoAWQ scale dtype was validated above"),
        },
    )?;
    let numel = rows.checked_mul(cols).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "AutoAWQ matrix `{canonical}` element count overflows"
        ))
    })?;
    let byte_len = qweight
        .byte_len
        .checked_add(qzeros.byte_len)
        .and_then(|bytes| bytes.checked_add(scales.byte_len))
        .ok_or_else(|| {
            XrtError::InvalidTensor(format!("AutoAWQ matrix `{canonical}` byte count overflows"))
        })?;

    Ok(ResidentTensorInfo {
        name: canonical.to_string(),
        dimensions: vec![rows, cols],
        dtype: scale_dtype,
        rank: 2,
        rows,
        cols,
        numel,
        byte_len,
        storage: ResidentTensorStorage::AwqGemm4 { group_size },
    })
}

fn normalize_hf_gptq_gemm4_matrix(
    bundle: &HfModelBundle,
    canonical: &str,
    qweight_name: &str,
    qzeros_name: &str,
    scales_name: &str,
    g_idx_name: &str,
    group_size: usize,
    explicit_zero_encoding: Option<GptqZeroEncoding>,
) -> Result<ResidentTensorInfo> {
    let qweight = bundle
        .tensor_info(qweight_name)
        .ok_or_else(|| XrtError::InvalidTensor(format!("missing GPTQ tensor `{qweight_name}`")))?;
    let qzeros = bundle
        .tensor_info(qzeros_name)
        .ok_or_else(|| XrtError::InvalidTensor(format!("missing GPTQ tensor `{qzeros_name}`")))?;
    let scales = bundle
        .tensor_info(scales_name)
        .ok_or_else(|| XrtError::InvalidTensor(format!("missing GPTQ tensor `{scales_name}`")))?;
    let g_idx = bundle
        .tensor_info(g_idx_name)
        .ok_or_else(|| XrtError::InvalidTensor(format!("missing GPTQ tensor `{g_idx_name}`")))?;
    if qweight.dtype != SafeTensorDType::I32
        || qzeros.dtype != SafeTensorDType::I32
        || g_idx.dtype != SafeTensorDType::I32
    {
        return Err(XrtError::Unsupported(format!(
            "GPTQ tensors `{qweight_name}`, `{qzeros_name}`, and `{g_idx_name}` must use I32 storage"
        )));
    }
    let scale_dtype = safe_float_dtype(scales_name, &scales.dtype)?;
    let (packed_cols, rows) = match qweight.shape.as_slice() {
        [packed_cols, rows] => (*packed_cols, *rows),
        shape => {
            return Err(XrtError::InvalidTensor(format!(
                "GPTQ qweight `{qweight_name}` must have shape [input/8, output], found {shape:?}"
            )))
        }
    };
    let cols = packed_cols.checked_mul(8).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "GPTQ qweight `{qweight_name}` input width overflows"
        ))
    })?;
    if cols == 0 || rows == 0 || rows % 8 != 0 || group_size == 0 || cols % group_size != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "GPTQ matrix `{canonical}` has incompatible rows={rows}, cols={cols}, group_size={group_size}"
        )));
    }
    let groups = cols / group_size;
    let packed_rows = rows / 8;
    if qzeros.shape != [groups, packed_rows] {
        return Err(XrtError::InvalidTensor(format!(
            "GPTQ qzeros `{qzeros_name}` has shape {:?}, expected [{groups}, {packed_rows}]",
            qzeros.shape
        )));
    }
    if scales.shape != [groups, rows] {
        return Err(XrtError::InvalidTensor(format!(
            "GPTQ scales `{scales_name}` has shape {:?}, expected [{groups}, {rows}]",
            scales.shape
        )));
    }
    if g_idx.shape != [cols] {
        return Err(XrtError::InvalidTensor(format!(
            "GPTQ group index `{g_idx_name}` has shape {:?}, expected [{cols}]",
            g_idx.shape
        )));
    }
    validate_tensor_bytes(qweight, 4)?;
    validate_tensor_bytes(qzeros, 4)?;
    validate_tensor_bytes(g_idx, 4)?;
    validate_tensor_bytes(
        scales,
        match scale_dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => unreachable!("GPTQ scale dtype was validated above"),
        },
    )?;
    let group_indices = bundle.require_tensor(g_idx_name)?.data;
    let mut group_counts = explicit_zero_encoding.map(|_| vec![0usize; groups]);
    for col in 0..cols {
        let offset = col.checked_mul(4).ok_or_else(|| {
            XrtError::InvalidTensor(format!("GPTQ group index `{g_idx_name}` offset overflows"))
        })?;
        let actual = i32::from_le_bytes([
            group_indices[offset],
            group_indices[offset + 1],
            group_indices[offset + 2],
            group_indices[offset + 3],
        ]);
        if let Some(group_counts) = group_counts.as_mut() {
            let group = usize::try_from(actual).map_err(|_| {
                XrtError::InvalidTensor(format!(
                    "GPTQ group index `{g_idx_name}` is negative at input {col}: {actual}"
                ))
            })?;
            if group >= groups {
                return Err(XrtError::InvalidTensor(format!(
                    "GPTQ group index `{g_idx_name}` is {group} at input {col}, expected less than {groups}"
                )));
            }
            group_counts[group] = group_counts[group].checked_add(1).ok_or_else(|| {
                XrtError::InvalidTensor(format!("GPTQ group index `{g_idx_name}` count overflow"))
            })?;
        } else {
            let expected = i32::try_from(col / group_size).map_err(|_| {
                XrtError::InvalidTensor(format!(
                    "GPTQ group index `{g_idx_name}` expected value exceeds i32"
                ))
            })?;
            if actual != expected {
                return Err(XrtError::Unsupported(format!(
                    "GPTQ group index `{g_idx_name}` uses act-order or a nonstandard map at input {col}: found {actual}, expected {expected}"
                )));
            }
        }
    }
    if let Some(group_counts) = group_counts {
        if let Some((group, count)) = group_counts
            .iter()
            .copied()
            .enumerate()
            .find(|(_, count)| *count != group_size)
        {
            return Err(XrtError::InvalidTensor(format!(
                "GPTQ group index `{g_idx_name}` maps {count} columns to group {group}, expected {group_size}"
            )));
        }
    }

    let numel = rows.checked_mul(cols).ok_or_else(|| {
        XrtError::InvalidTensor(format!("GPTQ matrix `{canonical}` element count overflows"))
    })?;
    let byte_len = qweight
        .byte_len
        .checked_add(qzeros.byte_len)
        .and_then(|bytes| bytes.checked_add(scales.byte_len))
        .and_then(|bytes| bytes.checked_add(g_idx.byte_len))
        .ok_or_else(|| {
            XrtError::InvalidTensor(format!("GPTQ matrix `{canonical}` byte count overflows"))
        })?;

    let storage = explicit_zero_encoding.map_or(
        ResidentTensorStorage::GptqGemm4 { group_size },
        |zero_encoding| ResidentTensorStorage::GptqExplicitGemm4 {
            group_size,
            zero_encoding,
        },
    );
    Ok(ResidentTensorInfo {
        name: canonical.to_string(),
        dimensions: vec![rows, cols],
        dtype: scale_dtype,
        rank: 2,
        rows,
        cols,
        numel,
        byte_len,
        storage,
    })
}

fn normalize_hf_compressed_tensors_w4a16_matrix(
    bundle: &HfModelBundle,
    canonical: &str,
    weight_packed_name: &str,
    weight_scale_name: &str,
    weight_shape_name: &str,
    weight_g_idx_name: &str,
    group_size: usize,
) -> Result<ResidentTensorInfo> {
    let weight_packed = bundle.tensor_info(weight_packed_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "missing compressed-tensors tensor `{weight_packed_name}`"
        ))
    })?;
    let weight_scale = bundle.tensor_info(weight_scale_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "missing compressed-tensors tensor `{weight_scale_name}`"
        ))
    })?;
    let weight_shape = bundle.tensor_info(weight_shape_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "missing compressed-tensors tensor `{weight_shape_name}`"
        ))
    })?;
    let weight_g_idx = bundle.tensor_info(weight_g_idx_name).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "missing compressed-tensors tensor `{weight_g_idx_name}`"
        ))
    })?;
    if weight_packed.dtype != SafeTensorDType::I32
        || weight_shape.dtype != SafeTensorDType::I64
        || weight_g_idx.dtype != SafeTensorDType::I32
    {
        return Err(XrtError::Unsupported(format!(
            "compressed-tensors `{weight_packed_name}`, `{weight_shape_name}`, and `{weight_g_idx_name}` must use I32, I64, and I32 storage respectively"
        )));
    }
    let scale_dtype = safe_float_dtype(weight_scale_name, &weight_scale.dtype)?;
    let (rows, packed_cols) = match weight_packed.shape.as_slice() {
        [rows, packed_cols] => (*rows, *packed_cols),
        shape => {
            return Err(XrtError::InvalidTensor(format!(
                "compressed-tensors packed weight `{weight_packed_name}` must have shape [output, input/8], found {shape:?}"
            )))
        }
    };
    let cols = packed_cols.checked_mul(8).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "compressed-tensors packed weight `{weight_packed_name}` input width overflows"
        ))
    })?;
    if rows == 0 || cols == 0 || group_size == 0 || cols % group_size != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors matrix `{canonical}` has incompatible rows={rows}, cols={cols}, group_size={group_size}"
        )));
    }
    let groups = cols / group_size;
    if weight_scale.shape != [rows, groups] {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors scales `{weight_scale_name}` have shape {:?}, expected [{rows}, {groups}]",
            weight_scale.shape
        )));
    }
    if weight_shape.shape != [2] {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors shape tensor `{weight_shape_name}` has shape {:?}, expected [2]",
            weight_shape.shape
        )));
    }
    if weight_g_idx.shape != [cols] {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors group index `{weight_g_idx_name}` has shape {:?}, expected [{cols}]",
            weight_g_idx.shape
        )));
    }

    validate_tensor_bytes(weight_packed, 4)?;
    validate_tensor_bytes(
        weight_scale,
        match scale_dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => unreachable!("compressed-tensors scale dtype was validated above"),
        },
    )?;
    validate_tensor_bytes(weight_shape, 8)?;
    validate_tensor_bytes(weight_g_idx, 4)?;

    let shape_data = bundle.require_tensor(weight_shape_name)?.data;
    let stored_shape = shape_data
        .chunks_exact(8)
        .map(|bytes| {
            i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect::<Vec<_>>();
    let expected_shape = [
        i64::try_from(rows).map_err(|_| {
            XrtError::InvalidTensor(format!(
                "compressed-tensors matrix `{canonical}` row count exceeds i64"
            ))
        })?,
        i64::try_from(cols).map_err(|_| {
            XrtError::InvalidTensor(format!(
                "compressed-tensors matrix `{canonical}` column count exceeds i64"
            ))
        })?,
    ];
    if stored_shape.as_slice() != expected_shape {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors shape tensor `{weight_shape_name}` contains {stored_shape:?}, expected {expected_shape:?}"
        )));
    }

    let group_indices = bundle.require_tensor(weight_g_idx_name)?.data;
    let mut group_counts = vec![0usize; groups];
    for (col, bytes) in group_indices.chunks_exact(4).enumerate() {
        let group = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let group = usize::try_from(group).map_err(|_| {
            XrtError::InvalidTensor(format!(
                "compressed-tensors group index `{weight_g_idx_name}` is negative at column {col}"
            ))
        })?;
        if group >= groups {
            return Err(XrtError::InvalidTensor(format!(
                "compressed-tensors group index `{weight_g_idx_name}` has value {group} at column {col}, expected less than {groups}"
            )));
        }
        group_counts[group] = group_counts[group].checked_add(1).ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "compressed-tensors group index `{weight_g_idx_name}` count overflows"
            ))
        })?;
    }
    if let Some((group, count)) = group_counts
        .iter()
        .copied()
        .enumerate()
        .find(|(_, count)| *count != group_size)
    {
        return Err(XrtError::InvalidTensor(format!(
            "compressed-tensors act-order group {group} in `{weight_g_idx_name}` has {count} columns, expected {group_size}"
        )));
    }

    let numel = rows.checked_mul(cols).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "compressed-tensors matrix `{canonical}` element count overflows"
        ))
    })?;
    let byte_len = weight_packed
        .byte_len
        .checked_add(weight_scale.byte_len)
        .and_then(|bytes| bytes.checked_add(weight_shape.byte_len))
        .and_then(|bytes| bytes.checked_add(weight_g_idx.byte_len))
        .ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "compressed-tensors matrix `{canonical}` byte count overflows"
            ))
        })?;

    Ok(ResidentTensorInfo {
        name: canonical.to_string(),
        dimensions: vec![rows, cols],
        dtype: scale_dtype,
        rank: 2,
        rows,
        cols,
        numel,
        byte_len,
        storage: ResidentTensorStorage::CompressedTensorsW4A16 { group_size },
    })
}

fn normalize_hf_tensor(canonical: &str, info: &SafeTensorInfo) -> Result<ResidentTensorInfo> {
    let dtype = safe_float_dtype(&info.name, &info.dtype)?;
    let (rows, cols) = match info.shape.as_slice() {
        [len] => (1, *len),
        [rows, cols] => (*rows, *cols),
        shape => {
            return Err(XrtError::Unsupported(format!(
                "dense SafeTensors tensor `{}` for `{canonical}` must be rank 1 or 2, found shape {shape:?}",
                info.name
            )));
        }
    };
    let numel = info.numel()?;
    let element_bytes = match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => unreachable!("SafeTensors dense dtype was validated above"),
    };
    validate_tensor_bytes(info, element_bytes)?;

    Ok(ResidentTensorInfo {
        name: canonical.to_string(),
        dimensions: info.shape.clone(),
        dtype,
        rank: info.shape.len(),
        rows,
        cols,
        numel,
        byte_len: info.byte_len,
        storage: ResidentTensorStorage::Dense,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use std::{env, fs, path::Path};

    const SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG: &str = r#"{
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "input_activations": null,
                "output_activations": null,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": true,
                    "strategy": "group",
                    "group_size": 32,
                    "dynamic": false,
                    "actorder": "group",
                    "block_structure": null
                }
            }
        },
        "kv_cache_scheme": null,
        "sparsity_config": {"format": "dense", "targets": []}
    }"#;

    struct OwnedTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn push_zero_tensor(
        tensors: &mut Vec<OwnedTensor>,
        name: impl Into<String>,
        dtype: Dtype,
        shape: Vec<usize>,
    ) {
        let element_bytes = match dtype {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::I64 => 8,
            _ => panic!("unsupported synthetic dtype {dtype:?}"),
        };
        let byte_len = shape.iter().product::<usize>() * element_bytes;
        tensors.push(OwnedTensor {
            name: name.into(),
            dtype,
            shape,
            bytes: vec![0; byte_len],
        });
    }

    fn push_i32_tensor(
        tensors: &mut Vec<OwnedTensor>,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<i32>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        tensors.push(OwnedTensor {
            name: name.into(),
            dtype: Dtype::I32,
            shape,
            bytes: values.into_iter().flat_map(i32::to_le_bytes).collect(),
        });
    }

    fn push_i64_tensor(
        tensors: &mut Vec<OwnedTensor>,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<i64>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        tensors.push(OwnedTensor {
            name: name.into(),
            dtype: Dtype::I64,
            shape,
            bytes: values.into_iter().flat_map(i64::to_le_bytes).collect(),
        });
    }

    fn push_awq_linear(
        tensors: &mut Vec<OwnedTensor>,
        base: &str,
        rows: usize,
        cols: usize,
        group_size: usize,
        malformed_qzeros: bool,
    ) {
        let groups = cols / group_size;
        push_zero_tensor(
            tensors,
            format!("{base}.qweight"),
            Dtype::I32,
            vec![cols, rows / 8],
        );
        push_zero_tensor(
            tensors,
            format!("{base}.qzeros"),
            Dtype::I32,
            vec![groups + usize::from(malformed_qzeros), rows / 8],
        );
        push_zero_tensor(
            tensors,
            format!("{base}.scales"),
            Dtype::F16,
            vec![groups, rows],
        );
    }

    fn push_gptq_linear(
        tensors: &mut Vec<OwnedTensor>,
        base: &str,
        rows: usize,
        cols: usize,
        group_size: usize,
        malformed_g_idx: bool,
        act_order: bool,
    ) {
        let groups = cols / group_size;
        push_zero_tensor(
            tensors,
            format!("{base}.qweight"),
            Dtype::I32,
            vec![cols / 8, rows],
        );
        push_zero_tensor(
            tensors,
            format!("{base}.qzeros"),
            Dtype::I32,
            vec![groups, rows / 8],
        );
        push_zero_tensor(
            tensors,
            format!("{base}.scales"),
            Dtype::F16,
            vec![groups, rows],
        );
        let mut g_idx = (0..cols)
            .map(|col| {
                let group = if act_order && groups > 1 {
                    col % groups
                } else {
                    col / group_size
                };
                i32::try_from(group).unwrap()
            })
            .collect::<Vec<_>>();
        if malformed_g_idx {
            *g_idx.last_mut().unwrap() = i32::try_from(groups).unwrap();
        }
        push_i32_tensor(tensors, format!("{base}.g_idx"), vec![cols], g_idx);
    }

    fn push_compressed_tensors_w4a16_linear(
        tensors: &mut Vec<OwnedTensor>,
        base: &str,
        rows: usize,
        cols: usize,
        group_size: usize,
        malformed_g_idx: bool,
        malformed_shape: bool,
    ) {
        let groups = cols / group_size;
        push_zero_tensor(
            tensors,
            format!("{base}.weight_packed"),
            Dtype::I32,
            vec![rows, cols / 8],
        );
        push_zero_tensor(
            tensors,
            format!("{base}.weight_scale"),
            Dtype::BF16,
            vec![rows, groups],
        );
        push_i64_tensor(
            tensors,
            format!("{base}.weight_shape"),
            vec![2],
            vec![
                i64::try_from(rows).unwrap(),
                i64::try_from(cols + usize::from(malformed_shape) * 8).unwrap(),
            ],
        );
        let mut g_idx = (0..cols)
            .map(|col| i32::try_from(col % groups).unwrap())
            .collect::<Vec<_>>();
        if malformed_g_idx {
            *g_idx.last_mut().unwrap() = i32::try_from(groups).unwrap();
        }
        push_i32_tensor(tensors, format!("{base}.weight_g_idx"), vec![cols], g_idx);
    }

    fn write_synthetic_awq_bundle(root: &Path, quantization_config: &str, malformed_qzeros: bool) {
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                    "_name_or_path": "synthetic/qwen2-awq",
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "max_position_embeddings": 64,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": true,
                    "hidden_act": "silu",
                    "torch_dtype": "float16",
                    "vocab_size": 16,
                    "quantization_config": {quantization_config}
                }}"#
            ),
        )
        .unwrap();

        let mut tensors = Vec::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![16, 32]),
            ("model.norm.weight", vec![32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.post_attention_layernorm.weight", vec![32]),
        ] {
            push_zero_tensor(&mut tensors, name, Dtype::F16, shape);
        }
        for (index, (base, rows, cols)) in [
            ("model.layers.0.self_attn.q_proj", 32, 32),
            ("model.layers.0.self_attn.k_proj", 16, 32),
            ("model.layers.0.self_attn.v_proj", 16, 32),
            ("model.layers.0.self_attn.o_proj", 32, 32),
            ("model.layers.0.mlp.gate_proj", 64, 32),
            ("model.layers.0.mlp.up_proj", 64, 32),
            ("model.layers.0.mlp.down_proj", 32, 64),
        ]
        .into_iter()
        .enumerate()
        {
            push_awq_linear(
                &mut tensors,
                base,
                rows,
                cols,
                32,
                malformed_qzeros && index == 0,
            );
        }
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.as_str(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, &None, &root.join("model.safetensors")).unwrap();
    }

    fn write_synthetic_gptq_bundle(root: &Path, quantization_config: &str, malformed_g_idx: bool) {
        write_synthetic_gptq_bundle_with_act_order(
            root,
            quantization_config,
            malformed_g_idx,
            false,
            false,
        );
    }

    fn write_synthetic_gptq_bundle_with_act_order(
        root: &Path,
        quantization_config: &str,
        malformed_g_idx: bool,
        act_order: bool,
        nonzero_auxiliary_bias: bool,
    ) {
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                    "_name_or_path": "synthetic/qwen2-gptq",
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "max_position_embeddings": 64,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": true,
                    "hidden_act": "silu",
                    "torch_dtype": "float16",
                    "vocab_size": 16,
                    "quantization_config": {quantization_config}
                }}"#
            ),
        )
        .unwrap();

        let mut tensors = Vec::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![16, 32]),
            ("model.norm.weight", vec![32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.post_attention_layernorm.weight", vec![32]),
        ] {
            push_zero_tensor(&mut tensors, name, Dtype::F16, shape);
        }
        if act_order {
            for (name, shape) in [
                ("model.layers.0.self_attn.o_proj.bias", vec![32]),
                ("model.layers.0.mlp.gate_proj.bias", vec![64]),
                ("model.layers.0.mlp.up_proj.bias", vec![64]),
                ("model.layers.0.mlp.down_proj.bias", vec![32]),
            ] {
                push_zero_tensor(&mut tensors, name, Dtype::F16, shape);
                if nonzero_auxiliary_bias && name.ends_with("o_proj.bias") {
                    tensors.last_mut().unwrap().bytes[0] = 1;
                }
            }
        }
        for (index, (base, rows, cols)) in [
            ("model.layers.0.self_attn.q_proj", 32, 32),
            ("model.layers.0.self_attn.k_proj", 16, 32),
            ("model.layers.0.self_attn.v_proj", 16, 32),
            ("model.layers.0.self_attn.o_proj", 32, 32),
            ("model.layers.0.mlp.gate_proj", 64, 32),
            ("model.layers.0.mlp.up_proj", 64, 32),
            ("model.layers.0.mlp.down_proj", 32, 64),
        ]
        .into_iter()
        .enumerate()
        {
            push_gptq_linear(
                &mut tensors,
                base,
                rows,
                cols,
                32,
                malformed_g_idx && index == 0,
                act_order,
            );
        }
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.as_str(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, &None, &root.join("model.safetensors")).unwrap();
    }

    fn write_synthetic_compressed_tensors_bundle(
        root: &Path,
        quantization_config: &str,
        malformed_g_idx: bool,
        malformed_shape: bool,
    ) {
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{
                    "_name_or_path": "synthetic/qwen2-compressed-tensors-w4a16",
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "max_position_embeddings": 64,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": false,
                    "hidden_act": "silu",
                    "torch_dtype": "bfloat16",
                    "vocab_size": 16,
                    "quantization_config": {quantization_config}
                }}"#
            ),
        )
        .unwrap();

        let mut tensors = Vec::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![16, 64]),
            ("lm_head.weight", vec![16, 64]),
            ("model.norm.weight", vec![64]),
            ("model.layers.0.input_layernorm.weight", vec![64]),
            ("model.layers.0.post_attention_layernorm.weight", vec![64]),
        ] {
            push_zero_tensor(&mut tensors, name, Dtype::BF16, shape);
        }
        for (index, (base, rows, cols)) in [
            ("model.layers.0.self_attn.q_proj", 64, 64),
            ("model.layers.0.self_attn.k_proj", 32, 64),
            ("model.layers.0.self_attn.v_proj", 32, 64),
            ("model.layers.0.self_attn.o_proj", 64, 64),
            ("model.layers.0.mlp.gate_proj", 128, 64),
            ("model.layers.0.mlp.up_proj", 128, 64),
            ("model.layers.0.mlp.down_proj", 64, 128),
        ]
        .into_iter()
        .enumerate()
        {
            push_compressed_tensors_w4a16_linear(
                &mut tensors,
                base,
                rows,
                cols,
                32,
                malformed_g_idx && index == 0,
                malformed_shape && index == 0,
            );
        }
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.as_str(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, &None, &root.join("model.safetensors")).unwrap();
    }

    #[test]
    fn gguf_metadata_is_normalized_without_changing_row_geometry() {
        let gguf = TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dimensions: vec![64, 3],
            strides: vec![1, 64],
            dtype: DType::Q8_0,
            offset: 128,
            nbytes: 204,
        };

        let info = ResidentTensorInfo::from_gguf(&gguf);
        assert_eq!(info.name, gguf.name);
        assert_eq!(info.dimensions, vec![64, 3]);
        assert_eq!(info.dtype, DType::Q8_0);
        assert_eq!(info.rank, 2);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 64);
        assert_eq!(info.numel, 192);
        assert_eq!(info.byte_len, 204);
        assert_eq!(info.storage, ResidentTensorStorage::Dense);
    }

    #[test]
    fn synthetic_autoawq_gemm_source_maps_versioned_tensor_groups() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_awq_bundle(
            directory.path(),
            r#"{
                "quant_method": "awq",
                "w_bit": 4,
                "q_group_size": 32,
                "zero_point": true,
                "version": "GEMM"
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let source = HfQwen2ResidentTensorSource::new(&bundle).unwrap();

        assert_eq!(bundle.tensor_count(), 25);
        assert_eq!(source.tensor_infos().len(), 11);
        let q = source.require_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.rows, 32);
        assert_eq!(q.cols, 32);
        assert_eq!(q.dtype, DType::F16);
        assert_eq!(
            q.storage,
            ResidentTensorStorage::AwqGemm4 { group_size: 32 }
        );
        assert!(source.tensor_data("blk.0.attn_q.weight").is_err());
        let q_data = source
            .awq_gemm4_data("blk.0.attn_q.weight")
            .unwrap()
            .unwrap();
        assert_eq!(q_data.qweight.len(), 32 * 4 * 4);
        assert_eq!(q_data.qzeros.len(), 4 * 4);
        assert_eq!(q_data.scales.len(), 32 * 2);
        assert_eq!(q_data.scale_dtype, DType::F16);

        let down = source.require_tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!(down.rows, 32);
        assert_eq!(down.cols, 64);
        assert_eq!(
            down.storage,
            ResidentTensorStorage::AwqGemm4 { group_size: 32 }
        );
        let embedding = source.require_tensor("token_embd.weight").unwrap();
        assert_eq!(embedding.storage, ResidentTensorStorage::Dense);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn synthetic_autoawq_runtime_executes_full_cuda_decode() -> Result<()> {
        use crate::{
            backend::{CausalLmBackend, CudaResidentBackend},
            gpu_resource::GpuResourceConfig,
            kv_cache::KvCacheMode,
        };

        let directory = tempfile::tempdir()?;
        write_synthetic_awq_bundle(
            directory.path(),
            r#"{
                "quant_method": "awq",
                "w_bit": 4,
                "q_group_size": 32,
                "zero_point": true,
                "version": "GEMM"
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path())?;
        let backend = CudaResidentBackend::from_hf_bundle(&bundle, GpuResourceConfig::default())?;
        assert!(backend.resident_dense_quant_decode_available());

        let mut session = backend.new_session(KvCacheMode::F32, 16);
        let mut logits = Vec::new();
        backend.forward_token(0, 0, &mut session, &mut logits)?;
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn synthetic_autoawq_source_rejects_wrong_packed_geometry() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_awq_bundle(
            directory.path(),
            r#"{
                "quant_method": "awq",
                "w_bit": 4,
                "q_group_size": 32,
                "zero_point": true,
                "version": "GEMM"
            }"#,
            true,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("malformed qzeros shape must fail");
        assert!(error.to_string().contains("qzeros"), "{error}");
    }

    #[test]
    fn synthetic_gptq_v1_source_maps_versioned_tensor_groups() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "desc_act": false,
                "exllama_config": {"version": 1}
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let source = HfQwen2ResidentTensorSource::new(&bundle).unwrap();

        assert_eq!(bundle.tensor_count(), 32);
        assert_eq!(source.tensor_infos().len(), 11);
        let q = source.require_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!((q.rows, q.cols), (32, 32));
        assert_eq!(
            q.storage,
            ResidentTensorStorage::GptqGemm4 { group_size: 32 }
        );
        assert!(source.tensor_data("blk.0.attn_q.weight").is_err());
        let q_data = source
            .gptq_gemm4_data("blk.0.attn_q.weight")
            .unwrap()
            .unwrap();
        assert_eq!(q_data.qweight.len(), 32 * 4 * 4);
        assert_eq!(q_data.qzeros.len(), 4 * 4);
        assert_eq!(q_data.scales.len(), 32 * 2);
        assert_eq!(q_data.scale_dtype, DType::F16);

        let down = source.require_tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!((down.rows, down.cols), (32, 64));
        assert_eq!(
            down.storage,
            ResidentTensorStorage::GptqGemm4 { group_size: 32 }
        );
        let embedding = source.require_tensor("token_embd.weight").unwrap();
        assert_eq!(embedding.storage, ResidentTensorStorage::Dense);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn synthetic_gptq_runtime_executes_full_cuda_decode() -> Result<()> {
        use crate::{
            backend::{CausalLmBackend, CudaResidentBackend},
            gpu_resource::GpuResourceConfig,
            kv_cache::KvCacheMode,
        };

        let directory = tempfile::tempdir()?;
        write_synthetic_gptq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "desc_act": false,
                "exllama_config": {"version": 1}
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path())?;
        let backend = CudaResidentBackend::from_hf_bundle(&bundle, GpuResourceConfig::default())?;
        assert!(backend.resident_dense_quant_decode_available());

        let mut session = backend.new_session(KvCacheMode::F32, 16);
        let mut logits = Vec::new();
        backend.forward_token(0, 0, &mut session, &mut logits)?;
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn synthetic_gptq_source_rejects_nonstandard_groups_without_desc_act() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "desc_act": false,
                "exllama_config": {"version": 1}
            }"#,
            true,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("nonstandard GPTQ g_idx must fail");
        assert!(error.to_string().contains("act-order"), "{error}");
    }

    #[test]
    fn synthetic_gptq_v1_act_order_source_maps_explicit_groups() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle_with_act_order(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "desc_act": true,
                "exllama_config": {"version": 1}
            }"#,
            false,
            true,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let source = HfQwen2ResidentTensorSource::new(&bundle).unwrap();
        assert_eq!(bundle.tensor_count(), 36);
        assert_eq!(source.tensor_infos().len(), 11);
        let down = source.require_tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!(
            down.storage,
            ResidentTensorStorage::GptqExplicitGemm4 {
                group_size: 32,
                zero_encoding: GptqZeroEncoding::V1MinusOne,
            }
        );
        let data = source
            .gptq_explicit_gemm4_data("blk.0.ffn_down.weight")
            .unwrap()
            .unwrap();
        let groups = data
            .group_indices
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        assert_eq!(&groups[..8], &[0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(data.zero_encoding, GptqZeroEncoding::V1MinusOne);
    }

    #[test]
    fn synthetic_gptq_source_rejects_nonzero_auxiliary_bias() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle_with_act_order(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "desc_act": true,
                "exllama_config": {"version": 1}
            }"#,
            false,
            true,
            true,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("nonzero auxiliary GPTQ bias must fail");
        assert!(error.to_string().contains("nonzero"), "{error}");
    }

    #[test]
    fn synthetic_gptq_v2_source_maps_direct_zero_encoding() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": false,
                "desc_act": false,
                "checkpoint_format": "gptq_v2",
                "format": "gptq_v2",
                "pack_dtype": "int32"
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let source = HfQwen2ResidentTensorSource::new(&bundle).unwrap();
        let q = source.require_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(
            q.storage,
            ResidentTensorStorage::GptqExplicitGemm4 {
                group_size: 32,
                zero_encoding: GptqZeroEncoding::V2Direct,
            }
        );
        let data = source
            .gptq_explicit_gemm4_data("blk.0.attn_q.weight")
            .unwrap()
            .unwrap();
        assert_eq!(data.zero_encoding, GptqZeroEncoding::V2Direct);
    }

    #[test]
    fn synthetic_gptq_source_rejects_missing_desc_act() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_gptq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "exllama_config": {"version": 1}
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("missing GPTQ desc_act must fail");
        assert!(error.to_string().contains("desc_act"), "{error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn synthetic_gptq_explicit_runtime_executes_v1_act_order_and_v2_decode() -> Result<()> {
        use crate::{
            backend::{CausalLmBackend, CudaResidentBackend},
            gpu_resource::GpuResourceConfig,
            kv_cache::KvCacheMode,
        };

        for (config, act_order) in [
            (
                r#"{
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 32,
                    "sym": true,
                    "desc_act": true,
                    "exllama_config": {"version": 1}
                }"#,
                true,
            ),
            (
                r#"{
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 32,
                    "sym": false,
                    "desc_act": false,
                    "checkpoint_format": "gptq_v2",
                    "format": "gptq_v2",
                    "pack_dtype": "int32"
                }"#,
                false,
            ),
        ] {
            let directory = tempfile::tempdir()?;
            write_synthetic_gptq_bundle_with_act_order(
                directory.path(),
                config,
                false,
                act_order,
                false,
            );
            let bundle = HfModelBundle::open(directory.path())?;
            let backend =
                CudaResidentBackend::from_hf_bundle(&bundle, GpuResourceConfig::default())?;
            assert!(backend.resident_dense_quant_decode_available());

            let mut session = backend.new_session(KvCacheMode::F32, 16);
            let mut logits = Vec::new();
            backend.forward_token(0, 0, &mut session, &mut logits)?;
            assert_eq!(logits.len(), 16);
            assert!(logits.iter().all(|value| value.is_finite()));
        }
        Ok(())
    }

    #[test]
    fn synthetic_compressed_tensors_w4a16_source_maps_permuted_groups() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG,
            false,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let source = HfQwen2ResidentTensorSource::new(&bundle).unwrap();

        assert_eq!(bundle.tensor_count(), 33);
        assert_eq!(source.tensor_infos().len(), 12);
        let q = source.require_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!((q.rows, q.cols), (64, 64));
        assert_eq!(q.dtype, DType::BF16);
        assert_eq!(
            q.storage,
            ResidentTensorStorage::CompressedTensorsW4A16 { group_size: 32 }
        );
        assert!(source.tensor_data("blk.0.attn_q.weight").is_err());
        let q_data = source
            .compressed_tensors_w4a16_data("blk.0.attn_q.weight")
            .unwrap()
            .unwrap();
        assert_eq!(q_data.weight_packed.len(), 64 * 8 * 4);
        assert_eq!(q_data.scales.len(), 64 * 2 * 2);
        assert_eq!(q_data.group_indices.len(), 64 * 4);
        assert_eq!(q_data.scale_dtype, DType::BF16);
        let groups = q_data
            .group_indices
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        assert_eq!(&groups[..8], &[0, 1, 0, 1, 0, 1, 0, 1]);

        let down = source.require_tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!((down.rows, down.cols), (64, 128));
        assert_eq!(
            down.storage,
            ResidentTensorStorage::CompressedTensorsW4A16 { group_size: 32 }
        );
        assert_eq!(
            source.require_tensor("output.weight").unwrap().storage,
            ResidentTensorStorage::Dense
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn synthetic_compressed_tensors_w4a16_runtime_executes_full_cuda_decode() -> Result<()> {
        use crate::{
            backend::{CausalLmBackend, CudaResidentBackend},
            gpu_resource::GpuResourceConfig,
            kv_cache::KvCacheMode,
        };

        let directory = tempfile::tempdir()?;
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG,
            false,
            false,
        );
        let bundle = HfModelBundle::open(directory.path())?;
        let backend = CudaResidentBackend::from_hf_bundle(&bundle, GpuResourceConfig::default())?;
        assert!(backend.resident_dense_quant_decode_available());

        let mut session = backend.new_session(KvCacheMode::F32, 16);
        let mut logits = Vec::new();
        backend.forward_token(0, 0, &mut session, &mut logits)?;
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn synthetic_compressed_tensors_source_rejects_wrong_format() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            &SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG
                .replace("\"pack-quantized\"", "\"fake-quantized\""),
            false,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("wrong compressed-tensors format must fail");
        assert!(error.to_string().contains("pack-quantized"), "{error}");
    }

    #[test]
    fn synthetic_compressed_tensors_source_rejects_activation_quantization() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            &SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG.replace(
                "\"input_activations\": null",
                "\"input_activations\": {\"num_bits\": 8}",
            ),
            false,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("activation quantization must fail");
        assert!(
            error.to_string().contains("unquantized activations"),
            "{error}"
        );
    }

    #[test]
    fn synthetic_compressed_tensors_source_rejects_asymmetric_weights() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            &SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG
                .replace("\"symmetric\": true", "\"symmetric\": false"),
            false,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("asymmetric compressed-tensors weights must fail");
        assert!(error.to_string().contains("symmetric weights"), "{error}");
    }

    #[test]
    fn synthetic_compressed_tensors_source_rejects_malformed_group_indices() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG,
            true,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("malformed compressed-tensors g_idx must fail");
        assert!(error.to_string().contains("group index"), "{error}");
    }

    #[test]
    fn synthetic_compressed_tensors_source_rejects_shape_payload_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_compressed_tensors_bundle(
            directory.path(),
            SUPPORTED_COMPRESSED_TENSORS_W4A16_CONFIG,
            false,
            true,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("compressed-tensors weight_shape mismatch must fail");
        assert!(error.to_string().contains("shape tensor"), "{error}");
    }

    #[test]
    #[ignore = "requires XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR with the pinned Qwen2.5 0.5B W4A16 bundle"]
    fn real_compressed_tensors_qwen2_source_maps_every_packed_tensor() -> Result<()> {
        let root = env::var("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR").map_err(|_| {
            XrtError::Runtime("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR is required".to_string())
        })?;
        let bundle = HfModelBundle::open(root)?;
        assert_eq!(bundle.shard_count(), 1);
        assert_eq!(bundle.tensor_count(), 795);

        let quantization = bundle.config().quantization.as_ref().ok_or_else(|| {
            XrtError::InvalidMetadata(
                "real compressed-tensors fixture has no quantization config".to_string(),
            )
        })?;
        assert_eq!(quantization.method, HfQuantizationMethod::CompressedTensors);
        assert_eq!(quantization.format.as_deref(), Some("pack-quantized"));
        assert_eq!(
            quantization
                .raw
                .get("quantization_status")
                .and_then(|value| value.as_str()),
            Some("compressed")
        );

        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let infos = source.tensor_infos();
        assert_eq!(infos.len(), 291);
        assert_eq!(
            infos
                .iter()
                .filter(|info| matches!(
                    info.storage,
                    ResidentTensorStorage::CompressedTensorsW4A16 { group_size: 64 }
                ))
                .count(),
            168
        );

        let embedding = source.require_tensor("token_embd.weight")?;
        assert_eq!(embedding.dtype, DType::BF16);
        assert_eq!((embedding.rows, embedding.cols), (151936, 896));
        assert_eq!(embedding.storage, ResidentTensorStorage::Dense);
        let output = source.require_tensor("output.weight")?;
        assert_eq!(output.dtype, DType::BF16);
        assert_eq!((output.rows, output.cols), (151936, 896));
        assert_eq!(output.storage, ResidentTensorStorage::Dense);

        let q = source.require_tensor("blk.0.attn_q.weight")?;
        assert_eq!((q.rows, q.cols), (896, 896));
        assert_eq!(
            q.storage,
            ResidentTensorStorage::CompressedTensorsW4A16 { group_size: 64 }
        );
        let k = source.require_tensor("blk.0.attn_k.weight")?;
        assert_eq!((k.rows, k.cols), (128, 896));
        let down = source.require_tensor("blk.0.ffn_down.weight")?;
        assert_eq!((down.rows, down.cols), (896, 4864));

        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.weight_packed")?
                .info
                .shape,
            vec![896, 112]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.weight_scale")?
                .info
                .shape,
            vec![896, 14]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.weight_shape")?
                .info
                .shape,
            vec![2]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.weight_g_idx")?
                .info
                .shape,
            vec![896]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR and a CUDA-capable device"]
    fn real_compressed_tensors_qwen2_kernels_match_host_dequantization() -> Result<()> {
        use xrt_cuda::CudaDevice;

        let root = env::var("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR").map_err(|_| {
            XrtError::Runtime("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR is required".to_string())
        })?;
        let bundle = HfModelBundle::open(root)?;
        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let device = CudaDevice::new(0)?;

        for name in ["blk.0.attn_q.weight", "blk.23.ffn_down.weight"] {
            let data = source.compressed_tensors_w4a16_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "missing real compressed-tensors data for `{name}`"
                ))
            })?;
            let input = (0..data.cols)
                .map(|index| ((index % 31) as f32 - 15.0) / 127.0)
                .collect::<Vec<_>>();
            let expected = host_compressed_tensors_w4a16_matvec(&data, &input)?;
            let matrix = device.upload_compressed_tensors_w4a16_matrix(
                data.weight_packed,
                data.scales,
                data.scale_dtype,
                data.group_indices,
                data.rows,
                data.cols,
                data.group_size,
            )?;
            let actual = device.matvec_compressed_tensors_w4a16_resident(&matrix, &input)?;
            assert_eq!(actual.len(), expected.len());
            for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 0.003 + expected.abs() * 0.0002;
                let delta = (actual - expected).abs();
                assert!(
                    delta <= tolerance,
                    "real compressed-tensors `{name}` row {row} differs: actual={actual}, expected={expected}, delta={delta}, tolerance={tolerance}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn host_compressed_tensors_w4a16_matvec(
        data: &ResidentCompressedTensorsW4A16Data<'_>,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != data.cols {
            return Err(XrtError::InvalidTensor(format!(
                "host compressed-tensors input has {} values, expected {}",
                input.len(),
                data.cols
            )));
        }
        let packed_cols = data.cols / 8;
        let groups = data.cols / data.group_size;
        let mut output = vec![0.0f32; data.rows];
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (col, &input_value) in input.iter().enumerate() {
                let word = read_u32(data.weight_packed, row * packed_cols + col / 8)?;
                let quant = ((word >> ((col % 8) * 4)) & 0x0f) as i32 - 8;
                let group = usize::try_from(read_i32(data.group_indices, col)?).map_err(|_| {
                    XrtError::InvalidTensor(format!(
                        "host compressed-tensors group index is negative at column {col}"
                    ))
                })?;
                if group >= groups {
                    return Err(XrtError::InvalidTensor(format!(
                        "host compressed-tensors group index {group} at column {col} exceeds {groups} groups"
                    )));
                }
                let scale = read_float(data.scales, data.scale_dtype, row * groups + group)?;
                sum += input_value * quant as f32 * scale;
            }
            *output_value = sum;
        }
        Ok(output)
    }

    #[test]
    #[ignore = "requires XRT_REAL_AWQ_MODEL_DIR with the pinned Qwen2.5 0.5B AutoAWQ bundle"]
    fn real_autoawq_qwen2_source_maps_every_packed_tensor() -> Result<()> {
        let root = env::var("XRT_REAL_AWQ_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_AWQ_MODEL_DIR is required".to_string()))?;
        let bundle = HfModelBundle::open(root)?;
        assert_eq!(bundle.shard_count(), 1);
        assert_eq!(bundle.tensor_count(), 627);

        let quantization = bundle.config().quantization.as_ref().ok_or_else(|| {
            XrtError::InvalidMetadata("real AutoAWQ fixture has no quantization config".to_string())
        })?;
        assert_eq!(quantization.method, HfQuantizationMethod::Awq);
        assert_eq!(quantization.bits, Some(4));
        assert_eq!(quantization.group_size, Some(128));
        assert_eq!(quantization.zero_point, Some(true));
        assert_eq!(quantization.format.as_deref(), Some("gemm"));

        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let infos = source.tensor_infos();
        assert_eq!(infos.len(), 291);
        assert_eq!(
            infos
                .iter()
                .filter(|info| matches!(
                    info.storage,
                    ResidentTensorStorage::AwqGemm4 { group_size: 128 }
                ))
                .count(),
            168
        );

        let embedding = source.require_tensor("token_embd.weight")?;
        assert_eq!(embedding.dtype, DType::F16);
        assert_eq!((embedding.rows, embedding.cols), (151936, 896));
        assert_eq!(embedding.storage, ResidentTensorStorage::Dense);
        let output = source.require_tensor("output.weight")?;
        assert_eq!((output.rows, output.cols), (151936, 896));
        assert_eq!(output.storage, ResidentTensorStorage::Dense);

        let q = source.require_tensor("blk.0.attn_q.weight")?;
        assert_eq!((q.rows, q.cols), (896, 896));
        assert_eq!(
            q.storage,
            ResidentTensorStorage::AwqGemm4 { group_size: 128 }
        );
        let k = source.require_tensor("blk.0.attn_k.weight")?;
        assert_eq!((k.rows, k.cols), (128, 896));
        let down = source.require_tensor("blk.0.ffn_down.weight")?;
        assert_eq!((down.rows, down.cols), (896, 4864));

        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.qweight")?
                .info
                .shape,
            vec![896, 112]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.qzeros")?
                .info
                .shape,
            vec![7, 112]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.scales")?
                .info
                .shape,
            vec![7, 896]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR with the pinned Qwen2.5 1.5B GPTQ act-order bundle"]
    fn real_gptq_v1_act_order_qwen2_source_maps_every_packed_tensor() -> Result<()> {
        let root = env::var("XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR").map_err(|_| {
            XrtError::Runtime("XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR is required".to_string())
        })?;
        let bundle = HfModelBundle::open(root)?;
        assert_eq!(bundle.shard_count(), 1);
        assert_eq!(bundle.tensor_count(), 1038);

        let quantization = bundle.config().quantization.as_ref().ok_or_else(|| {
            XrtError::InvalidMetadata(
                "real GPTQ act-order fixture has no quantization config".to_string(),
            )
        })?;
        assert_eq!(quantization.method, HfQuantizationMethod::Gptq);
        assert_eq!(quantization.bits, Some(4));
        assert_eq!(quantization.group_size, Some(64));
        assert_eq!(quantization.zero_point, Some(false));
        assert_eq!(quantization.desc_act, Some(true));

        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let infos = source.tensor_infos();
        assert_eq!(infos.len(), 338);
        assert_eq!(
            infos
                .iter()
                .filter(|info| matches!(
                    info.storage,
                    ResidentTensorStorage::GptqExplicitGemm4 {
                        group_size: 64,
                        zero_encoding: GptqZeroEncoding::V1MinusOne,
                    }
                ))
                .count(),
            196
        );

        let embedding = source.require_tensor("token_embd.weight")?;
        assert_eq!(embedding.dtype, DType::F16);
        assert_eq!((embedding.rows, embedding.cols), (151936, 1536));
        assert!(source.tensor_info("output.weight").is_none());

        let q = source.require_tensor("blk.0.attn_q.weight")?;
        assert_eq!((q.rows, q.cols), (1536, 1536));
        assert_eq!(
            q.storage,
            ResidentTensorStorage::GptqExplicitGemm4 {
                group_size: 64,
                zero_encoding: GptqZeroEncoding::V1MinusOne,
            }
        );
        let q_data = source
            .gptq_explicit_gemm4_data("blk.0.attn_q.weight")?
            .ok_or_else(|| XrtError::InvalidTensor("missing act-order q_proj data".to_string()))?;
        let q_groups = decode_i32_values(q_data.group_indices);
        assert_eq!(q_groups.len(), 1536);
        assert_eq!(&q_groups[..8], &[5, 14, 7, 4, 13, 3, 21, 13]);
        let mut q_group_counts = [0usize; 24];
        for &group in &q_groups {
            let group = usize::try_from(group).map_err(|_| {
                XrtError::InvalidTensor("real act-order q_proj has a negative group".to_string())
            })?;
            *q_group_counts.get_mut(group).ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "real act-order q_proj group {group} is out of range"
                ))
            })? += 1;
        }
        assert!(q_group_counts.iter().all(|count| *count == 64));
        assert_eq!(
            q_groups
                .iter()
                .enumerate()
                .filter(|(col, group)| **group != (*col / 64) as i32)
                .count(),
            1473
        );

        let down = source.require_tensor("blk.0.ffn_down.weight")?;
        assert_eq!((down.rows, down.cols), (1536, 8960));
        let down_data = source
            .gptq_explicit_gemm4_data("blk.0.ffn_down.weight")?
            .ok_or_else(|| {
                XrtError::InvalidTensor("missing act-order down_proj data".to_string())
            })?;
        let down_groups = decode_i32_values(down_data.group_indices);
        assert_eq!(down_groups.len(), 8960);
        assert_eq!(
            down_groups
                .iter()
                .enumerate()
                .filter(|(col, group)| **group != (*col / 64) as i32)
                .count(),
            8899
        );

        for suffix in [
            "self_attn.o_proj.bias",
            "mlp.gate_proj.bias",
            "mlp.up_proj.bias",
            "mlp.down_proj.bias",
        ] {
            let name = format!("model.layers.0.{suffix}");
            let bias = bundle.require_tensor(&name)?;
            assert!(
                bias.data.iter().all(|byte| *byte == 0),
                "real GPTQ auxiliary bias `{name}` must be exact zero"
            );
        }
        assert!(source.tensor_info("blk.0.attn_output.bias").is_none());
        assert!(source.tensor_info("blk.0.ffn_down.bias").is_none());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR and a CUDA-capable device"]
    fn real_gptq_v1_act_order_qwen2_kernels_match_host_dequantization() -> Result<()> {
        use xrt_cuda::CudaDevice;

        let root = env::var("XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR").map_err(|_| {
            XrtError::Runtime("XRT_REAL_GPTQ_ACT_ORDER_MODEL_DIR is required".to_string())
        })?;
        let bundle = HfModelBundle::open(root)?;
        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let device = CudaDevice::new(0)?;
        assert_real_gptq_explicit_kernels_match_host(
            &source,
            &device,
            &["blk.0.attn_q.weight", "blk.27.ffn_down.weight"],
        )
    }

    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_V1_MODEL_DIR and XRT_REAL_GPTQ_V2_MODEL_DIR"]
    fn real_derived_gptq_v2_qwen2_source_maps_direct_zero_semantics() -> Result<()> {
        let v1_root = env::var("XRT_REAL_GPTQ_V1_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_GPTQ_V1_MODEL_DIR is required".to_string()))?;
        let v2_root = env::var("XRT_REAL_GPTQ_V2_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_GPTQ_V2_MODEL_DIR is required".to_string()))?;
        let v1_bundle = HfModelBundle::open(v1_root)?;
        let v2_bundle = HfModelBundle::open(v2_root)?;
        assert_eq!(v2_bundle.shard_count(), 1);
        assert_eq!(v2_bundle.tensor_count(), 794);

        let quantization = v2_bundle.config().quantization.as_ref().ok_or_else(|| {
            XrtError::InvalidMetadata(
                "derived GPTQ v2 fixture has no quantization config".to_string(),
            )
        })?;
        assert_eq!(quantization.method, HfQuantizationMethod::Gptq);
        assert_eq!(quantization.bits, Some(4));
        assert_eq!(quantization.group_size, Some(128));
        assert_eq!(quantization.desc_act, Some(false));
        assert_eq!(quantization.format.as_deref(), Some("gptq_v2"));

        let v1_source = HfQwen2ResidentTensorSource::new(&v1_bundle)?;
        let v2_source = HfQwen2ResidentTensorSource::new(&v2_bundle)?;
        let infos = v2_source.tensor_infos();
        assert_eq!(infos.len(), 290);
        assert_eq!(
            infos
                .iter()
                .filter(|info| matches!(
                    info.storage,
                    ResidentTensorStorage::GptqExplicitGemm4 {
                        group_size: 128,
                        zero_encoding: GptqZeroEncoding::V2Direct,
                    }
                ))
                .count(),
            168
        );

        for name in ["blk.0.attn_q.weight", "blk.23.ffn_down.weight"] {
            let v1_data = v1_source.gptq_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!("missing GPTQ v1 data for `{name}`"))
            })?;
            let v2_data = v2_source.gptq_explicit_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!("missing GPTQ v2 data for `{name}`"))
            })?;
            assert_eq!(v2_data.zero_encoding, GptqZeroEncoding::V2Direct);
            assert_eq!(v2_data.qweight, v1_data.qweight);
            assert_eq!(v2_data.scales, v1_data.scales);
            assert_eq!((v2_data.rows, v2_data.cols), (v1_data.rows, v1_data.cols));
            assert_eq!(v2_data.group_size, v1_data.group_size);
            assert_eq!(v2_data.qzeros.len(), v1_data.qzeros.len());
            for (&v1_byte, &v2_byte) in v1_data.qzeros.iter().zip(v2_data.qzeros) {
                assert_eq!(v2_byte & 0x0f, ((v1_byte & 0x0f) + 1) & 0x0f);
                assert_eq!(v2_byte >> 4, ((v1_byte >> 4) + 1) & 0x0f);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_V2_MODEL_DIR and a CUDA-capable device"]
    fn real_derived_gptq_v2_qwen2_kernels_match_host_dequantization() -> Result<()> {
        use xrt_cuda::CudaDevice;

        let root = env::var("XRT_REAL_GPTQ_V2_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_GPTQ_V2_MODEL_DIR is required".to_string()))?;
        let bundle = HfModelBundle::open(root)?;
        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let device = CudaDevice::new(0)?;
        assert_real_gptq_explicit_kernels_match_host(
            &source,
            &device,
            &["blk.0.attn_q.weight", "blk.23.ffn_down.weight"],
        )
    }

    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_MODEL_DIR with the pinned Qwen2.5 0.5B GPTQ v1 bundle"]
    fn real_gptq_v1_qwen2_source_maps_every_packed_tensor() -> Result<()> {
        let root = env::var("XRT_REAL_GPTQ_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_GPTQ_MODEL_DIR is required".to_string()))?;
        let bundle = HfModelBundle::open(root)?;
        assert_eq!(bundle.shard_count(), 1);
        assert_eq!(bundle.tensor_count(), 794);

        let quantization = bundle.config().quantization.as_ref().ok_or_else(|| {
            XrtError::InvalidMetadata("real GPTQ fixture has no quantization config".to_string())
        })?;
        assert_eq!(quantization.method, HfQuantizationMethod::Gptq);
        assert_eq!(quantization.bits, Some(4));
        assert_eq!(quantization.group_size, Some(128));
        assert_eq!(quantization.zero_point, Some(false));
        assert_eq!(quantization.desc_act, Some(false));
        assert_eq!(
            quantization
                .raw
                .get("exllama_config")
                .and_then(|value| value.get("version"))
                .and_then(|value| value.as_u64()),
            Some(1)
        );

        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let infos = source.tensor_infos();
        assert_eq!(infos.len(), 290);
        assert_eq!(
            infos
                .iter()
                .filter(|info| matches!(
                    info.storage,
                    ResidentTensorStorage::GptqGemm4 { group_size: 128 }
                ))
                .count(),
            168
        );

        let embedding = source.require_tensor("token_embd.weight")?;
        assert_eq!(embedding.dtype, DType::F16);
        assert_eq!((embedding.rows, embedding.cols), (151936, 896));
        assert_eq!(embedding.storage, ResidentTensorStorage::Dense);
        assert!(source.tensor_info("output.weight").is_none());

        let q = source.require_tensor("blk.0.attn_q.weight")?;
        assert_eq!((q.rows, q.cols), (896, 896));
        assert_eq!(
            q.storage,
            ResidentTensorStorage::GptqGemm4 { group_size: 128 }
        );
        let k = source.require_tensor("blk.0.attn_k.weight")?;
        assert_eq!((k.rows, k.cols), (128, 896));
        let down = source.require_tensor("blk.0.ffn_down.weight")?;
        assert_eq!((down.rows, down.cols), (896, 4864));

        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.qweight")?
                .info
                .shape,
            vec![112, 896]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.qzeros")?
                .info
                .shape,
            vec![7, 112]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.scales")?
                .info
                .shape,
            vec![7, 896]
        );
        assert_eq!(
            bundle
                .require_tensor("model.layers.0.self_attn.q_proj.g_idx")?
                .info
                .shape,
            vec![896]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires XRT_REAL_GPTQ_MODEL_DIR and a CUDA-capable device"]
    fn real_gptq_v1_qwen2_kernels_match_host_dequantization() -> Result<()> {
        use xrt_cuda::CudaDevice;

        let root = env::var("XRT_REAL_GPTQ_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_GPTQ_MODEL_DIR is required".to_string()))?;
        let bundle = HfModelBundle::open(root)?;
        let source = HfQwen2ResidentTensorSource::new(&bundle)?;
        let device = CudaDevice::new(0)?;

        for name in ["blk.0.attn_q.weight", "blk.23.ffn_down.weight"] {
            let data = source.gptq_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!("missing real GPTQ data for `{name}`"))
            })?;
            let input = (0..data.cols)
                .map(|index| ((index % 31) as f32 - 15.0) / 127.0)
                .collect::<Vec<_>>();
            let expected = host_gptq_gemm4_matvec(&data, &input)?;
            let matrix = device.upload_gptq_gemm4_matrix(
                data.qweight,
                data.qzeros,
                data.scales,
                data.scale_dtype,
                data.rows,
                data.cols,
                data.group_size,
            )?;
            let actual = device.matvec_gptq_gemm4_resident(&matrix, &input)?;
            assert_eq!(actual.len(), expected.len());
            for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 0.002 + expected.abs() * 0.0001;
                let delta = (actual - expected).abs();
                assert!(
                    delta <= tolerance,
                    "real GPTQ `{name}` row {row} differs: actual={actual}, expected={expected}, delta={delta}, tolerance={tolerance}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn assert_real_gptq_explicit_kernels_match_host(
        source: &HfQwen2ResidentTensorSource<'_>,
        device: &xrt_cuda::CudaDevice,
        names: &[&str],
    ) -> Result<()> {
        for &name in names {
            let data = source.gptq_explicit_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!("missing real explicit GPTQ data for `{name}`"))
            })?;
            let input = (0..data.cols)
                .map(|index| ((index % 31) as f32 - 15.0) / 127.0)
                .collect::<Vec<_>>();
            let expected = host_gptq_explicit_gemm4_matvec(&data, &input)?;
            let matrix = device.upload_gptq_explicit_gemm4_matrix(
                data.qweight,
                data.qzeros,
                data.scales,
                data.scale_dtype,
                data.group_indices,
                data.rows,
                data.cols,
                data.group_size,
                data.zero_encoding,
            )?;
            let actual = device.matvec_gptq_explicit_gemm4_resident(&matrix, &input)?;
            assert_eq!(actual.len(), expected.len());
            for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 0.002 + expected.abs() * 0.0001;
                let delta = (actual - expected).abs();
                assert!(
                    delta <= tolerance,
                    "real explicit GPTQ `{name}` row {row} differs: actual={actual}, expected={expected}, delta={delta}, tolerance={tolerance}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn host_gptq_explicit_gemm4_matvec(
        data: &ResidentGptqExplicitGemm4Data<'_>,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != data.cols {
            return Err(XrtError::InvalidTensor(format!(
                "host explicit GPTQ input has {} values, expected {}",
                input.len(),
                data.cols
            )));
        }
        let packed_rows = data.rows / 8;
        let groups = data.cols / data.group_size;
        let mut output = vec![0.0f32; data.rows];
        for row in 0..data.rows {
            let packed_row = row / 8;
            let zero_shift = (row % 8) * 4;
            let mut sum = 0.0f32;
            for (col, &input_value) in input.iter().enumerate() {
                let group = usize::try_from(read_i32(data.group_indices, col)?).map_err(|_| {
                    XrtError::InvalidTensor(format!(
                        "host explicit GPTQ group index is negative at column {col}"
                    ))
                })?;
                if group >= groups {
                    return Err(XrtError::InvalidTensor(format!(
                        "host explicit GPTQ group index {group} at column {col} exceeds {groups} groups"
                    )));
                }
                let packed_col = col / 8;
                let weight_shift = (col % 8) * 4;
                let weight_word = read_u32(data.qweight, packed_col * data.rows + row)?;
                let zero_word = read_u32(data.qzeros, group * packed_rows + packed_row)?;
                let quant = ((weight_word >> weight_shift) & 0x0f) as i32;
                let encoded_zero = ((zero_word >> zero_shift) & 0x0f) as i32;
                let zero = match data.zero_encoding {
                    GptqZeroEncoding::V1MinusOne => (encoded_zero + 1) & 0x0f,
                    GptqZeroEncoding::V2Direct => encoded_zero,
                };
                let scale = read_float(data.scales, data.scale_dtype, group * data.rows + row)?;
                sum += input_value * (quant - zero) as f32 * scale;
            }
            output[row] = sum;
        }
        Ok(output)
    }

    fn decode_i32_values(bytes: &[u8]) -> Vec<i32> {
        assert_eq!(bytes.len() % 4, 0, "I32 payload must contain full words");
        bytes
            .chunks_exact(4)
            .map(|value| i32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn host_gptq_gemm4_matvec(data: &ResidentGptqGemm4Data<'_>, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != data.cols {
            return Err(XrtError::InvalidTensor(format!(
                "host GPTQ input has {} values, expected {}",
                input.len(),
                data.cols
            )));
        }
        let packed_rows = data.rows / 8;
        let mut output = vec![0.0f32; data.rows];
        for row in 0..data.rows {
            let packed_row = row / 8;
            let zero_shift = (row % 8) * 4;
            let mut sum = 0.0f32;
            for (col, &input_value) in input.iter().enumerate() {
                let group = col / data.group_size;
                let packed_col = col / 8;
                let weight_shift = (col % 8) * 4;
                let weight_word = read_u32(data.qweight, packed_col * data.rows + row)?;
                let zero_word = read_u32(data.qzeros, group * packed_rows + packed_row)?;
                let quant = ((weight_word >> weight_shift) & 0x0f) as i32;
                let zero = ((((zero_word >> zero_shift) & 0x0f) + 1) & 0x0f) as i32;
                let scale = read_float(data.scales, data.scale_dtype, group * data.rows + row)?;
                sum += input_value * (quant - zero) as f32 * scale;
            }
            output[row] = sum;
        }
        Ok(output)
    }

    #[cfg(feature = "cuda")]
    fn read_u32(bytes: &[u8], index: usize) -> Result<u32> {
        let offset = index
            .checked_mul(4)
            .ok_or_else(|| XrtError::InvalidTensor("packed u32 offset overflow".to_string()))?;
        let value = bytes.get(offset..offset + 4).ok_or_else(|| {
            XrtError::InvalidTensor(format!("packed u32 index {index} is out of bounds"))
        })?;
        Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
    }

    #[cfg(feature = "cuda")]
    fn read_i32(bytes: &[u8], index: usize) -> Result<i32> {
        read_u32(bytes, index).map(|value| i32::from_le_bytes(value.to_le_bytes()))
    }

    #[cfg(feature = "cuda")]
    fn read_float(bytes: &[u8], dtype: DType, index: usize) -> Result<f32> {
        match dtype {
            DType::F32 => {
                let offset = index.checked_mul(4).ok_or_else(|| {
                    XrtError::InvalidTensor("F32 scale offset overflow".to_string())
                })?;
                let value = bytes.get(offset..offset + 4).ok_or_else(|| {
                    XrtError::InvalidTensor(format!("F32 scale index {index} is out of bounds"))
                })?;
                Ok(f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            }
            DType::F16 => {
                let offset = index.checked_mul(2).ok_or_else(|| {
                    XrtError::InvalidTensor("F16 scale offset overflow".to_string())
                })?;
                xrt_core::decode_f16(bytes.get(offset..offset + 2).ok_or_else(|| {
                    XrtError::InvalidTensor(format!("F16 scale index {index} is out of bounds"))
                })?)
            }
            DType::BF16 => {
                let offset = index.checked_mul(2).ok_or_else(|| {
                    XrtError::InvalidTensor("BF16 scale offset overflow".to_string())
                })?;
                xrt_core::decode_bf16(bytes.get(offset..offset + 2).ok_or_else(|| {
                    XrtError::InvalidTensor(format!("BF16 scale index {index} is out of bounds"))
                })?)
            }
            _ => Err(XrtError::Unsupported(format!(
                "host GPTQ scale dtype {dtype:?} is unsupported"
            ))),
        }
    }

    #[test]
    #[ignore = "requires XRT_REAL_HF_MODEL_DIR with the VibeThinker Qwen2 SafeTensors bundle"]
    fn real_hf_qwen2_source_maps_every_dense_tensor() -> Result<()> {
        let root = env::var("XRT_REAL_HF_MODEL_DIR")
            .map_err(|_| XrtError::Runtime("XRT_REAL_HF_MODEL_DIR is required".to_string()))?;
        let bundle = HfModelBundle::open(root)?;
        let source = HfQwen2ResidentTensorSource::new(&bundle)?;

        assert_eq!(source.tensor_infos().len(), bundle.tensor_count());
        assert!(source.tensor_info("output.weight").is_none());
        let embedding = source.require_tensor("token_embd.weight")?;
        assert_eq!(embedding.dtype, DType::BF16);
        assert_eq!(embedding.rows, 151936);
        assert_eq!(embedding.cols, 2048);
        let q = source.require_tensor("blk.0.attn_q.weight")?;
        assert_eq!(q.dtype, DType::BF16);
        assert_eq!(q.rows, 2048);
        assert_eq!(q.cols, 2048);
        let k_bias = source.require_tensor("blk.0.attn_k.bias")?;
        assert_eq!(k_bias.numel, 256);

        Ok(())
    }
}
