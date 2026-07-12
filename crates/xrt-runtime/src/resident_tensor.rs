use std::collections::{BTreeMap, BTreeSet};
use xrt_core::{DType, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_safetensors::{HfModelBundle, HfQuantizationMethod, SafeTensorDType, SafeTensorInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentTensorStorage {
    Dense,
    AwqGemm4 { group_size: usize },
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

pub(crate) trait ResidentTensorSource: Send + Sync {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo>;

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]>;

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo>;

    fn awq_gemm4_data<'a>(&'a self, _name: &str) -> Result<Option<ResidentAwqGemm4Data<'a>>> {
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
        }
    }
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
        let awq_group_size = supported_awq_group_size(bundle)?;

        let mut mappings = BTreeMap::new();
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
                awq_group_size,
            )?;
        } else if !bundle.config().tie_word_embeddings {
            return Err(XrtError::InvalidTensor(
                "untied Qwen2 SafeTensors model is missing `lm_head.weight` or an AWQ `lm_head` tensor group"
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
                    awq_group_size,
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
        }

        let mapped_actual_names = mappings
            .values()
            .flat_map(HfTensorMapping::actual_names)
            .map(ToOwned::to_owned)
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
            HfTensorMapping::AwqGemm4 { .. } => Err(XrtError::InvalidTensor(format!(
                "canonical tensor `{name}` uses grouped AWQ storage, not a single dense payload"
            ))),
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
}

fn supported_awq_group_size(bundle: &HfModelBundle) -> Result<Option<i64>> {
    let Some(quantization) = bundle.config().quantization.as_ref() else {
        return Ok(None);
    };
    if quantization.method != HfQuantizationMethod::Awq {
        return Err(XrtError::Unsupported(format!(
            "SafeTensors Qwen2 CUDA decode supports dense tensors or AutoAWQ GEMM, found {:?}",
            quantization.method
        )));
    }
    if quantization.bits != Some(4) {
        return Err(XrtError::Unsupported(format!(
            "AutoAWQ GEMM requires explicit 4-bit weights, found bits={:?}",
            quantization.bits
        )));
    }
    let group_size = quantization.group_size.ok_or_else(|| {
        XrtError::InvalidMetadata(
            "AutoAWQ GEMM requires an explicit group_size/q_group_size".to_string(),
        )
    })?;
    if !matches!(group_size, -1 | 32 | 64 | 128) {
        return Err(XrtError::Unsupported(format!(
            "AutoAWQ GEMM group size {group_size} is unsupported; expected -1, 32, 64, or 128"
        )));
    }
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
    Ok(Some(group_size))
}

fn has_hf_linear(bundle: &HfModelBundle, base: &str) -> bool {
    ["weight", "qweight", "qzeros", "scales"]
        .into_iter()
        .any(|suffix| bundle.tensor_info(&format!("{base}.{suffix}")).is_some())
}

fn add_required_linear_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, HfTensorMapping>,
    canonical: &str,
    base: &str,
    awq_group_size: Option<i64>,
) -> Result<()> {
    let weight = format!("{base}.weight");
    let qweight = format!("{base}.qweight");
    let qzeros = format!("{base}.qzeros");
    let scales = format!("{base}.scales");
    let has_weight = bundle.tensor_info(&weight).is_some();
    let component_presence = [
        bundle.tensor_info(&qweight).is_some(),
        bundle.tensor_info(&qzeros).is_some(),
        bundle.tensor_info(&scales).is_some(),
    ];
    if has_weight && component_presence.iter().any(|present| *present) {
        return Err(XrtError::InvalidTensor(format!(
            "SafeTensors linear `{base}` mixes dense `.weight` and AWQ components"
        )));
    }
    if has_weight {
        return insert_mapping(mappings, canonical, HfTensorMapping::Dense(weight));
    }

    let configured_group_size = awq_group_size.ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "Qwen2 SafeTensors model is missing dense tensor `{weight}` for `{canonical}`"
        ))
    })?;
    if component_presence.iter().any(|present| !*present) {
        return Err(XrtError::InvalidTensor(format!(
            "AutoAWQ linear `{base}` requires `.qweight`, `.qzeros`, and `.scales`"
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
    let group_size = if configured_group_size == -1 {
        cols
    } else {
        usize::try_from(configured_group_size).map_err(|_| {
            XrtError::InvalidMetadata(format!(
                "AutoAWQ group size {configured_group_size} exceeds usize"
            ))
        })?
    };
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
    fn synthetic_quantized_source_rejects_gptq_without_reinterpreting_it_as_awq() {
        let directory = tempfile::tempdir().unwrap();
        write_synthetic_awq_bundle(
            directory.path(),
            r#"{
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": true,
                "format": "gptq"
            }"#,
            false,
        );
        let bundle = HfModelBundle::open(directory.path()).unwrap();
        let error = HfQwen2ResidentTensorSource::new(&bundle)
            .err()
            .expect("GPTQ must remain explicitly unsupported");
        assert!(error.to_string().contains("Gptq"), "{error}");
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
