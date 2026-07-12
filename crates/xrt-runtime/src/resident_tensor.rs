use std::collections::{BTreeMap, BTreeSet};
use xrt_core::{DType, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_safetensors::{HfModelBundle, SafeTensorDType, SafeTensorInfo};

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
        }
    }
}

pub(crate) trait ResidentTensorSource: Send + Sync {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo>;

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]>;

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo>;

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

pub(crate) struct HfQwen2ResidentTensorSource<'a> {
    bundle: &'a HfModelBundle,
    infos: BTreeMap<String, ResidentTensorInfo>,
    actual_names: BTreeMap<String, String>,
}

impl<'a> HfQwen2ResidentTensorSource<'a> {
    pub(crate) fn new(bundle: &'a HfModelBundle) -> Result<Self> {
        if !bundle.config().model_type.eq_ignore_ascii_case("qwen2") {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors resident source currently supports Qwen2, found `{}`",
                bundle.config().model_type
            )));
        }
        if bundle.config().quantization.is_some() {
            return Err(XrtError::Unsupported(
                "SafeTensors AWQ, GPTQ, and compressed-tensors layouts require dedicated resident matrix types and kernels"
                    .to_string(),
            ));
        }

        let mut actual_names = BTreeMap::new();
        add_required_mapping(
            bundle,
            &mut actual_names,
            "token_embd.weight",
            "model.embed_tokens.weight",
        )?;
        add_required_mapping(
            bundle,
            &mut actual_names,
            "output_norm.weight",
            "model.norm.weight",
        )?;
        if bundle.tensor_info("lm_head.weight").is_some() {
            add_required_mapping(bundle, &mut actual_names, "output.weight", "lm_head.weight")?;
        } else if !bundle.config().tie_word_embeddings {
            return Err(XrtError::InvalidTensor(
                "untied Qwen2 SafeTensors model is missing `lm_head.weight`".to_string(),
            ));
        }

        for layer in 0..bundle.config().num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for (canonical_suffix, hf_suffix) in [
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
                ("attn_q.weight", "self_attn.q_proj.weight"),
                ("attn_k.weight", "self_attn.k_proj.weight"),
                ("attn_v.weight", "self_attn.v_proj.weight"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
                ("ffn_gate.weight", "mlp.gate_proj.weight"),
                ("ffn_up.weight", "mlp.up_proj.weight"),
                ("ffn_down.weight", "mlp.down_proj.weight"),
            ] {
                add_required_mapping(
                    bundle,
                    &mut actual_names,
                    &format!("blk.{layer}.{canonical_suffix}"),
                    &format!("{prefix}.{hf_suffix}"),
                )?;
            }
            for (canonical_suffix, hf_suffix) in [
                ("attn_q.bias", "self_attn.q_proj.bias"),
                ("attn_k.bias", "self_attn.k_proj.bias"),
                ("attn_v.bias", "self_attn.v_proj.bias"),
            ] {
                add_optional_mapping(
                    bundle,
                    &mut actual_names,
                    &format!("blk.{layer}.{canonical_suffix}"),
                    &format!("{prefix}.{hf_suffix}"),
                )?;
            }
        }

        let mapped_actual_names = actual_names.values().cloned().collect::<BTreeSet<_>>();
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

        let infos = actual_names
            .iter()
            .map(|(canonical, actual)| {
                let info = bundle.tensor_info(actual).ok_or_else(|| {
                    XrtError::InvalidTensor(format!(
                        "mapped SafeTensors tensor `{actual}` disappeared"
                    ))
                })?;
                Ok((canonical.clone(), normalize_hf_tensor(canonical, info)?))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self {
            bundle,
            infos,
            actual_names,
        })
    }
}

impl ResidentTensorSource for HfQwen2ResidentTensorSource<'_> {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo> {
        self.infos.get(name).cloned()
    }

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        let actual = self
            .actual_names
            .get(name)
            .ok_or_else(|| XrtError::InvalidTensor(format!("missing canonical tensor `{name}`")))?;
        Ok(self.bundle.require_tensor(actual)?.data)
    }

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo> {
        self.infos.values().cloned().collect()
    }
}

fn add_required_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, String>,
    canonical: &str,
    actual: &str,
) -> Result<()> {
    if bundle.tensor_info(actual).is_none() {
        return Err(XrtError::InvalidTensor(format!(
            "Qwen2 SafeTensors model is missing required tensor `{actual}` for `{canonical}`"
        )));
    }
    insert_mapping(mappings, canonical, actual)
}

fn add_optional_mapping(
    bundle: &HfModelBundle,
    mappings: &mut BTreeMap<String, String>,
    canonical: &str,
    actual: &str,
) -> Result<()> {
    if bundle.tensor_info(actual).is_some() {
        insert_mapping(mappings, canonical, actual)?;
    }
    Ok(())
}

fn insert_mapping(
    mappings: &mut BTreeMap<String, String>,
    canonical: &str,
    actual: &str,
) -> Result<()> {
    if mappings
        .insert(canonical.to_string(), actual.to_string())
        .is_some()
    {
        return Err(XrtError::InvalidTensor(format!(
            "duplicate canonical tensor mapping `{canonical}`"
        )));
    }
    Ok(())
}

fn normalize_hf_tensor(canonical: &str, info: &SafeTensorInfo) -> Result<ResidentTensorInfo> {
    let dtype = match &info.dtype {
        SafeTensorDType::F32 => DType::F32,
        SafeTensorDType::F16 => DType::F16,
        SafeTensorDType::Bf16 => DType::BF16,
        dtype => {
            return Err(XrtError::Unsupported(format!(
                "dense SafeTensors tensor `{}` for `{canonical}` has unsupported dtype {dtype:?}; expected F32, F16, or BF16",
                info.name
            )));
        }
    };
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
    let expected_bytes = numel.checked_mul(element_bytes).ok_or_else(|| {
        XrtError::InvalidTensor(format!(
            "SafeTensors tensor `{}` byte length overflows",
            info.name
        ))
    })?;
    if info.byte_len != expected_bytes {
        return Err(XrtError::InvalidTensor(format!(
            "SafeTensors tensor `{}` has {} bytes, expected {expected_bytes} for {:?} shape {:?}",
            info.name, info.byte_len, info.dtype, info.shape
        )));
    }

    Ok(ResidentTensorInfo {
        name: canonical.to_string(),
        dimensions: info.shape.clone(),
        dtype,
        rank: info.shape.len(),
        rows,
        cols,
        numel,
        byte_len: info.byte_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

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
