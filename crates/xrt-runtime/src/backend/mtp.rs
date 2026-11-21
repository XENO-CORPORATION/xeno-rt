//! Official Qwen NextN/MTP companion model loading.
//!
//! Recent Qwen GGUF releases publish the target and one-layer NextN predictor
//! as separate artifacts. XRT historically admitted only artifacts with the
//! predictor tensors appended to the target GGUF. This module overlays the
//! companion's unique tensors during CUDA upload while preserving the target
//! model and its public API contract.

use super::*;
use std::{path::PathBuf, sync::Arc};

const QWEN_MTP_DRAFT_MODEL_ENV: &str = "XRT_QWEN_MTP_DRAFT_MODEL";

pub(super) struct QwenMtpDraftPlan {
    pub(super) path: PathBuf,
    pub(super) gguf: Arc<GgufFile>,
    pub(super) config: LlamaConfig,
}

impl QwenMtpDraftPlan {
    pub(super) fn from_env(target: &LlamaConfig) -> Result<Option<Self>> {
        let Some(path) = env::var_os(QWEN_MTP_DRAFT_MODEL_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        else {
            return Ok(None);
        };
        if target.has_nextn_predictor() {
            return Err(XrtError::Unsupported(format!(
                "{QWEN_MTP_DRAFT_MODEL_ENV} cannot be combined with a target GGUF that already contains an appended NextN predictor"
            )));
        }
        if !target.is_qwen35_family() || !target.is_hybrid() || target.is_moe() {
            return Err(XrtError::Unsupported(format!(
                "{QWEN_MTP_DRAFT_MODEL_ENV} currently requires a dense hybrid Qwen3.5-compatible target"
            )));
        }

        let gguf = Arc::new(GgufFile::open(&path).map_err(|error| {
            XrtError::Runtime(format!(
                "failed to open Qwen MTP draft GGUF `{}`: {error}",
                path.display()
            ))
        })?);
        let config = LlamaConfig::from_gguf(&gguf)?;
        validate_companion_config(target, &config)?;
        let source = GgufResidentTensorSource::new(&gguf);
        if !ResidentQwen35MtpWeights::supports(&source, &config) {
            return Err(XrtError::Unsupported(format!(
                "Qwen MTP draft GGUF `{}` does not match the admitted one-layer NextN tensor contract",
                path.display()
            )));
        }

        Ok(Some(Self { path, gguf, config }))
    }
}

fn validate_companion_config(target: &LlamaConfig, companion: &LlamaConfig) -> Result<()> {
    if !companion.is_qwen35_family()
        || !companion.is_hybrid()
        || companion.is_moe()
        || companion.nextn_predict_layers != 1
        || companion.total_block_count != companion.block_count + 1
    {
        return Err(XrtError::Unsupported(
            "Qwen MTP companion must describe one NextN layer appended to a dense hybrid Qwen3.5-compatible trunk"
                .to_string(),
        ));
    }

    let compatible = target.vocab_size == companion.vocab_size
        && target.context_length == companion.context_length
        && target.embedding_length == companion.embedding_length
        && target.feed_forward_length == companion.feed_forward_length
        && target.block_count == companion.block_count
        && target.attention_head_count == companion.attention_head_count
        && target.attention_head_count_kv == companion.attention_head_count_kv
        && target.rope_dimension_count == companion.rope_dimension_count
        && target.head_dim() == companion.head_dim()
        && target.ssm_conv_kernel == companion.ssm_conv_kernel
        && target.ssm_state_size == companion.ssm_state_size
        && target.ssm_group_count == companion.ssm_group_count
        && target.ssm_inner_size == companion.ssm_inner_size
        && target.ssm_dt_rank == companion.ssm_dt_rank
        && target.rope_freq_base.to_bits() == companion.rope_freq_base.to_bits()
        && target.rope_freq_scale.to_bits() == companion.rope_freq_scale.to_bits();
    if !compatible {
        return Err(XrtError::InvalidMetadata(format!(
            "Qwen MTP companion geometry does not match the target (target blocks={}, dim={}, vocab={}; companion blocks={}, dim={}, vocab={})",
            target.block_count,
            target.embedding_length,
            target.vocab_size,
            companion.block_count,
            companion.embedding_length,
            companion.vocab_size
        )));
    }
    Ok(())
}

/// A target-first tensor view. Duplicate embeddings/output tensors published
/// in the companion never replace the target weights or count twice in CUDA
/// admission; only companion tensors absent from the target are exposed.
pub(super) struct QwenMtpTensorSource<'a> {
    target: &'a dyn ResidentTensorSource,
    companion: GgufResidentTensorSource<'a>,
}

impl<'a> QwenMtpTensorSource<'a> {
    pub(super) fn new(target: &'a dyn ResidentTensorSource, companion: &'a GgufFile) -> Self {
        Self {
            target,
            companion: GgufResidentTensorSource::new(companion),
        }
    }
}

impl ResidentTensorSource for QwenMtpTensorSource<'_> {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo> {
        self.target
            .tensor_info(name)
            .or_else(|| self.companion.tensor_info(name))
    }

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        if self.target.tensor_info(name).is_some() {
            self.target.tensor_data(name)
        } else {
            self.companion.tensor_data(name)
        }
    }

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo> {
        let mut infos = self.target.tensor_infos();
        let mut names = infos
            .iter()
            .map(|info| info.name.clone())
            .collect::<BTreeSet<_>>();
        infos.extend(
            self.companion
                .tensor_infos()
                .into_iter()
                .filter(|info| names.insert(info.name.clone())),
        );
        infos
    }
}
