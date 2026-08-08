use parking_lot::RwLock;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::info;
use xrt_core::{decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_kernels::cpu::{
    accumulate_scaled, add_inplace, apply_rmsnorm, delta_rule_group_out_of_place,
    dequantize_mxfp4_row, dequantize_q4_0_row, dequantize_q4_k_row, dequantize_q5_k_row,
    dequantize_q6_k_row, dequantize_q8_0_row, dot, gated_rmsnorm, geglu_pytorch_tanh,
    global_expert_pool, global_pool, l2_normalize, matvec_quantized, matvec_quantized_batch,
    matvec_quantized_fused, matvec_quantized_fused_mixed, matvec_quantized_independent,
    quantized_row_dot, silu_inplace_fast, swiglu, RopeFreqs,
};
use xrt_safetensors::{HfModelConfig, HfQuantizationMethod};

use crate::{
    hybrid_state::{DeltaNetState, DeltaNetStateDescriptor},
    moe::{
        group_route_slot_by_expert, route_top_k, MoeCpuExecution, MoeLayerDescriptor,
        MoeRoutingRow, MoeTelemetry, MoeTelemetrySnapshot,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchitectureFamily {
    Llama,
    Qwen2,
    Qwen3,
    Qwen35Like,
    Gemma4,
}

#[derive(Debug, Clone, Copy)]
struct ArchitectureDescriptor {
    family: ArchitectureFamily,
    metadata_prefixes: &'static [&'static str],
}

fn describe_architecture(architecture: &str) -> Result<ArchitectureDescriptor> {
    match architecture {
        "llama" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Llama,
            metadata_prefixes: &["llama"],
        }),
        "qwen2" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen2,
            metadata_prefixes: &["qwen2", "qwen2_5", "qwen2.5"],
        }),
        "qwen2_5" | "qwen2.5" | "qwen2_5_coder" | "qwen2.5-coder" => {
            Ok(ArchitectureDescriptor {
                family: ArchitectureFamily::Qwen2,
                metadata_prefixes: &["qwen2_5", "qwen2.5", "qwen2"],
            })
        }
        "qwen2_5_vl" | "qwen2_5_vl_text" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen2,
            metadata_prefixes: &["qwen2_5_vl", "qwen2_5_vl_text", "qwen2_5", "qwen2"],
        }),
        "qwen3" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen3,
            metadata_prefixes: &["qwen3"],
        }),
        "qwen3moe" | "qwen3_moe" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen3,
            metadata_prefixes: &["qwen3moe", "qwen3_moe", "qwen3"],
        }),
        "qwen35" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen35", "qwen3_5", "qwen3_5_moe", "qwen3_next"],
        }),
        "qwen3_5" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen3_5", "qwen35", "qwen3_5_moe", "qwen3_next"],
        }),
        "qwen3_5_moe" | "qwen35moe" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &[
                "qwen3_5_moe",
                "qwen35moe",
                "qwen3_5",
                "qwen35",
                "qwen3_next",
            ],
        }),
        "qwen3_next" | "qwen3next" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen3_next", "qwen35", "qwen3_5", "qwen3_5_moe"],
        }),
        "qwen3_omni_moe" | "qwen2_5_omni" => Err(XrtError::Unsupported(format!(
            "unsupported architecture {architecture}: this is a composite omni model with native thinker/vision/audio modules; xeno-rt currently supports text backbones plus mmproj-style vision only"
        ))),
        "glm46v" | "glm4v" | "glm4v_moe" | "glm_4v" => Err(XrtError::Unsupported(format!(
            "unsupported architecture {architecture}: GLM vision-language models require a native multimodal stack that xeno-rt has not implemented yet"
        ))),
        other if other.starts_with("glm") => Err(XrtError::Unsupported(format!(
            "unsupported architecture {other}: GLM model support has not been implemented yet"
        ))),
        "gemma4" | "gemma_4" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Gemma4,
            metadata_prefixes: &["gemma4", "gemma_4"],
        }),
        _ => Err(XrtError::Unsupported(format!(
            "xrt-models supports llama, qwen2/qwen2.5, qwen3/qwen3moe, qwen35/qwen3.5/qwen35moe, qwen3-next, and gemma4 architectures, found {architecture}"
        ))),
    }
}

fn metadata_usize_any(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Option<usize> {
    prefixes
        .iter()
        .find_map(|prefix| gguf.metadata_usize(&format!("{prefix}.{suffix}")))
}

fn metadata_f32_any(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Option<f32> {
    prefixes
        .iter()
        .find_map(|prefix| gguf.metadata_f32(&format!("{prefix}.{suffix}")))
}

fn metadata_usize_array_any(
    gguf: &GgufFile,
    prefixes: &[&str],
    suffix: &str,
) -> Option<Vec<usize>> {
    prefixes.iter().find_map(|prefix| {
        gguf.metadata_array(&format!("{prefix}.{suffix}"))
            .and_then(|array| {
                array
                    .values
                    .iter()
                    .map(|value| value.to_usize())
                    .collect::<Option<Vec<_>>>()
            })
    })
}

fn metadata_bool_array_any(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Option<Vec<bool>> {
    prefixes.iter().find_map(|prefix| {
        gguf.metadata_array(&format!("{prefix}.{suffix}"))?
            .as_bool_vec()
    })
}

fn required_usize_any(gguf: &GgufFile, prefixes: &[&str], suffix: &str) -> Result<usize> {
    metadata_usize_any(gguf, prefixes, suffix).ok_or_else(|| {
        XrtError::InvalidMetadata(format!(
            "missing required metadata key: {}",
            prefixes
                .iter()
                .map(|prefix| format!("{prefix}.{suffix}"))
                .collect::<Vec<_>>()
                .join(" or ")
        ))
    })
}

/// Raw mutable pointer as usize for Send+Sync in parallel attention.
#[derive(Clone, Copy)]
struct SendPtr(usize);
impl SendPtr {
    fn new(ptr: *mut f32) -> Self {
        Self(ptr as usize)
    }
}
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[derive(Debug, Clone)]
struct Gemma4Config {
    layers: Vec<Gemma4LayerConfig>,
    max_q_width: usize,
    max_kv_width: usize,
    max_head_dim: usize,
    max_rope_dimension_count: usize,
    final_logit_softcapping: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Gemma4LayerConfig {
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    q_width: usize,
    kv_width: usize,
    rope_dimension_count: usize,
    rope_freq_base: f32,
    sliding_window: Option<usize>,
    has_kv: bool,
}

#[derive(Debug, Clone)]
pub struct Gemma4TraceStage {
    pub name: &'static str,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Gemma4LayerTrace {
    pub layer_index: usize,
    pub position: usize,
    pub stages: Vec<Gemma4TraceStage>,
}

impl Gemma4LayerTrace {
    pub fn new(layer_index: usize, position: usize) -> Self {
        Self {
            layer_index,
            position,
            stages: Vec::new(),
        }
    }

    pub fn record(&mut self, name: &'static str, values: &[f32]) {
        self.stages.push(Gemma4TraceStage {
            name,
            values: values.to_vec(),
        });
    }
}

impl Gemma4LayerConfig {
    pub fn head_count(&self) -> usize {
        self.head_count
    }

    pub fn kv_head_count(&self) -> usize {
        self.kv_head_count
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn q_width(&self) -> usize {
        self.q_width
    }

    pub fn kv_width(&self) -> usize {
        self.kv_width
    }

    pub fn rope_dimension_count(&self) -> usize {
        self.rope_dimension_count
    }

    pub fn rope_freq_base(&self) -> f32 {
        self.rope_freq_base
    }

    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub architecture: String,
    architecture_family: ArchitectureFamily,
    pub vocab_size: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    /// Decoder blocks in the target model trunk. Appended Qwen NextN/MTP
    /// predictor blocks are deliberately excluded from this count.
    pub block_count: usize,
    /// Decoder blocks physically described by the source artifact, including
    /// any appended Qwen NextN/MTP predictor blocks.
    pub total_block_count: usize,
    /// Appended Qwen NextN/MTP predictor blocks. Zero for ordinary models.
    pub nextn_predict_layers: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub rope_dimension_count: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub head_dim_override: Option<usize>,
    // MoE (Mixture of Experts) parameters (None for dense models)
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
    pub expert_shared_feed_forward_length: Option<usize>,
    // Qwen3.5 DeltaNet SSM parameters (None for standard transformer models)
    pub ssm_conv_kernel: Option<usize>,
    pub ssm_state_size: Option<usize>,
    pub ssm_group_count: Option<usize>,
    pub ssm_inner_size: Option<usize>,
    pub ssm_dt_rank: Option<usize>,
    gemma4: Option<Gemma4Config>,
    deltanet_state_descriptor: Option<DeltaNetStateDescriptor>,
}

fn qwen_vl_default_text_rope(value: &serde_json::Value, head_dim: usize) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "mrope_section" | "rope_type" | "type"))
    {
        return false;
    }
    for key in ["rope_type", "type"] {
        if object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind != "default")
        {
            return false;
        }
    }
    let Some(sections) = object
        .get("mrope_section")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    let Some(section_sum) = sections.iter().try_fold(0usize, |sum, section| {
        let section = usize::try_from(section.as_u64()?).ok()?;
        (section > 0).then(|| sum.checked_add(section)).flatten()
    }) else {
        return false;
    };
    section_sum
        .checked_mul(2)
        .is_some_and(|covered| covered == head_dim)
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .unwrap_or("llama")
            .to_string();
        let descriptor = describe_architecture(&architecture)?;
        let prefixes = descriptor.metadata_prefixes;
        let vocab_size = gguf
            .metadata_usize("vocab_size")
            .or_else(|| metadata_usize_any(gguf, prefixes, "vocab_size"))
            .or_else(|| {
                gguf.metadata_array("tokenizer.ggml.tokens")
                    .map(|array| array.len())
            })
            .ok_or_else(|| {
                XrtError::InvalidMetadata("missing llama vocab size metadata".to_string())
            })?;
        let context_length = required_usize_any(gguf, prefixes, "context_length")?;
        let embedding_length = required_usize_any(gguf, prefixes, "embedding_length")?;
        let expert_count = metadata_usize_any(gguf, prefixes, "expert_count");
        let expert_used_count = metadata_usize_any(gguf, prefixes, "expert_used_count");
        let dense_feed_forward_length = metadata_usize_any(gguf, prefixes, "feed_forward_length");
        let expert_feed_forward_length =
            metadata_usize_any(gguf, prefixes, "expert_feed_forward_length");
        let expert_shared_feed_forward_length =
            metadata_usize_any(gguf, prefixes, "expert_shared_feed_forward_length")
                .filter(|length| *length > 0);
        let feed_forward_length = if expert_count.is_some_and(|count| count > 1) {
            expert_feed_forward_length.or(dense_feed_forward_length)
        } else {
            dense_feed_forward_length.or(expert_feed_forward_length)
        }
        .ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "missing required metadata key: {}",
                prefixes
                    .iter()
                    .flat_map(|prefix| {
                        [
                            format!("{prefix}.feed_forward_length"),
                            format!("{prefix}.expert_feed_forward_length"),
                        ]
                    })
                    .collect::<Vec<_>>()
                    .join(" or ")
            ))
        })?;
        let total_block_count = required_usize_any(gguf, prefixes, "block_count")?;
        let nextn_predict_layers =
            metadata_usize_any(gguf, prefixes, "nextn_predict_layers").unwrap_or(0);
        if nextn_predict_layers > 0 && descriptor.family != ArchitectureFamily::Qwen35Like {
            return Err(XrtError::InvalidMetadata(format!(
                "nextn_predict_layers is only valid for Qwen3.5-compatible artifacts, found architecture `{architecture}`"
            )));
        }
        if nextn_predict_layers >= total_block_count {
            return Err(XrtError::InvalidMetadata(format!(
                "Qwen NextN predictor count {nextn_predict_layers} must be smaller than total block count {total_block_count}"
            )));
        }
        let block_count = total_block_count - nextn_predict_layers;
        let attention_head_count = required_usize_any(gguf, prefixes, "attention.head_count")?;
        let attention_head_count_kv = metadata_usize_any(gguf, prefixes, "attention.head_count_kv")
            .or_else(|| {
                metadata_usize_array_any(gguf, prefixes, "attention.head_count_kv")
                    .and_then(|values| values.into_iter().max())
            })
            .unwrap_or(attention_head_count);
        if attention_head_count == 0 || attention_head_count_kv == 0 {
            return Err(XrtError::InvalidMetadata(
                "attention head counts must be non-zero".to_string(),
            ));
        }
        if descriptor.family != ArchitectureFamily::Gemma4
            && embedding_length % attention_head_count != 0
        {
            return Err(XrtError::InvalidMetadata(format!(
                "embedding length {embedding_length} is not divisible by attention head count {attention_head_count}"
            )));
        }
        if attention_head_count % attention_head_count_kv != 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "attention head count {attention_head_count} is not divisible by KV head count {attention_head_count_kv}"
            )));
        }

        let default_head_dim = if descriptor.family == ArchitectureFamily::Gemma4 {
            metadata_usize_any(gguf, prefixes, "attention.key_length")
                .unwrap_or(embedding_length / attention_head_count)
        } else {
            embedding_length / attention_head_count
        };
        let head_dim_override = metadata_usize_any(gguf, prefixes, "attention.key_length")
            .filter(|&dim| dim != default_head_dim);
        let actual_head_dim = head_dim_override.unwrap_or(default_head_dim);
        let rope_dimension_count =
            metadata_usize_any(gguf, prefixes, "rope.dimension_count").unwrap_or(actual_head_dim);
        let rms_norm_eps = metadata_f32_any(gguf, prefixes, "attention.layer_norm_rms_epsilon")
            .or_else(|| metadata_f32_any(gguf, prefixes, "attention.layer_norm_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base = metadata_f32_any(gguf, prefixes, "rope.freq_base")
            .or_else(|| metadata_f32_any(gguf, prefixes, "rope.freq_base_train"))
            .unwrap_or(10000.0);
        let rope_freq_scale = metadata_f32_any(gguf, prefixes, "rope.scale_linear")
            .or_else(|| metadata_f32_any(gguf, prefixes, "rope.scaling.factor"))
            .unwrap_or(1.0);

        // MoE parameters (optional — only present for MoE models)
        match (expert_count, expert_used_count) {
            (Some(count), Some(selected)) if count > 1 && selected > 0 && selected <= count => {
                if selected > crate::moe::MAX_SELECTED_EXPERTS {
                    return Err(XrtError::Unsupported(format!(
                        "MoE selects {selected} experts per token, exceeding the fixed routing capacity of {}",
                        crate::moe::MAX_SELECTED_EXPERTS
                    )));
                }
            }
            (None, None) => {}
            (Some(count), Some(selected)) => {
                return Err(XrtError::InvalidMetadata(format!(
                    "invalid MoE expert geometry: expert_count={count}, expert_used_count={selected}"
                )));
            }
            (Some(_), None) => {
                return Err(XrtError::InvalidMetadata(
                    "MoE metadata defines expert_count but omits expert_used_count".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(XrtError::InvalidMetadata(
                    "MoE metadata defines expert_used_count but omits expert_count".to_string(),
                ));
            }
        }

        // Qwen3.5 SSM parameters (optional — only present for hybrid models)
        let ssm_conv_kernel = metadata_usize_any(gguf, prefixes, "ssm.conv_kernel");
        let ssm_state_size = metadata_usize_any(gguf, prefixes, "ssm.state_size");
        let ssm_group_count = metadata_usize_any(gguf, prefixes, "ssm.group_count");
        let ssm_inner_size = metadata_usize_any(gguf, prefixes, "ssm.inner_size");
        let ssm_dt_rank = metadata_usize_any(gguf, prefixes, "ssm.time_step_rank");
        let gemma4 = if descriptor.family == ArchitectureFamily::Gemma4 {
            Some(Self::load_gemma4_config(
                gguf,
                prefixes,
                block_count,
                attention_head_count,
                attention_head_count_kv,
                actual_head_dim,
                rope_dimension_count,
                rope_freq_base,
            )?)
        } else {
            None
        };

        let mut config = Self {
            architecture,
            architecture_family: descriptor.family,
            vocab_size,
            context_length,
            embedding_length,
            feed_forward_length,
            block_count,
            total_block_count,
            nextn_predict_layers,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count,
            rms_norm_eps,
            rope_freq_base,
            rope_freq_scale,
            head_dim_override,
            expert_count,
            expert_used_count,
            expert_shared_feed_forward_length,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_inner_size,
            ssm_dt_rank,
            gemma4,
            deltanet_state_descriptor: None,
        };
        config.deltanet_state_descriptor = DeltaNetStateDescriptor::from_config(&config)?;
        Ok(config)
    }

    pub fn from_hf(config: &HfModelConfig) -> Result<Self> {
        let model_type = config.model_type.trim().to_ascii_lowercase();
        let qwen_vl_text = matches!(model_type.as_str(), "qwen2_5_vl" | "qwen2_5_vl_text");
        if !matches!(
            model_type.as_str(),
            "qwen2" | "qwen3" | "qwen2_5_vl" | "qwen2_5_vl_text"
        ) {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors CUDA decode currently supports standard dense Qwen2, Qwen2.5-VL text, and Qwen3 models, found model_type `{}`",
                config.model_type
            )));
        }
        if let Some(quantization) = &config.quantization {
            if !matches!(
                &quantization.method,
                HfQuantizationMethod::Awq
                    | HfQuantizationMethod::Gptq
                    | HfQuantizationMethod::CompressedTensors
            ) {
                return Err(XrtError::Unsupported(format!(
                    "SafeTensors CUDA decode currently supports dense weights, AutoAWQ GEMM/GEMV, GPTQ v1/v2 GEMM4, or compressed-tensors W4A16, found {:?}",
                    quantization.method
                )));
            }
        }
        if !matches!(
            config.hidden_act.trim().to_ascii_lowercase().as_str(),
            "silu" | "swish"
        ) {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors standard-dense CUDA decode requires SiLU activation, found `{}`",
                config.hidden_act
            )));
        }
        if config.use_sliding_window {
            return Err(XrtError::Unsupported(
                "SafeTensors sliding-window attention is not wired into the standard dense CUDA path"
                    .to_string(),
            ));
        }
        if let Some(rope_scaling) = config
            .raw
            .get("rope_scaling")
            .filter(|value| !value.is_null())
        {
            if qwen_vl_text
                && qwen_vl_default_text_rope(
                    rope_scaling,
                    config
                        .head_dim
                        .unwrap_or(config.hidden_size / config.num_attention_heads),
                )
            {
                // Qwen2.5-VL uses three position axes. Generation prompts contain
                // text only, for which all three axes have the same position and
                // the audited default MRoPE reduces to ordinary 1D Qwen RoPE.
            } else {
                return Err(XrtError::Unsupported(
                    "SafeTensors rope_scaling variants are not wired into the CUDA path"
                        .to_string(),
                ));
            }
        }

        let descriptor = describe_architecture(&model_type)?;
        let default_head_dim = config.hidden_size / config.num_attention_heads;
        let actual_head_dim = config.head_dim.unwrap_or(default_head_dim);
        if actual_head_dim == 0 {
            return Err(XrtError::InvalidMetadata(
                "SafeTensors head dimension must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            architecture: model_type,
            architecture_family: descriptor.family,
            vocab_size: config.vocab_size,
            context_length: config.max_position_embeddings,
            embedding_length: config.hidden_size,
            feed_forward_length: config.intermediate_size,
            block_count: config.num_hidden_layers,
            total_block_count: config.num_hidden_layers,
            nextn_predict_layers: 0,
            attention_head_count: config.num_attention_heads,
            attention_head_count_kv: config.num_key_value_heads,
            rope_dimension_count: actual_head_dim,
            rms_norm_eps: config.rms_norm_eps,
            rope_freq_base: config.rope_theta,
            rope_freq_scale: 1.0,
            head_dim_override: (actual_head_dim != default_head_dim).then_some(actual_head_dim),
            expert_count: None,
            expert_used_count: None,
            expert_shared_feed_forward_length: None,
            ssm_conv_kernel: None,
            ssm_state_size: None,
            ssm_group_count: None,
            ssm_inner_size: None,
            ssm_dt_rank: None,
            gemma4: None,
            deltanet_state_descriptor: None,
        })
    }

    fn load_gemma4_config(
        gguf: &GgufFile,
        prefixes: &[&str],
        block_count: usize,
        default_head_count: usize,
        default_kv_head_count: usize,
        full_head_dim: usize,
        full_rope_dimension_count: usize,
        full_rope_freq_base: f32,
    ) -> Result<Gemma4Config> {
        let shared_kv_layers =
            metadata_usize_any(gguf, prefixes, "attention.shared_kv_layers").unwrap_or(0);
        if shared_kv_layers > 0 {
            return Err(XrtError::Unsupported(format!(
                "Gemma4 shared-KV layers are not supported yet: attention.shared_kv_layers={shared_kv_layers}"
            )));
        }

        let head_counts = expand_layer_usizes(
            metadata_usize_array_any(gguf, prefixes, "attention.head_count"),
            default_head_count,
            block_count,
            "gemma4.attention.head_count",
        )?;
        let kv_head_counts = expand_layer_usizes(
            metadata_usize_array_any(gguf, prefixes, "attention.head_count_kv"),
            default_kv_head_count,
            block_count,
            "gemma4.attention.head_count_kv",
        )?;
        let sliding_pattern = expand_layer_bools(
            metadata_bool_array_any(gguf, prefixes, "attention.sliding_window_pattern"),
            false,
            block_count,
            "gemma4.attention.sliding_window_pattern",
        )?;

        let sliding_window = metadata_usize_any(gguf, prefixes, "attention.sliding_window");
        let swa_head_dim =
            metadata_usize_any(gguf, prefixes, "attention.key_length_swa").unwrap_or(full_head_dim);
        let swa_value_dim = metadata_usize_any(gguf, prefixes, "attention.value_length_swa")
            .unwrap_or(swa_head_dim);
        let full_value_dim =
            metadata_usize_any(gguf, prefixes, "attention.value_length").unwrap_or(full_head_dim);
        if full_head_dim != full_value_dim {
            return Err(XrtError::Unsupported(format!(
                "Gemma4 with different full K/V head dims is not supported: key={full_head_dim}, value={full_value_dim}"
            )));
        }
        if swa_head_dim != swa_value_dim {
            return Err(XrtError::Unsupported(format!(
                "Gemma4 with different SWA K/V head dims is not supported: key={swa_head_dim}, value={swa_value_dim}"
            )));
        }

        let swa_rope_dimension_count =
            metadata_usize_any(gguf, prefixes, "rope.dimension_count_swa").unwrap_or(swa_head_dim);
        let swa_rope_freq_base =
            metadata_f32_any(gguf, prefixes, "rope.freq_base_swa").unwrap_or(full_rope_freq_base);

        let mut layers = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let is_swa = sliding_pattern[index];
            let head_count = head_counts[index];
            let kv_head_count = kv_head_counts[index];
            if head_count == 0 || kv_head_count == 0 {
                return Err(XrtError::InvalidMetadata(format!(
                    "Gemma4 layer {index} has invalid head counts: heads={head_count}, kv_heads={kv_head_count}"
                )));
            }
            if head_count % kv_head_count != 0 {
                return Err(XrtError::InvalidMetadata(format!(
                    "Gemma4 layer {index} head count {head_count} is not divisible by KV head count {kv_head_count}"
                )));
            }

            let head_dim = if is_swa { swa_head_dim } else { full_head_dim };
            let rope_dimension_count = if is_swa {
                swa_rope_dimension_count
            } else {
                full_rope_dimension_count
            };
            let rope_freq_base = if is_swa {
                swa_rope_freq_base
            } else {
                full_rope_freq_base
            };
            layers.push(Gemma4LayerConfig {
                head_count,
                kv_head_count,
                head_dim,
                q_width: head_count * head_dim,
                kv_width: kv_head_count * head_dim,
                rope_dimension_count,
                rope_freq_base,
                sliding_window: is_swa
                    .then_some(sliding_window.unwrap_or(0))
                    .filter(|&v| v > 0),
                has_kv: true,
            });
        }

        let max_q_width = layers.iter().map(|layer| layer.q_width).max().unwrap_or(0);
        let max_kv_width = layers.iter().map(|layer| layer.kv_width).max().unwrap_or(0);
        let max_head_dim = layers.iter().map(|layer| layer.head_dim).max().unwrap_or(0);
        let max_rope_dimension_count = layers
            .iter()
            .map(|layer| layer.rope_dimension_count)
            .max()
            .unwrap_or(0);
        let final_logit_softcapping = metadata_f32_any(gguf, prefixes, "final_logit_softcapping")
            .filter(|value| *value > 0.0);

        Ok(Gemma4Config {
            layers,
            max_q_width,
            max_kv_width,
            max_head_dim,
            max_rope_dimension_count,
            final_logit_softcapping,
        })
    }

    pub fn head_dim(&self) -> usize {
        if let Some(gemma4) = &self.gemma4 {
            return gemma4.max_head_dim;
        }
        self.head_dim_override
            .unwrap_or(self.embedding_length / self.attention_head_count)
    }

    pub fn q_width(&self) -> usize {
        if let Some(gemma4) = &self.gemma4 {
            return gemma4.max_q_width;
        }
        self.attention_head_count * self.head_dim()
    }

    pub fn kv_width(&self) -> usize {
        if let Some(gemma4) = &self.gemma4 {
            return gemma4.max_kv_width;
        }
        self.attention_head_count_kv * self.head_dim()
    }

    /// Whether this is a Mixture of Experts model.
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some_and(|n| n > 1)
    }

    /// Whether this is a hybrid model with DeltaNet (linear attention) layers.
    pub fn is_hybrid(&self) -> bool {
        self.ssm_conv_kernel.is_some()
    }

    pub fn deltanet_state_descriptor(&self) -> Option<&DeltaNetStateDescriptor> {
        self.deltanet_state_descriptor.as_ref()
    }

    pub fn is_qwen35_family(&self) -> bool {
        self.architecture_family == ArchitectureFamily::Qwen35Like
    }

    pub fn has_nextn_predictor(&self) -> bool {
        self.nextn_predict_layers > 0
    }

    pub fn nextn_layer_range(&self) -> std::ops::Range<usize> {
        self.block_count..self.total_block_count
    }

    pub fn is_gemma4(&self) -> bool {
        self.architecture_family == ArchitectureFamily::Gemma4
    }

    pub fn gemma4_layer_config(&self, layer: usize) -> Option<&Gemma4LayerConfig> {
        self.gemma4.as_ref()?.layers.get(layer)
    }

    pub fn gemma4_layer_kv_widths(&self) -> Option<Vec<usize>> {
        self.gemma4
            .as_ref()
            .map(|config| config.layers.iter().map(|layer| layer.kv_width).collect())
    }

    pub fn gemma4_final_logit_softcapping(&self) -> Option<f32> {
        self.gemma4.as_ref()?.final_logit_softcapping
    }

    /// For hybrid models, returns true if the given layer uses DeltaNet (recurrent)
    /// rather than full attention. Pattern: every 4th layer (3, 7, 11, ...) is full attention.
    pub fn is_recurrent(&self, layer: usize) -> bool {
        self.is_hybrid() && (layer % 4 != 3)
    }
}

/// Pre-resolved tensor metadata to avoid HashMap lookups during forward pass.
/// Each forward_token call does 7 linear projections × 28 layers = 196 calls,
/// each requiring 2 HashMap lookups (require_tensor + tensor_data). Pre-resolving
/// eliminates ~400 string hash+compare operations per token.
#[derive(Debug, Clone)]
struct ResolvedWeight {
    /// Byte offset of this tensor's data within the GGUF data section.
    data_offset: usize,
    /// Total byte size of this tensor's data.
    nbytes: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
    /// Original tensor name for LoRA adapter lookup.
    name: String,
}

/// Attention mechanism weights — varies by layer type.
#[derive(Debug, Clone)]
enum AttnWeights {
    /// Standard transformer attention (llama, qwen3).
    Standard {
        attn_q: ResolvedWeight,
        attn_k: ResolvedWeight,
        attn_v: ResolvedWeight,
        attn_output: ResolvedWeight,
        attn_q_norm: Option<String>,
        attn_k_norm: Option<String>,
        attn_q_bias: Option<String>,
        attn_k_bias: Option<String>,
        attn_v_bias: Option<String>,
    },
    /// Qwen3.5 full attention (Q+gate interleaved, GQA).
    Qwen35Attn {
        /// Q projection includes interleaved gate: [n_heads * (head_dim + head_dim)].
        attn_qg: ResolvedWeight,
        attn_k: ResolvedWeight,
        attn_v: ResolvedWeight,
        attn_output: ResolvedWeight,
        attn_q_norm: String,
        attn_k_norm: String,
    },
    /// Gemma4 dense attention with per-layer local/global widths and optional V projection.
    Gemma4 {
        attn_q: ResolvedWeight,
        attn_k: ResolvedWeight,
        attn_v: Option<ResolvedWeight>,
        attn_output: ResolvedWeight,
        attn_q_norm: String,
        attn_k_norm: String,
        attn_post_norm: String,
    },
    /// Qwen3.5 DeltaNet (linear attention with recurrent state).
    DeltaNet {
        attn_qkv: ResolvedWeight,
        attn_gate: ResolvedWeight,
        ssm_alpha: ResolvedWeight,
        ssm_beta: ResolvedWeight,
        ssm_a: String,
        ssm_dt_bias: String,
        ssm_norm: String,
        ssm_out: ResolvedWeight,
    },
}

#[derive(Debug, Clone)]
enum FfnWeights {
    /// Standard dense FFN (SwiGLU: gate + up → swiglu → down).
    Dense {
        gate: ResolvedWeight,
        down: ResolvedWeight,
        up: ResolvedWeight,
    },
    /// Gemma4 dense FFN (GELU-gated), followed by a post-FFN RMSNorm and optional layer scale.
    Gemma4Dense {
        gate: ResolvedWeight,
        down: ResolvedWeight,
        up: ResolvedWeight,
        post_ffw_norm: String,
        layer_output_scale: Option<String>,
    },
    /// Mixture of Experts: router selects top-K experts, each with own gate/up/down.
    Moe {
        descriptor: MoeLayerDescriptor,
        router: ResolvedWeight,
        experts: Vec<MoeExpertWeights>,
        shared: Option<MoeSharedExpertWeights>,
    },
}

#[derive(Debug, Clone)]
struct MoeExpertWeights {
    gate: ResolvedWeight,
    down: ResolvedWeight,
    up: ResolvedWeight,
}

#[derive(Debug, Clone)]
struct MoeSharedExpertWeights {
    gate_selector: ResolvedWeight,
    gate: ResolvedWeight,
    down: ResolvedWeight,
    up: ResolvedWeight,
    intermediate_size: usize,
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm: String,
    attn: AttnWeights,
    ffn_norm: String,
    ffn: FfnWeights,
}

/// Reusable scratch buffers to avoid per-token heap allocations in the forward pass.
struct ForwardScratch {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    down: Vec<f32>,
    /// Reusable RoPE sin/cos cache (avoids allocation per layer per position)
    sin_cache: Vec<f32>,
    cos_cache: Vec<f32>,
    // MoE scratch buffers (only non-empty for MoE models)
    moe_router_logits: Vec<f32>,
    moe_expert_out: Vec<f32>,
    // DeltaNet scratch buffers (only non-empty for hybrid models)
    dn_qkv: Vec<f32>,
    dn_gate: Vec<f32>,
    dn_alpha: Vec<f32>,
    dn_beta: Vec<f32>,
    dn_conv_out: Vec<f32>,
    dn_out: Vec<f32>,
    // Qwen3.5 full attention: Q+gate interleaved buffer
    q35_qg: Vec<f32>,
}

impl ForwardScratch {
    fn new(config: &LlamaConfig) -> Self {
        let rope_dim = config
            .gemma4
            .as_ref()
            .map(|gemma4| gemma4.max_rope_dimension_count / 2)
            .unwrap_or(config.rope_dimension_count / 2);
        let inner = config.ssm_inner_size.unwrap_or(0);
        let groups = config.ssm_group_count.unwrap_or(0);
        let state_size = config.ssm_state_size.unwrap_or(0);
        let dt_rank = config.ssm_dt_rank.unwrap_or(0);
        let conv_channels = if config.is_hybrid() {
            state_size * groups * 2 + inner
        } else {
            0
        };
        // For qwen35 full attention: Q+gate interleaved = head_count * head_dim * 2
        let qg_size = if config.is_hybrid() {
            config.attention_head_count * config.head_dim() * 2
        } else {
            0
        };
        let n_experts = config.expert_count.unwrap_or(0);
        let moe_intermediate = config
            .expert_shared_feed_forward_length
            .unwrap_or(0)
            .max(config.feed_forward_length);
        Self {
            normed: vec![0.0; config.embedding_length],
            q: vec![0.0; config.q_width().max(qg_size / 2)],
            k: vec![0.0; config.kv_width()],
            v: vec![0.0; config.kv_width()],
            gate: vec![0.0; moe_intermediate],
            up: vec![0.0; moe_intermediate],
            attn_out: vec![0.0; config.q_width()],
            proj: vec![0.0; config.embedding_length],
            down: vec![0.0; config.embedding_length],
            sin_cache: vec![0.0; rope_dim],
            cos_cache: vec![0.0; rope_dim],
            moe_router_logits: vec![0.0; n_experts],
            moe_expert_out: vec![0.0; config.embedding_length],
            dn_qkv: vec![0.0; conv_channels],
            dn_gate: vec![0.0; inner],
            dn_alpha: vec![0.0; dt_rank],
            dn_beta: vec![0.0; dt_rank],
            dn_conv_out: vec![0.0; conv_channels],
            dn_out: vec![0.0; inner],
            q35_qg: vec![0.0; qg_size],
        }
    }
}

/// Reusable scratch buffers for batch forward pass (prefill).
/// Inspired by XenoMind's FieldPool pattern: allocate once, reuse across calls.
struct BatchScratch {
    xs: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    down: Vec<f32>,
    moe_router_logits: Vec<f32>,
    moe_routes: Vec<MoeRoutingRow>,
    moe_expert_counts: Vec<usize>,
    moe_expert_offsets: Vec<usize>,
    moe_expert_cursors: Vec<usize>,
    moe_token_indices: Vec<usize>,
    moe_inputs: Vec<f32>,
    moe_shared_gate: Vec<f32>,
    /// The seq_len these buffers were sized for
    capacity: usize,
}

impl BatchScratch {
    fn new() -> Self {
        Self {
            xs: Vec::new(),
            normed: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            attn_out: Vec::new(),
            proj: Vec::new(),
            down: Vec::new(),
            moe_router_logits: Vec::new(),
            moe_routes: Vec::new(),
            moe_expert_counts: Vec::new(),
            moe_expert_offsets: Vec::new(),
            moe_expert_cursors: Vec::new(),
            moe_token_indices: Vec::new(),
            moe_inputs: Vec::new(),
            moe_shared_gate: Vec::new(),
            capacity: 0,
        }
    }

    /// Ensure all buffers can hold `seq_len` tokens for the given config.
    /// Only reallocates if the current capacity is insufficient.
    fn ensure_capacity(&mut self, seq_len: usize, config: &LlamaConfig) {
        if seq_len <= self.capacity {
            // Just zero the portions we'll use
            let dim = config.embedding_length;
            self.xs[..seq_len * dim].fill(0.0);
            self.normed[..seq_len * dim].fill(0.0);
            self.q[..seq_len * config.q_width()].fill(0.0);
            self.k[..seq_len * config.kv_width()].fill(0.0);
            self.v[..seq_len * config.kv_width()].fill(0.0);
            let moe_intermediate = config
                .expert_shared_feed_forward_length
                .unwrap_or(0)
                .max(config.feed_forward_length);
            self.gate[..seq_len * moe_intermediate].fill(0.0);
            self.up[..seq_len * moe_intermediate].fill(0.0);
            self.attn_out[..seq_len * config.q_width()].fill(0.0);
            self.proj[..seq_len * dim].fill(0.0);
            self.down[..seq_len * dim].fill(0.0);
            let expert_count = config.expert_count.unwrap_or(0);
            self.moe_router_logits[..seq_len * expert_count].fill(0.0);
            self.moe_routes[..seq_len].fill(MoeRoutingRow::default());
            self.moe_expert_counts[..expert_count].fill(0);
            self.moe_expert_offsets[..expert_count.saturating_add(1)].fill(0);
            self.moe_expert_cursors[..expert_count].fill(0);
            self.moe_token_indices[..seq_len].fill(0);
            self.moe_inputs[..seq_len * dim].fill(0.0);
            self.moe_shared_gate[..seq_len].fill(0.0);
            return;
        }
        let dim = config.embedding_length;
        self.xs = vec![0.0; seq_len * dim];
        self.normed = vec![0.0; seq_len * dim];
        self.q = vec![0.0; seq_len * config.q_width()];
        self.k = vec![0.0; seq_len * config.kv_width()];
        self.v = vec![0.0; seq_len * config.kv_width()];
        let moe_intermediate = config
            .expert_shared_feed_forward_length
            .unwrap_or(0)
            .max(config.feed_forward_length);
        self.gate = vec![0.0; seq_len * moe_intermediate];
        self.up = vec![0.0; seq_len * moe_intermediate];
        self.attn_out = vec![0.0; seq_len * config.q_width()];
        self.proj = vec![0.0; seq_len * dim];
        self.down = vec![0.0; seq_len * dim];
        let expert_count = config.expert_count.unwrap_or(0);
        self.moe_router_logits = vec![0.0; seq_len * expert_count];
        self.moe_routes = vec![MoeRoutingRow::default(); seq_len];
        self.moe_expert_counts = vec![0; expert_count];
        self.moe_expert_offsets = vec![0; expert_count.saturating_add(1)];
        self.moe_expert_cursors = vec![0; expert_count];
        self.moe_token_indices = vec![0; seq_len];
        self.moe_inputs = vec![0.0; seq_len * dim];
        self.moe_shared_gate = vec![0.0; seq_len];
        self.capacity = seq_len;
    }
}

pub struct LlamaModel {
    gguf: Arc<GgufFile>,
    config: LlamaConfig,
    token_embedding: String,
    output_norm: String,
    output: ResolvedWeight,
    layers: Vec<LayerWeights>,
    model_name: String,
    moe_cpu_execution: MoeCpuExecution,
    moe_telemetry: MoeTelemetry,
    lora: Option<crate::lora::LoraAdapter>,
    vector_cache: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    rope_freqs: RopeFreqs,
    gemma4_rope_freqs: Vec<RopeFreqs>,
    scratch: RwLock<ForwardScratch>,
    batch_scratch: RwLock<BatchScratch>,
    /// Pre-dequantized conv1d kernels per DeltaNet layer: [layer_index] -> flat f32 data.
    /// Layout: [conv_channels][kernel_size] (row-major, channels are rows).
    conv1d_kernels: Vec<Option<Vec<f32>>>,
}

impl LlamaModel {
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<ResolvedWeight> {
        let info = gguf.require_tensor(name)?;
        Ok(ResolvedWeight {
            data_offset: info.offset,
            nbytes: info.nbytes,
            rows: info.rows(),
            cols: info.row_len(),
            dtype: info.dtype,
            name: name.to_string(),
        })
    }

    fn resolve_packed_expert_weight(
        gguf: &GgufFile,
        name: &str,
        logical_expert: usize,
        expert_count: usize,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<ResolvedWeight> {
        let info = gguf.require_tensor(name)?;
        let expected_dimensions = [expected_cols, expected_rows, expert_count];
        if info.dimensions.as_slice() != expected_dimensions {
            return Err(XrtError::InvalidTensor(format!(
                "packed MoE tensor `{name}` has GGUF dimensions {:?}, expected {:?}",
                info.dimensions, expected_dimensions
            )));
        }
        if logical_expert >= expert_count || info.nbytes % expert_count != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "packed MoE tensor `{name}` cannot resolve logical expert {logical_expert} from {expert_count} equal byte spans"
            )));
        }
        let nbytes = info.nbytes / expert_count;
        let expert_offset = logical_expert.checked_mul(nbytes).ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "packed MoE tensor `{name}` expert offset overflowed"
            ))
        })?;
        let data_offset = info.offset.checked_add(expert_offset).ok_or_else(|| {
            XrtError::InvalidTensor(format!("packed MoE tensor `{name}` data offset overflowed"))
        })?;
        Ok(ResolvedWeight {
            data_offset,
            nbytes,
            rows: expected_rows,
            cols: expected_cols,
            dtype: info.dtype,
            name: name.to_string(),
        })
    }

    fn validate_weight_shape(
        weight: &ResolvedWeight,
        expected_rows: usize,
        expected_cols: usize,
        role: &str,
    ) -> Result<()> {
        if weight.rows != expected_rows || weight.cols != expected_cols {
            return Err(XrtError::InvalidTensor(format!(
                "{role} tensor `{}` has matrix shape {}x{}, expected {expected_rows}x{expected_cols}",
                weight.name, weight.rows, weight.cols
            )));
        }
        Ok(())
    }

    fn validate_vector_shape(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<()> {
        let tensor = gguf.require_tensor(name)?;
        if tensor.numel() != expected_len {
            return Err(XrtError::InvalidTensor(format!(
                "vector tensor `{name}` has {} elements, expected {expected_len}",
                tensor.numel()
            )));
        }
        Ok(())
    }

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Self> {
        Self::from_gguf_with_moe_execution(gguf, MoeCpuExecution::Legacy)
    }

    pub fn from_gguf_with_moe_execution(
        gguf: Arc<GgufFile>,
        moe_cpu_execution: MoeCpuExecution,
    ) -> Result<Self> {
        let config = LlamaConfig::from_gguf(&gguf)?;
        let token_embedding = "token_embd.weight".to_string();
        let output_norm = "output_norm.weight".to_string();
        let output_name = if gguf.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };

        gguf.require_tensor(&token_embedding)?;
        gguf.require_tensor(&output_norm)?;
        let output = Self::resolve_weight(&gguf, output_name)?;

        let mut layers = Vec::with_capacity(config.block_count);
        let mut conv1d_kernels = Vec::with_capacity(config.block_count);

        for index in 0..config.block_count {
            let attn_norm = format!("blk.{index}.attn_norm.weight");
            gguf.require_tensor(&attn_norm)?;

            // Detect layer type by probing for DeltaNet-specific tensor
            let is_recurrent = config.is_recurrent(index);

            let attn = if config.is_gemma4() {
                let v_name = format!("blk.{index}.attn_v.weight");
                gguf.require_tensor(&format!("blk.{index}.attn_q_norm.weight"))?;
                gguf.require_tensor(&format!("blk.{index}.attn_k_norm.weight"))?;
                gguf.require_tensor(&format!("blk.{index}.post_attention_norm.weight"))?;
                AttnWeights::Gemma4 {
                    attn_q: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_q.weight"))?,
                    attn_k: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_k.weight"))?,
                    attn_v: if gguf.tensor_info(&v_name).is_some() {
                        Some(Self::resolve_weight(&gguf, &v_name)?)
                    } else {
                        None
                    },
                    attn_output: Self::resolve_weight(
                        &gguf,
                        &format!("blk.{index}.attn_output.weight"),
                    )?,
                    attn_q_norm: format!("blk.{index}.attn_q_norm.weight"),
                    attn_k_norm: format!("blk.{index}.attn_k_norm.weight"),
                    attn_post_norm: format!("blk.{index}.post_attention_norm.weight"),
                }
            } else if is_recurrent {
                let descriptor = config.deltanet_state_descriptor().ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "hybrid recurrent layer is missing validated DeltaNet geometry".to_string(),
                    )
                })?;
                let conv_channels = descriptor
                    .state_size()
                    .checked_mul(descriptor.group_count())
                    .and_then(|value| value.checked_mul(2))
                    .and_then(|value| value.checked_add(descriptor.inner_size()))
                    .ok_or_else(|| {
                        XrtError::InvalidMetadata(
                            "DeltaNet convolution channel geometry overflowed".to_string(),
                        )
                    })?;
                let head_v_dim = descriptor.inner_size() / descriptor.dt_rank();
                let attn_qkv =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.attn_qkv.weight"))?;
                let attn_gate =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.attn_gate.weight"))?;
                let ssm_alpha =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.ssm_alpha.weight"))?;
                let ssm_beta =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.ssm_beta.weight"))?;
                let ssm_out = Self::resolve_weight(&gguf, &format!("blk.{index}.ssm_out.weight"))?;
                Self::validate_weight_shape(
                    &attn_qkv,
                    conv_channels,
                    config.embedding_length,
                    "DeltaNet QKV",
                )?;
                Self::validate_weight_shape(
                    &attn_gate,
                    descriptor.inner_size(),
                    config.embedding_length,
                    "DeltaNet gate",
                )?;
                Self::validate_weight_shape(
                    &ssm_alpha,
                    descriptor.dt_rank(),
                    config.embedding_length,
                    "DeltaNet alpha",
                )?;
                Self::validate_weight_shape(
                    &ssm_beta,
                    descriptor.dt_rank(),
                    config.embedding_length,
                    "DeltaNet beta",
                )?;
                Self::validate_weight_shape(
                    &ssm_out,
                    config.embedding_length,
                    descriptor.inner_size(),
                    "DeltaNet output",
                )?;
                let ssm_a = format!("blk.{index}.ssm_a");
                let ssm_dt_bias = format!("blk.{index}.ssm_dt.bias");
                let ssm_norm = format!("blk.{index}.ssm_norm.weight");
                Self::validate_vector_shape(&gguf, &ssm_a, descriptor.dt_rank())?;
                Self::validate_vector_shape(&gguf, &ssm_dt_bias, descriptor.dt_rank())?;
                Self::validate_vector_shape(&gguf, &ssm_norm, head_v_dim)?;
                AttnWeights::DeltaNet {
                    attn_qkv,
                    attn_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    ssm_out,
                }
            } else if config.is_hybrid() {
                // Qwen3.5 full attention layer (Q+gate interleaved)
                let attn_qg = Self::resolve_weight(&gguf, &format!("blk.{index}.attn_q.weight"))?;
                let attn_k = Self::resolve_weight(&gguf, &format!("blk.{index}.attn_k.weight"))?;
                let attn_v = Self::resolve_weight(&gguf, &format!("blk.{index}.attn_v.weight"))?;
                let attn_output =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.attn_output.weight"))?;
                let qg_width = config.q_width().checked_mul(2).ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "Qwen3.5 interleaved query/gate width overflowed".to_string(),
                    )
                })?;
                Self::validate_weight_shape(
                    &attn_qg,
                    qg_width,
                    config.embedding_length,
                    "Qwen3.5 interleaved query/gate",
                )?;
                Self::validate_weight_shape(
                    &attn_k,
                    config.kv_width(),
                    config.embedding_length,
                    "Qwen3.5 key",
                )?;
                Self::validate_weight_shape(
                    &attn_v,
                    config.kv_width(),
                    config.embedding_length,
                    "Qwen3.5 value",
                )?;
                Self::validate_weight_shape(
                    &attn_output,
                    config.embedding_length,
                    config.q_width(),
                    "Qwen3.5 attention output",
                )?;
                let attn_q_norm = format!("blk.{index}.attn_q_norm.weight");
                let attn_k_norm = format!("blk.{index}.attn_k_norm.weight");
                Self::validate_vector_shape(&gguf, &attn_q_norm, config.head_dim())?;
                Self::validate_vector_shape(&gguf, &attn_k_norm, config.head_dim())?;
                AttnWeights::Qwen35Attn {
                    attn_qg,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                }
            } else {
                // Standard transformer attention (llama, qwen2, qwen3)
                let q_norm_name = format!("blk.{index}.attn_q_norm.weight");
                let k_norm_name = format!("blk.{index}.attn_k_norm.weight");
                let has_qk_norm = gguf.tensor_info(&q_norm_name).is_some();
                let q_bias_name = format!("blk.{index}.attn_q.bias");
                let k_bias_name = format!("blk.{index}.attn_k.bias");
                let v_bias_name = format!("blk.{index}.attn_v.bias");
                AttnWeights::Standard {
                    attn_q: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_q.weight"))?,
                    attn_k: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_k.weight"))?,
                    attn_v: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_v.weight"))?,
                    attn_output: Self::resolve_weight(
                        &gguf,
                        &format!("blk.{index}.attn_output.weight"),
                    )?,
                    attn_q_norm: if has_qk_norm { Some(q_norm_name) } else { None },
                    attn_k_norm: if has_qk_norm { Some(k_norm_name) } else { None },
                    attn_q_bias: if gguf.tensor_info(&q_bias_name).is_some() {
                        Some(q_bias_name)
                    } else {
                        None
                    },
                    attn_k_bias: if gguf.tensor_info(&k_bias_name).is_some() {
                        Some(k_bias_name)
                    } else {
                        None
                    },
                    attn_v_bias: if gguf.tensor_info(&v_bias_name).is_some() {
                        Some(v_bias_name)
                    } else {
                        None
                    },
                }
            };

            // FFN norm name differs by architecture.
            let ffn_norm = if config.is_hybrid() {
                format!("blk.{index}.post_attention_norm.weight")
            } else {
                format!("blk.{index}.ffn_norm.weight")
            };
            gguf.require_tensor(&ffn_norm)?;

            let ffn = if config.is_gemma4() {
                let layer_output_scale = format!("blk.{index}.layer_output_scale.weight");
                gguf.require_tensor(&format!("blk.{index}.post_ffw_norm.weight"))?;
                FfnWeights::Gemma4Dense {
                    gate: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_gate.weight"))?,
                    down: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_down.weight"))?,
                    up: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_up.weight"))?,
                    post_ffw_norm: format!("blk.{index}.post_ffw_norm.weight"),
                    layer_output_scale: if gguf.tensor_info(&layer_output_scale).is_some() {
                        Some(layer_output_scale)
                    } else {
                        None
                    },
                }
            } else if config.is_moe() {
                let n_experts = config.expert_count.ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "MoE model is missing expert_count metadata".to_string(),
                    )
                })?;
                let selected_per_token = config.expert_used_count.ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "MoE model is missing expert_used_count metadata".to_string(),
                    )
                })?;
                let descriptor = MoeLayerDescriptor::new(
                    index,
                    n_experts,
                    selected_per_token,
                    config.embedding_length,
                    config.feed_forward_length,
                )?;
                let router =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_gate_inp.weight"))?;
                Self::validate_weight_shape(
                    &router,
                    n_experts,
                    config.embedding_length,
                    "MoE router",
                )?;
                let mut experts = Vec::with_capacity(n_experts);
                let packed_gate = format!("blk.{index}.ffn_gate_exps.weight");
                let packed_down = format!("blk.{index}.ffn_down_exps.weight");
                let packed_up = format!("blk.{index}.ffn_up_exps.weight");
                let packed_presence = [
                    gguf.tensor_info(&packed_gate).is_some(),
                    gguf.tensor_info(&packed_down).is_some(),
                    gguf.tensor_info(&packed_up).is_some(),
                ];
                if packed_presence.iter().any(|present| *present)
                    && !packed_presence.iter().all(|present| *present)
                {
                    return Err(XrtError::InvalidTensor(format!(
                        "MoE layer {index} must provide all of ffn_gate_exps, ffn_down_exps, and ffn_up_exps packed tensors"
                    )));
                }
                for e in 0..n_experts {
                    let (gate, down, up) = if packed_presence[0] {
                        (
                            Self::resolve_packed_expert_weight(
                                &gguf,
                                &packed_gate,
                                e,
                                n_experts,
                                config.feed_forward_length,
                                config.embedding_length,
                            )?,
                            Self::resolve_packed_expert_weight(
                                &gguf,
                                &packed_down,
                                e,
                                n_experts,
                                config.embedding_length,
                                config.feed_forward_length,
                            )?,
                            Self::resolve_packed_expert_weight(
                                &gguf,
                                &packed_up,
                                e,
                                n_experts,
                                config.feed_forward_length,
                                config.embedding_length,
                            )?,
                        )
                    } else {
                        (
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{index}.ffn_gate.{e}.weight"),
                            )?,
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{index}.ffn_down.{e}.weight"),
                            )?,
                            Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_up.{e}.weight"))?,
                        )
                    };
                    Self::validate_weight_shape(
                        &gate,
                        config.feed_forward_length,
                        config.embedding_length,
                        "MoE expert gate",
                    )?;
                    Self::validate_weight_shape(
                        &up,
                        config.feed_forward_length,
                        config.embedding_length,
                        "MoE expert up",
                    )?;
                    Self::validate_weight_shape(
                        &down,
                        config.embedding_length,
                        config.feed_forward_length,
                        "MoE expert down",
                    )?;
                    experts.push(MoeExpertWeights { gate, down, up });
                }
                let shared_names = [
                    format!("blk.{index}.ffn_gate_inp_shexp.weight"),
                    format!("blk.{index}.ffn_gate_shexp.weight"),
                    format!("blk.{index}.ffn_down_shexp.weight"),
                    format!("blk.{index}.ffn_up_shexp.weight"),
                ];
                let shared_presence = shared_names
                    .iter()
                    .map(|name| gguf.tensor_info(name).is_some())
                    .collect::<Vec<_>>();
                let shared = match config.expert_shared_feed_forward_length {
                    Some(intermediate_size) => {
                        if !shared_presence.iter().all(|present| *present) {
                            return Err(XrtError::InvalidTensor(format!(
                                "MoE layer {index} declares a shared expert but does not provide all gate selector, gate, down, and up tensors"
                            )));
                        }
                        let gate_selector = Self::resolve_weight(&gguf, &shared_names[0])?;
                        let gate = Self::resolve_weight(&gguf, &shared_names[1])?;
                        let down = Self::resolve_weight(&gguf, &shared_names[2])?;
                        let up = Self::resolve_weight(&gguf, &shared_names[3])?;
                        Self::validate_weight_shape(
                            &gate_selector,
                            1,
                            config.embedding_length,
                            "MoE shared expert selector",
                        )?;
                        Self::validate_weight_shape(
                            &gate,
                            intermediate_size,
                            config.embedding_length,
                            "MoE shared expert gate",
                        )?;
                        Self::validate_weight_shape(
                            &up,
                            intermediate_size,
                            config.embedding_length,
                            "MoE shared expert up",
                        )?;
                        Self::validate_weight_shape(
                            &down,
                            config.embedding_length,
                            intermediate_size,
                            "MoE shared expert down",
                        )?;
                        Some(MoeSharedExpertWeights {
                            gate_selector,
                            gate,
                            down,
                            up,
                            intermediate_size,
                        })
                    }
                    None => {
                        if shared_presence.iter().any(|present| *present) {
                            return Err(XrtError::InvalidTensor(format!(
                                "MoE layer {index} contains shared-expert tensors but expert_shared_feed_forward_length is missing or zero"
                            )));
                        }
                        None
                    }
                };
                FfnWeights::Moe {
                    descriptor,
                    router,
                    experts,
                    shared,
                }
            } else {
                FfnWeights::Dense {
                    gate: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_gate.weight"))?,
                    down: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_down.weight"))?,
                    up: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_up.weight"))?,
                }
            };

            let layer = LayerWeights {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            };

            // Pre-dequantize conv1d kernels for DeltaNet layers (F32, small)
            if is_recurrent {
                let conv_name = format!("blk.{index}.ssm_conv1d.weight");
                let info = gguf.require_tensor(&conv_name)?;
                let descriptor = config.deltanet_state_descriptor().ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "hybrid recurrent layer is missing validated DeltaNet geometry".to_string(),
                    )
                })?;
                let conv_channels = descriptor
                    .state_size()
                    .checked_mul(descriptor.group_count())
                    .and_then(|value| value.checked_mul(2))
                    .and_then(|value| value.checked_add(descriptor.inner_size()))
                    .ok_or_else(|| {
                        XrtError::InvalidMetadata(
                            "DeltaNet convolution channel geometry overflowed".to_string(),
                        )
                    })?;
                if info.dtype != DType::F32
                    || info.row_len() != descriptor.conv_kernel()
                    || info.rows() != conv_channels
                {
                    return Err(XrtError::InvalidTensor(format!(
                        "DeltaNet convolution tensor `{conv_name}` must be F32 with shape {}x{}, found {:?} {}x{}",
                        conv_channels,
                        descriptor.conv_kernel(),
                        info.dtype,
                        info.rows(),
                        info.row_len()
                    )));
                }
                let bytes = gguf.tensor_data(&conv_name)?;
                let total = info.numel();
                let expected_bytes =
                    total
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            XrtError::InvalidTensor(format!(
                                "DeltaNet convolution tensor `{conv_name}` byte size overflowed"
                            ))
                        })?;
                if bytes.len() != expected_bytes {
                    return Err(XrtError::InvalidTensor(format!(
                        "DeltaNet convolution tensor `{conv_name}` has {} bytes, expected {expected_bytes}",
                        bytes.len()
                    )));
                }
                let mut kernel = vec![0.0f32; total];
                // F32 tensor: direct decode
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    kernel[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                conv1d_kernels.push(Some(kernel));
            } else {
                conv1d_kernels.push(None);
            }

            layers.push(layer);
        }

        let model_name = gguf
            .metadata_string("general.name")
            .map(ToOwned::to_owned)
            .or_else(|| {
                Path::new(gguf.path())
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "llama".to_string());

        let rope_freqs = RopeFreqs::new(
            config.rope_dimension_count,
            config.rope_freq_base,
            config.rope_freq_scale,
        );
        let gemma4_rope_freqs = config
            .gemma4
            .as_ref()
            .map(|gemma4| {
                gemma4
                    .layers
                    .iter()
                    .map(|layer| {
                        RopeFreqs::new(
                            layer.rope_dimension_count,
                            layer.rope_freq_base,
                            config.rope_freq_scale,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let scratch = RwLock::new(ForwardScratch::new(&config));
        let moe_telemetry = MoeTelemetry::new(config.expert_count.unwrap_or(0));

        if config.is_moe() {
            info!(
                "loaded {} model {} with {} layers, {} heads, {} kv heads, MoE {}/{} experts",
                config.architecture,
                model_name,
                config.block_count,
                config.attention_head_count,
                config.attention_head_count_kv,
                config.expert_used_count.unwrap_or(2),
                config.expert_count.unwrap_or(0),
            );
        } else {
            info!(
                "loaded {} model {} with {} layers ({} recurrent, {} attention), {} heads, {} kv heads",
                config.architecture,
                model_name,
                config.block_count,
                (0..config.block_count).filter(|&i| config.is_recurrent(i)).count(),
                (0..config.block_count).filter(|&i| !config.is_recurrent(i)).count(),
                config.attention_head_count,
                config.attention_head_count_kv
            );
        }

        Ok(Self {
            gguf,
            config,
            token_embedding,
            output_norm,
            output,
            layers,
            model_name,
            moe_cpu_execution,
            moe_telemetry,
            lora: None,
            vector_cache: RwLock::new(HashMap::new()),
            rope_freqs,
            gemma4_rope_freqs,
            scratch,
            batch_scratch: RwLock::new(BatchScratch::new()),
            conv1d_kernels,
        })
    }

    /// Load a LoRA adapter from a GGUF file. Replaces any previously loaded adapter.
    pub fn load_lora(&mut self, path: &str) -> Result<()> {
        let adapter = crate::lora::LoraAdapter::load(path)?;
        self.lora = Some(adapter);
        Ok(())
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn moe_cpu_execution(&self) -> MoeCpuExecution {
        self.moe_cpu_execution
    }

    pub fn moe_telemetry(&self) -> MoeTelemetrySnapshot {
        self.moe_telemetry.snapshot()
    }

    #[cfg(feature = "moe-route-trace")]
    pub fn start_moe_route_trace(&self, max_entries: usize) -> Result<()> {
        self.moe_telemetry.start_route_trace(max_entries)
    }

    #[cfg(feature = "moe-route-trace")]
    pub fn take_moe_route_trace(&self) -> Option<crate::moe::MoeRouteTrace> {
        self.moe_telemetry.take_route_trace()
    }

    pub fn moe_layer_descriptor(&self, layer_index: usize) -> Option<&MoeLayerDescriptor> {
        match &self.layers.get(layer_index)?.ffn {
            FfnWeights::Moe { descriptor, .. } => Some(descriptor),
            FfnWeights::Dense { .. } | FfnWeights::Gemma4Dense { .. } => None,
        }
    }

    /// Compute the canonical logical route for one MoE layer.
    ///
    /// Placement remains a runtime concern; this method exposes only model
    /// semantics and logical expert IDs.
    pub fn route_moe_layer(
        &self,
        layer_index: usize,
        input: &[f32],
        router_logits: &mut [f32],
        route: &mut MoeRoutingRow,
    ) -> Result<()> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!("missing model layer {layer_index} for MoE routing"))
        })?;
        let FfnWeights::Moe {
            descriptor, router, ..
        } = &layer.ffn
        else {
            return Err(XrtError::Unsupported(format!(
                "model layer {layer_index} is not an MoE layer"
            )));
        };
        if input.len() != descriptor.hidden_size()
            || router_logits.len() != descriptor.expert_count()
        {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} router input/scratch geometry does not match its descriptor"
            )));
        }
        self.linear_resolved(router, input, router_logits)?;
        route_top_k(router_logits, descriptor.selected_per_token(), route)?;
        self.moe_telemetry.record_route(route);
        #[cfg(feature = "moe-route-trace")]
        self.moe_telemetry.record_route_trace(layer_index, route);
        Ok(())
    }

    /// Normalize already-computed router logits into the canonical logical
    /// route while retaining model-level routing telemetry.
    pub fn route_moe_logits(
        &self,
        layer_index: usize,
        router_logits: &[f32],
        route: &mut MoeRoutingRow,
    ) -> Result<()> {
        let descriptor = self.moe_layer_descriptor(layer_index).ok_or_else(|| {
            XrtError::Unsupported(format!("model layer {layer_index} is not an MoE layer"))
        })?;
        if router_logits.len() != descriptor.expert_count() {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} received {} router logits, expected {}",
                router_logits.len(),
                descriptor.expert_count()
            )));
        }
        route_top_k(router_logits, descriptor.selected_per_token(), route)?;
        self.moe_telemetry.record_route(route);
        #[cfg(feature = "moe-route-trace")]
        self.moe_telemetry.record_route_trace(layer_index, route);
        Ok(())
    }

    /// Execute one logical expert exactly on CPU into caller-owned scratch.
    pub fn execute_moe_expert_into(
        &self,
        layer_index: usize,
        logical_expert: usize,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing model layer {layer_index} for MoE execution"
            ))
        })?;
        let FfnWeights::Moe {
            descriptor,
            experts,
            ..
        } = &layer.ffn
        else {
            return Err(XrtError::Unsupported(format!(
                "model layer {layer_index} is not an MoE layer"
            )));
        };
        let expert = experts.get(logical_expert).ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} has no logical expert {logical_expert}"
            ))
        })?;
        if input.len() != descriptor.hidden_size()
            || gate.len() != descriptor.intermediate_size()
            || up.len() != descriptor.intermediate_size()
            || output.len() != descriptor.hidden_size()
        {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} expert scratch geometry does not match its descriptor"
            )));
        }
        self.linear_pair_resolved(&expert.gate, &expert.up, input, gate, up)?;
        swiglu(gate, up);
        self.linear_resolved(&expert.down, gate, output)
    }

    /// Execute independent selected experts concurrently inside the existing
    /// bounded CPU worker budget.
    ///
    /// Each logical expert owns one disjoint row in every caller-provided
    /// scratch buffer. Nested projection kernels therefore execute serially on
    /// their assigned worker instead of recursively dispatching or
    /// oversubscribing the host.
    pub fn execute_moe_experts_parallel_into(
        &self,
        layer_index: usize,
        logical_experts: &[usize],
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<()> {
        let descriptor = self.moe_layer_descriptor(layer_index).ok_or_else(|| {
            XrtError::Unsupported(format!("model layer {layer_index} is not an MoE layer"))
        })?;
        let task_count = logical_experts.len();
        let gate_len = task_count
            .checked_mul(descriptor.intermediate_size())
            .ok_or_else(|| XrtError::Runtime("parallel MoE gate size overflowed".to_string()))?;
        let output_len = task_count
            .checked_mul(descriptor.hidden_size())
            .ok_or_else(|| XrtError::Runtime("parallel MoE output size overflowed".to_string()))?;
        if input.len() != descriptor.hidden_size()
            || gate.len() != gate_len
            || up.len() != gate_len
            || outputs.len() != output_len
        {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} parallel expert scratch geometry does not match {} tasks",
                task_count
            )));
        }
        if logical_experts
            .iter()
            .any(|&logical_expert| logical_expert >= descriptor.expert_count())
        {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} parallel expert list exceeds {} logical experts",
                descriptor.expert_count()
            )));
        }
        if task_count == 0 {
            return Ok(());
        }
        if task_count == 1 {
            return self.execute_moe_expert_into(
                layer_index,
                logical_experts[0],
                input,
                gate,
                up,
                outputs,
            );
        }

        let layer = self.layers.get(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing model layer {layer_index} for parallel MoE execution"
            ))
        })?;
        let FfnWeights::Moe { experts, .. } = &layer.ffn else {
            return Err(XrtError::Unsupported(format!(
                "model layer {layer_index} is not an MoE layer"
            )));
        };
        let selected_experts: Vec<&MoeExpertWeights> = logical_experts
            .iter()
            .map(|&logical_expert| &experts[logical_expert])
            .collect();
        let grouped_dtype_supported = |dtype: DType| {
            matches!(
                dtype,
                DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::MXFP4
            )
        };
        let gate_up_dtype = selected_experts[0].gate.dtype;
        let down_dtype = selected_experts[0].down.dtype;
        let can_group_rows = self.lora.is_none()
            && grouped_dtype_supported(gate_up_dtype)
            && grouped_dtype_supported(down_dtype)
            && selected_experts.iter().all(|expert| {
                expert.gate.dtype == gate_up_dtype
                    && expert.up.dtype == gate_up_dtype
                    && expert.down.dtype == down_dtype
            });
        if can_group_rows {
            let mut gate_up_matrices = Vec::with_capacity(task_count * 2);
            for expert in &selected_experts {
                gate_up_matrices.push(
                    self.gguf
                        .tensor_data_raw(expert.gate.data_offset, expert.gate.nbytes),
                );
            }
            for expert in &selected_experts {
                gate_up_matrices.push(
                    self.gguf
                        .tensor_data_raw(expert.up.data_offset, expert.up.nbytes),
                );
            }
            let gate_up_rows = vec![descriptor.intermediate_size(); task_count * 2];
            let mut gate_up_outputs: Vec<&mut [f32]> = gate
                .chunks_exact_mut(descriptor.intermediate_size())
                .take(task_count)
                .collect();
            gate_up_outputs.extend(
                up.chunks_exact_mut(descriptor.intermediate_size())
                    .take(task_count),
            );
            matvec_quantized_fused(
                &gate_up_matrices,
                &gate_up_rows,
                descriptor.hidden_size(),
                gate_up_dtype,
                input,
                &mut gate_up_outputs,
            )?;
            drop(gate_up_outputs);
            for (gate_row, up_row) in gate
                .chunks_exact_mut(descriptor.intermediate_size())
                .zip(up.chunks_exact(descriptor.intermediate_size()))
                .take(task_count)
            {
                swiglu(gate_row, up_row);
            }

            let down_matrices: Vec<&[u8]> = selected_experts
                .iter()
                .map(|expert| {
                    self.gguf
                        .tensor_data_raw(expert.down.data_offset, expert.down.nbytes)
                })
                .collect();
            let down_inputs: Vec<&[f32]> = gate
                .chunks_exact(descriptor.intermediate_size())
                .take(task_count)
                .collect();
            let mut down_outputs: Vec<&mut [f32]> = outputs
                .chunks_exact_mut(descriptor.hidden_size())
                .take(task_count)
                .collect();
            return matvec_quantized_independent(
                &down_matrices,
                descriptor.hidden_size(),
                descriptor.intermediate_size(),
                down_dtype,
                &down_inputs,
                &mut down_outputs,
            );
        }

        let intermediate_size = descriptor.intermediate_size();
        let hidden_size = descriptor.hidden_size();
        let gate_address = gate.as_mut_ptr() as usize;
        let up_address = up.as_mut_ptr() as usize;
        let output_address = outputs.as_mut_ptr() as usize;
        let errors = std::sync::Mutex::new(None);
        let worker_result = global_expert_pool().execute_scoped(task_count, |start, end| {
            for task_index in start..end {
                if errors
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .is_some()
                {
                    return;
                }
                // SAFETY: every task owns a distinct fixed-size row in each
                // mutable buffer. The enclosing scoped dispatch joins before
                // any caller can reuse or move those buffers.
                let result = unsafe {
                    let gate = std::slice::from_raw_parts_mut(
                        (gate_address as *mut f32).add(task_index * intermediate_size),
                        intermediate_size,
                    );
                    let up = std::slice::from_raw_parts_mut(
                        (up_address as *mut f32).add(task_index * intermediate_size),
                        intermediate_size,
                    );
                    let output = std::slice::from_raw_parts_mut(
                        (output_address as *mut f32).add(task_index * hidden_size),
                        hidden_size,
                    );
                    self.execute_moe_expert_into(
                        layer_index,
                        logical_experts[task_index],
                        input,
                        gate,
                        up,
                        output,
                    )
                };
                if let Err(error) = result {
                    let mut first_error = errors
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if first_error.is_none() {
                        *first_error = Some(error);
                    }
                    return;
                }
            }
        });
        if let Err(error) = worker_result {
            self.moe_telemetry.record_worker_failure();
            return Err(error);
        }
        if let Some(error) = errors
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            self.moe_telemetry.record_worker_failure();
            return Err(error);
        }
        Ok(())
    }

    pub fn moe_layer_has_shared_expert(&self, layer_index: usize) -> bool {
        self.layers.get(layer_index).is_some_and(|layer| {
            matches!(
                &layer.ffn,
                FfnWeights::Moe {
                    shared: Some(_),
                    ..
                }
            )
        })
    }

    pub fn moe_shared_intermediate_size(&self, layer_index: usize) -> Option<usize> {
        self.layers
            .get(layer_index)
            .and_then(|layer| match &layer.ffn {
                FfnWeights::Moe {
                    shared: Some(shared),
                    ..
                } => Some(shared.intermediate_size),
                _ => None,
            })
    }

    /// Execute the optional always-on shared expert and return its sigmoid
    /// selector weight. The caller performs the canonical ordered merge.
    pub fn execute_shared_moe_expert_into(
        &self,
        layer_index: usize,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<f32> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing model layer {layer_index} for shared MoE execution"
            ))
        })?;
        let FfnWeights::Moe {
            descriptor,
            shared: Some(shared),
            ..
        } = &layer.ffn
        else {
            return Err(XrtError::Unsupported(format!(
                "model layer {layer_index} has no shared MoE expert"
            )));
        };
        if input.len() != descriptor.hidden_size()
            || gate.len() != shared.intermediate_size
            || up.len() != shared.intermediate_size
            || output.len() != descriptor.hidden_size()
        {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {layer_index} shared-expert scratch geometry does not match its descriptor"
            )));
        }
        let mut selector = [0.0f32; 1];
        self.linear_resolved(&shared.gate_selector, input, &mut selector)?;
        let shared_weight = 1.0 / (1.0 + (-selector[0]).exp());
        if !shared_weight.is_finite() {
            return Err(XrtError::Runtime(format!(
                "MoE layer {layer_index} shared-expert selector is non-finite"
            )));
        }
        self.linear_pair_resolved(&shared.gate, &shared.up, input, gate, up)?;
        swiglu(gate, up);
        self.linear_resolved(&shared.down, gate, output)?;
        Ok(shared_weight)
    }

    /// Run a forward pass through only the first `n_layers` layers.
    /// Used as a cheap draft model for self-speculative decoding.
    /// The cache must have at least `n_layers` layers.
    pub fn forward_draft<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_token_inner(
            token_id,
            position,
            Some(n_layers),
            None,
            cache,
            output_logits,
        )
    }

    pub fn forward_draft_with_state<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_token_inner(
            token_id,
            position,
            Some(n_layers),
            recurrent_state,
            cache,
            output_logits,
        )
    }

    pub fn forward_token<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_token_inner(token_id, position, None, None, cache, output_logits)
    }

    pub fn forward_token_with_state<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_token_inner(
            token_id,
            position,
            None,
            recurrent_state,
            cache,
            output_logits,
        )
    }

    fn forward_token_inner<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        max_layers: Option<usize>,
        mut recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        let n_layers = max_layers.unwrap_or(self.config.block_count);
        if n_layers > self.config.block_count {
            return Err(XrtError::Model(format!(
                "draft requested {n_layers} layers, but model has {}",
                self.config.block_count
            )));
        }
        if self.config.is_hybrid() && n_layers != self.config.block_count {
            return Err(XrtError::Unsupported(
                "partial-layer drafting is not supported for hybrid recurrent models".to_string(),
            ));
        }
        if let Some(descriptor) = self.config.deltanet_state_descriptor() {
            let state = recurrent_state.as_deref_mut().ok_or_else(|| {
                XrtError::Runtime(
                    "hybrid model forward requires session-owned DeltaNet state".to_string(),
                )
            })?;
            if state.descriptor() != descriptor {
                return Err(XrtError::Runtime(
                    "DeltaNet session state geometry does not match the loaded model".to_string(),
                ));
            }
            state.validate_position(position)?;
        } else if recurrent_state.is_some() {
            return Err(XrtError::Runtime(
                "DeltaNet session state was supplied to a non-hybrid model".to_string(),
            ));
        }
        let mut recurrent_transaction = match recurrent_state.take() {
            Some(state) => Some(state.begin_token(position)?),
            None => None,
        };
        if cache.layers() < n_layers {
            return Err(XrtError::Model(format!(
                "KV cache has {} layers, but model requires {}",
                cache.layers(),
                n_layers
            )));
        }
        if self.config.is_gemma4() {
            return self.forward_gemma4_token_inner(
                token_id,
                position,
                n_layers,
                cache,
                output_logits,
            );
        }
        // Width check only meaningful for non-hybrid models (hybrid models have mixed widths)
        if !self.config.is_hybrid() && cache.width() != self.config.kv_width() {
            return Err(XrtError::Model(format!(
                "KV cache width {} does not match model width {}",
                cache.width(),
                self.config.kv_width()
            )));
        }

        let mut scratch = self.scratch.write();
        let head_dim = self.config.head_dim();
        let eps = self.config.rms_norm_eps;

        // Ensure caller's logits buffer is the right size.
        output_logits.resize(self.config.vocab_size, 0.0);

        // Destructure so the borrow checker can track each field independently.
        let ForwardScratch {
            normed,
            q,
            k,
            v,
            gate,
            up,
            attn_out,
            proj,
            down,
            sin_cache,
            cos_cache,
            dn_qkv,
            dn_gate,
            dn_alpha,
            dn_beta,
            dn_conv_out,
            dn_out,
            q35_qg,
            moe_router_logits,
            moe_expert_out,
        } = &mut *scratch;

        let mut x = self.embedding_lookup(token_id as usize)?;
        for (layer_index, layer) in self.layers[..n_layers].iter().enumerate() {
            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            apply_rmsnorm(&x, &attn_norm_weight, eps, normed);

            match &layer.attn {
                AttnWeights::Standard {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    attn_q_bias,
                    attn_k_bias,
                    attn_v_bias,
                } => {
                    if cache.len(layer_index) != position {
                        return Err(XrtError::Runtime(format!(
                            "KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                            cache.len(layer_index)
                        )));
                    }

                    // Fused QKV projection
                    {
                        let can_fuse_all = attn_q.dtype == attn_k.dtype
                            && attn_q.dtype == attn_v.dtype
                            && attn_q.cols == attn_k.cols
                            && attn_q.cols == attn_v.cols
                            && attn_q.dtype.is_quantized();

                        if can_fuse_all {
                            let q_data =
                                self.gguf.tensor_data_raw(attn_q.data_offset, attn_q.nbytes);
                            let k_data =
                                self.gguf.tensor_data_raw(attn_k.data_offset, attn_k.nbytes);
                            let v_data =
                                self.gguf.tensor_data_raw(attn_v.data_offset, attn_v.nbytes);
                            matvec_quantized_fused(
                                &[q_data, k_data, v_data],
                                &[attn_q.rows, attn_k.rows, attn_v.rows],
                                attn_q.cols,
                                attn_q.dtype,
                                normed,
                                &mut [&mut q[..], &mut k[..], &mut v[..]],
                            )?;
                        } else {
                            let can_fuse_qk = attn_q.dtype == attn_k.dtype
                                && attn_q.cols == attn_k.cols
                                && attn_q.dtype.is_quantized();
                            if can_fuse_qk {
                                let q_data =
                                    self.gguf.tensor_data_raw(attn_q.data_offset, attn_q.nbytes);
                                let k_data =
                                    self.gguf.tensor_data_raw(attn_k.data_offset, attn_k.nbytes);
                                matvec_quantized_fused(
                                    &[q_data, k_data],
                                    &[attn_q.rows, attn_k.rows],
                                    attn_q.cols,
                                    attn_q.dtype,
                                    normed,
                                    &mut [&mut q[..], &mut k[..]],
                                )?;
                            } else {
                                self.linear_resolved(attn_q, normed, q)?;
                                self.linear_resolved(attn_k, normed, k)?;
                            }
                            self.linear_resolved(attn_v, normed, v)?;
                        }
                    }

                    if let Some(q_bias_name) = attn_q_bias {
                        let q_bias = self.load_vector(q_bias_name)?;
                        Self::add_bias(q, &q_bias)?;
                    }
                    if let Some(k_bias_name) = attn_k_bias {
                        let k_bias = self.load_vector(k_bias_name)?;
                        Self::add_bias(k, &k_bias)?;
                    }
                    if let Some(v_bias_name) = attn_v_bias {
                        let v_bias = self.load_vector(v_bias_name)?;
                        Self::add_bias(v, &v_bias)?;
                    }

                    if let Some(ref q_norm_name) = attn_q_norm {
                        let q_norm_w = self.load_vector(q_norm_name)?;
                        self.apply_head_norm(
                            q,
                            self.config.attention_head_count,
                            head_dim,
                            &q_norm_w,
                        );
                    }
                    if let Some(ref k_norm_name) = attn_k_norm {
                        let k_norm_w = self.load_vector(k_norm_name)?;
                        self.apply_head_norm(
                            k,
                            self.config.attention_head_count_kv,
                            head_dim,
                            &k_norm_w,
                        );
                    }

                    self.rope_freqs
                        .precompute_sincos_into(position, sin_cache, cos_cache);
                    self.rope_freqs.apply_rotary_cached(
                        q,
                        self.config.attention_head_count,
                        head_dim,
                        sin_cache,
                        cos_cache,
                    );
                    self.rope_freqs.apply_rotary_cached(
                        k,
                        self.config.attention_head_count_kv,
                        head_dim,
                        sin_cache,
                        cos_cache,
                    );

                    cache.append(layer_index, k, v)?;
                    self.compute_attention(q, cache, layer_index, head_dim, attn_out)?;
                    self.linear_resolved(attn_output, attn_out, proj)?;
                    add_inplace(&mut x, proj);
                }

                AttnWeights::Qwen35Attn {
                    attn_qg,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                } => {
                    if cache.len(layer_index) != position {
                        return Err(XrtError::Runtime(format!(
                            "KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                            cache.len(layer_index)
                        )));
                    }

                    // Q+gate interleaved projection
                    self.linear_resolved(attn_qg, normed, q35_qg)?;
                    self.linear_resolved(attn_k, normed, k)?;
                    self.linear_resolved(attn_v, normed, v)?;

                    // Deinterleave Q and gate: layout is [Q_h0, gate_h0, Q_h1, gate_h1, ...]
                    let n_heads = self.config.attention_head_count;
                    for h in 0..n_heads {
                        let src_off = h * head_dim * 2;
                        let q_dst_off = h * head_dim;
                        q[q_dst_off..q_dst_off + head_dim]
                            .copy_from_slice(&q35_qg[src_off..src_off + head_dim]);
                        let gate_dst_off = h * head_dim;
                        // Store gate in dn_out temporarily (reuse buffer, it's large enough)
                        dn_out[gate_dst_off..gate_dst_off + head_dim]
                            .copy_from_slice(&q35_qg[src_off + head_dim..src_off + head_dim * 2]);
                    }

                    // Q and K normalization (per-head RMSNorm)
                    let q_norm_w = self.load_vector(attn_q_norm)?;
                    self.apply_head_norm(q, n_heads, head_dim, &q_norm_w);
                    let k_norm_w = self.load_vector(attn_k_norm)?;
                    self.apply_head_norm(
                        k,
                        self.config.attention_head_count_kv,
                        head_dim,
                        &k_norm_w,
                    );

                    // RoPE
                    self.rope_freqs
                        .precompute_sincos_into(position, sin_cache, cos_cache);
                    self.rope_freqs
                        .apply_rotary_cached(q, n_heads, head_dim, sin_cache, cos_cache);
                    self.rope_freqs.apply_rotary_cached(
                        k,
                        self.config.attention_head_count_kv,
                        head_dim,
                        sin_cache,
                        cos_cache,
                    );

                    cache.append(layer_index, k, v)?;
                    self.compute_attention(q, cache, layer_index, head_dim, attn_out)?;

                    // Apply sigmoid gate to attention output
                    let gate_total = n_heads * head_dim;
                    for i in 0..gate_total {
                        let g = 1.0 / (1.0 + (-dn_out[i]).exp()); // sigmoid
                        attn_out[i] *= g;
                    }

                    self.linear_resolved(attn_output, attn_out, proj)?;
                    add_inplace(&mut x, proj);
                }

                AttnWeights::DeltaNet {
                    attn_qkv,
                    attn_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    ssm_out,
                } => {
                    let transaction = recurrent_transaction.as_mut().ok_or_else(|| {
                        XrtError::Runtime(
                            "hybrid model forward requires session-owned DeltaNet state"
                                .to_string(),
                        )
                    })?;
                    let (conv_state, pending_conv_state, recurrent_state, pending_recurrent_state) =
                        transaction.layer_buffers_mut(layer_index)?;
                    let descriptor = self
                        .config
                        .deltanet_state_descriptor()
                        .expect("hybrid model descriptor was validated at load");
                    let num_groups = descriptor.group_count();
                    let state_size = descriptor.state_size(); // head_k_dim
                    let inner_size = descriptor.inner_size();
                    let dt_rank = descriptor.dt_rank(); // num_v_heads
                    let head_v_dim = inner_size / dt_rank;
                    let conv_kernel = descriptor.conv_kernel();
                    let conv_channels = state_size * num_groups * 2 + inner_size;

                    // 1. Fused QKV + gate + alpha + beta projections (all share normed input)
                    // Single dispatch: quantize input once, all 4 projections in one par_for
                    {
                        let projections = [attn_qkv, attn_gate, ssm_alpha, ssm_beta];
                        let can_fuse = projections.iter().all(|weight| weight.dtype.is_quantized())
                            && projections
                                .iter()
                                .all(|weight| weight.cols == attn_qkv.cols);
                        if can_fuse {
                            let qkv_data = self
                                .gguf
                                .tensor_data_raw(attn_qkv.data_offset, attn_qkv.nbytes);
                            let gate_data = self
                                .gguf
                                .tensor_data_raw(attn_gate.data_offset, attn_gate.nbytes);
                            let alpha_data = self
                                .gguf
                                .tensor_data_raw(ssm_alpha.data_offset, ssm_alpha.nbytes);
                            let beta_data = self
                                .gguf
                                .tensor_data_raw(ssm_beta.data_offset, ssm_beta.nbytes);
                            matvec_quantized_fused_mixed(
                                &[qkv_data, gate_data, alpha_data, beta_data],
                                &[attn_qkv.rows, attn_gate.rows, ssm_alpha.rows, ssm_beta.rows],
                                attn_qkv.cols,
                                &[
                                    attn_qkv.dtype,
                                    attn_gate.dtype,
                                    ssm_alpha.dtype,
                                    ssm_beta.dtype,
                                ],
                                normed,
                                &mut [
                                    &mut dn_qkv[..],
                                    &mut dn_gate[..],
                                    &mut dn_alpha[..],
                                    &mut dn_beta[..],
                                ],
                            )?;
                        } else {
                            self.linear_resolved(attn_qkv, normed, dn_qkv)?;
                            self.linear_resolved(attn_gate, normed, dn_gate)?;
                            self.linear_resolved(ssm_alpha, normed, dn_alpha)?;
                            self.linear_resolved(ssm_beta, normed, dn_beta)?;
                        }
                    }
                    let ssm_a_vec = self.load_vector(ssm_a)?;
                    let ssm_dt_vec = self.load_vector(ssm_dt_bias)?;

                    // Compute decay: exp(softplus(alpha + dt_bias) * ssm_a)
                    // ssm_a stores -exp(A_log) (already negative), so decay ∈ (0,1)
                    let mut decays = [0.0f32; 64]; // max groups
                    let mut betas = [0.0f32; 64];
                    for g in 0..dt_rank {
                        let alpha_biased = dn_alpha[g] + ssm_dt_vec[g];
                        let sp = (1.0 + alpha_biased.exp()).ln(); // softplus
                        decays[g] = (sp * ssm_a_vec[g]).exp();
                        betas[g] = 1.0 / (1.0 + (-dn_beta[g]).exp()); // sigmoid
                    }

                    // 3. Causal conv1d (kernel_size=4, depthwise)
                    let history = conv_kernel - 1; // 3 history slots
                    let kernel = self.conv1d_kernels[layer_index].as_ref().unwrap();

                    // Compute conv output BEFORE updating state.
                    // State holds (t-3, t-2, t-1), current input is dn_qkv (t-0).
                    // Restructured: iterate per-channel, 4-tap dot product.
                    // kernel layout: [channels][kernel_size], state layout: [history][channels]
                    {
                        let cs = conv_state;
                        let qkv = &dn_qkv[..conv_channels];
                        let out = &mut dn_conv_out[..conv_channels];
                        let kern = kernel.as_slice();
                        // For kernel_size=4: taps 0,1,2 from history, tap 3 from current input
                        // Unroll the 4-tap loop manually for speed
                        if conv_kernel == 4 {
                            for c in 0..conv_channels {
                                let kbase = c * 4;
                                out[c] = cs[c] * kern[kbase]
                                    + cs[conv_channels + c] * kern[kbase + 1]
                                    + cs[2 * conv_channels + c] * kern[kbase + 2]
                                    + qkv[c] * kern[kbase + 3];
                            }
                        } else {
                            for c in 0..conv_channels {
                                let mut sum = 0.0f32;
                                for tap in 0..history {
                                    sum +=
                                        cs[tap * conv_channels + c] * kern[c * conv_kernel + tap];
                                }
                                sum += qkv[c] * kern[c * conv_kernel + history];
                                out[c] = sum;
                            }
                        }
                    }

                    // Build the next convolution state out of place. It becomes visible only
                    // after the complete token (including output projection) succeeds.
                    if history > 0 {
                        if history > 1 {
                            pending_conv_state[..(history - 1) * conv_channels].copy_from_slice(
                                &conv_state[conv_channels..history * conv_channels],
                            );
                        }
                        pending_conv_state[(history - 1) * conv_channels..history * conv_channels]
                            .copy_from_slice(&dn_qkv[..conv_channels]);
                    }

                    // SiLU activation on conv output (SIMD)
                    silu_inplace_fast(&mut dn_conv_out[..conv_channels]);

                    // 4. Split into Q, K, V and L2 normalize Q, K in-place (SIMD)
                    let q_dim = state_size * num_groups;
                    let k_dim = state_size * num_groups;
                    for g in 0..num_groups {
                        let q_start = g * state_size;
                        l2_normalize(&mut dn_conv_out[q_start..q_start + state_size], eps);
                        let k_start = q_dim + g * state_size;
                        l2_normalize(&mut dn_conv_out[k_start..k_start + state_size], eps);
                    }

                    // 5. Delta rule update per group (SIMD-fused autoregressive)
                    // Fused: decay + dot(state,k) → sk, then state = decay*state + d*k
                    //        AND output = dot(updated_state, q) * scale in the same pass.
                    let v_heads = dt_rank.max(1);
                    let q_scale = 1.0 / (state_size as f32).sqrt();
                    for v_head in 0..v_heads {
                        // Qwen3.5 small variants can have more V heads than QK groups
                        // (for example 32 V heads and 16 QK groups in 4B / 9B). Map each
                        // value head onto its owning QK group instead of assuming a 1:1 layout.
                        let qk_group = qwen35_delta_qk_group(v_head, v_heads, num_groups);
                        let v_off = q_dim + k_dim + v_head * head_v_dim;
                        let q_off = qk_group * state_size;
                        let k_off = q_dim + qk_group * state_size;
                        let state_off = v_head * head_v_dim * state_size;
                        let decay = decays[v_head];
                        let beta = betas[v_head];

                        let state_g =
                            &recurrent_state[state_off..state_off + head_v_dim * state_size];
                        let next_state_g = &mut pending_recurrent_state
                            [state_off..state_off + head_v_dim * state_size];

                        unsafe {
                            delta_rule_group_out_of_place(
                                state_g,
                                next_state_g,
                                dn_conv_out.as_ptr().add(k_off),
                                dn_conv_out.as_ptr().add(q_off),
                                dn_conv_out.as_ptr().add(v_off),
                                dn_out.as_mut_ptr().add(v_head * head_v_dim),
                                head_v_dim,
                                state_size,
                                decay,
                                beta,
                                q_scale,
                            );
                        }
                    }

                    // 7. Gated RMSNorm: RMSNorm(output, ssm_norm) * SiLU(gate) (SIMD)
                    let norm_w = self.load_vector(ssm_norm)?;
                    for v_head in 0..v_heads {
                        let off = v_head * head_v_dim;
                        unsafe {
                            gated_rmsnorm(
                                &mut dn_out[off..off + head_v_dim],
                                dn_gate.as_ptr().add(off),
                                norm_w.as_ptr(),
                                eps,
                            );
                        }
                    }

                    // 8. Output projection
                    self.linear_resolved(ssm_out, &dn_out[..inner_size], proj)?;
                    add_inplace(&mut x, proj);
                }
                AttnWeights::Gemma4 { .. } => unreachable!(
                    "Gemma4 attention uses forward_gemma4_token_inner and never reaches this path"
                ),
            }

            // FFN (shared across all layer types)
            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            apply_rmsnorm(&x, &ffn_norm_weight, eps, normed);

            match &layer.ffn {
                FfnWeights::Dense {
                    gate: ffn_gate,
                    down: ffn_down,
                    up: ffn_up,
                } => {
                    if ffn_gate.dtype == ffn_up.dtype
                        && ffn_gate.cols == ffn_up.cols
                        && ffn_gate.dtype.is_quantized()
                    {
                        let gate_data = self
                            .gguf
                            .tensor_data_raw(ffn_gate.data_offset, ffn_gate.nbytes);
                        let up_data = self.gguf.tensor_data_raw(ffn_up.data_offset, ffn_up.nbytes);
                        matvec_quantized_fused(
                            &[gate_data, up_data],
                            &[ffn_gate.rows, ffn_up.rows],
                            ffn_gate.cols,
                            ffn_gate.dtype,
                            normed,
                            &mut [&mut gate[..], &mut up[..]],
                        )?;
                    } else {
                        self.linear_resolved(ffn_gate, normed, gate)?;
                        self.linear_resolved(ffn_up, normed, up)?;
                    }
                    swiglu(gate, up);
                    self.linear_resolved(ffn_down, gate, down)?;
                    add_inplace(&mut x, down);
                }
                FfnWeights::Moe {
                    descriptor,
                    router,
                    experts,
                    shared,
                } => {
                    let intermediate_size = descriptor.intermediate_size();
                    self.execute_moe_token(
                        descriptor,
                        router,
                        experts,
                        normed,
                        &mut moe_router_logits[..experts.len()],
                        &mut gate[..intermediate_size],
                        &mut up[..intermediate_size],
                        &mut moe_expert_out[..self.config.embedding_length],
                        down,
                    )?;
                    if let Some(shared) = shared {
                        self.execute_shared_moe_token(
                            shared,
                            normed,
                            &mut gate[..shared.intermediate_size],
                            &mut up[..shared.intermediate_size],
                            &mut moe_expert_out[..self.config.embedding_length],
                            down,
                        )?;
                    }
                    add_inplace(&mut x, down);
                }
                FfnWeights::Gemma4Dense { .. } => unreachable!(
                    "Gemma4 FFN uses forward_gemma4_token_inner and never reaches this path"
                ),
            }
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        apply_rmsnorm(&x, &output_norm_weight, eps, normed);

        // Output projection directly into caller's buffer (zero alloc per token).
        self.linear_resolved(&self.output, normed, output_logits)?;
        if let Some(transaction) = recurrent_transaction {
            transaction.commit()?;
        }
        Ok(())
    }

    fn forward_gemma4_token_inner<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_gemma4_token_inner_impl(
            token_id,
            position,
            n_layers,
            cache,
            output_logits,
            None,
        )
    }

    pub fn gemma4_layer0_trace<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut C,
    ) -> Result<Gemma4LayerTrace> {
        if !self.config.is_gemma4() {
            return Err(XrtError::Unsupported(
                "Gemma4 layer tracing requires a Gemma4 model".to_string(),
            ));
        }
        let mut trace = Gemma4LayerTrace::new(0, position);
        let mut logits = Vec::new();
        self.forward_gemma4_token_inner_impl(
            token_id,
            position,
            1,
            cache,
            &mut logits,
            Some(&mut trace),
        )?;
        Ok(trace)
    }

    fn forward_gemma4_token_inner_impl<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
        mut trace: Option<&mut Gemma4LayerTrace>,
    ) -> Result<()> {
        if cache.width() != self.config.kv_width() {
            return Err(XrtError::Model(format!(
                "KV cache width {} does not match Gemma4 max KV width {}",
                cache.width(),
                self.config.kv_width()
            )));
        }

        let gemma4 = self.config.gemma4.as_ref().ok_or_else(|| {
            XrtError::Runtime("Gemma4 config missing from Gemma4 model".to_string())
        })?;
        let eps = self.config.rms_norm_eps;
        let dim = self.config.embedding_length;
        let cache_width = cache.width();

        output_logits.resize(self.config.vocab_size, 0.0);

        let mut scratch = self.scratch.write();
        let ForwardScratch {
            normed,
            q,
            k,
            v,
            gate,
            up,
            attn_out,
            proj,
            down,
            sin_cache,
            cos_cache,
            ..
        } = &mut *scratch;

        macro_rules! trace_stage {
            ($name:literal, $values:expr) => {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record($name, $values);
                }
            };
        }

        let mut x = self.embedding_lookup(token_id as usize)?;
        let embedding_scale = (dim as f32).sqrt();
        for value in &mut x {
            *value *= embedding_scale;
        }
        trace_stage!("input", &x);

        for (layer_index, layer) in self.layers[..n_layers].iter().enumerate() {
            let layer_config = &gemma4.layers[layer_index];

            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            apply_rmsnorm(&x, &attn_norm_weight, eps, normed);
            trace_stage!("attention_norm", normed);

            match &layer.attn {
                AttnWeights::Gemma4 {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    attn_post_norm,
                } => {
                    if !layer_config.has_kv {
                        return Err(XrtError::Unsupported(format!(
                            "Gemma4 shared-KV reuse is not implemented for layer {layer_index}"
                        )));
                    }
                    if cache.len(layer_index) != position {
                        return Err(XrtError::Runtime(format!(
                            "KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                            cache.len(layer_index)
                        )));
                    }

                    let q_width = layer_config.q_width;
                    let kv_width = layer_config.kv_width;
                    q[..q_width].fill(0.0);
                    k[..cache_width].fill(0.0);
                    v[..cache_width].fill(0.0);

                    self.linear_resolved(attn_q, normed, &mut q[..q_width])?;
                    self.linear_resolved(attn_k, normed, &mut k[..kv_width])?;
                    if let Some(attn_v) = attn_v {
                        self.linear_resolved(attn_v, normed, &mut v[..kv_width])?;
                    } else {
                        v[..kv_width].copy_from_slice(&k[..kv_width]);
                    }
                    trace_stage!("q_projection", &q[..q_width]);
                    if let Some(trace) = trace.as_deref_mut() {
                        let mut reference = vec![0.0f32; q_width];
                        self.linear_resolved_float_reference(attn_q, normed, &mut reference)?;
                        trace.record("q_projection_float_reference", &reference);
                    }
                    trace_stage!("k_projection", &k[..kv_width]);
                    if let Some(trace) = trace.as_deref_mut() {
                        let mut reference = vec![0.0f32; kv_width];
                        self.linear_resolved_float_reference(attn_k, normed, &mut reference)?;
                        trace.record("k_projection_float_reference", &reference);
                    }
                    trace_stage!("v_projection", &v[..kv_width]);
                    if let Some(trace) = trace.as_deref_mut() {
                        let mut reference = vec![0.0f32; kv_width];
                        if let Some(attn_v) = attn_v {
                            self.linear_resolved_float_reference(attn_v, normed, &mut reference)?;
                        } else {
                            reference.copy_from_slice(&k[..kv_width]);
                        }
                        trace.record("v_projection_float_reference", &reference);
                    }

                    let q_norm_w = self.load_vector(attn_q_norm)?;
                    self.apply_head_norm(
                        &mut q[..q_width],
                        layer_config.head_count,
                        layer_config.head_dim,
                        &q_norm_w,
                    );
                    let k_norm_w = self.load_vector(attn_k_norm)?;
                    self.apply_head_norm(
                        &mut k[..kv_width],
                        layer_config.kv_head_count,
                        layer_config.head_dim,
                        &k_norm_w,
                    );
                    self.apply_head_rmsnorm_unweighted(
                        &mut v[..kv_width],
                        layer_config.kv_head_count,
                        layer_config.head_dim,
                    );
                    trace_stage!("q_head_norm", &q[..q_width]);
                    trace_stage!("k_head_norm", &k[..kv_width]);
                    trace_stage!("v_head_norm", &v[..kv_width]);

                    let rope = &self.gemma4_rope_freqs[layer_index];
                    let rope_half = layer_config.rope_dimension_count / 2;
                    rope.precompute_sincos_into(
                        position,
                        &mut sin_cache[..rope_half],
                        &mut cos_cache[..rope_half],
                    );
                    rope.apply_rotary_cached(
                        &mut q[..q_width],
                        layer_config.head_count,
                        layer_config.head_dim,
                        &sin_cache[..rope_half],
                        &cos_cache[..rope_half],
                    );
                    rope.apply_rotary_cached(
                        &mut k[..kv_width],
                        layer_config.kv_head_count,
                        layer_config.head_dim,
                        &sin_cache[..rope_half],
                        &cos_cache[..rope_half],
                    );
                    trace_stage!("q_rope", &q[..q_width]);
                    trace_stage!("k_rope", &k[..kv_width]);

                    cache.append(layer_index, &k[..cache_width], &v[..cache_width])?;
                    self.compute_attention_gemma4(
                        &q[..q_width],
                        cache,
                        layer_index,
                        layer_config,
                        &mut attn_out[..q_width],
                    )?;
                    trace_stage!("attention", &attn_out[..q_width]);

                    self.linear_resolved(attn_output, &attn_out[..q_width], proj)?;
                    trace_stage!("attention_projection", proj);
                    let attn_post_norm_w = self.load_vector(attn_post_norm)?;
                    apply_rmsnorm(proj, &attn_post_norm_w, eps, normed);
                    trace_stage!("post_attention_norm", normed);
                    add_inplace(&mut x, normed);
                    trace_stage!("post_attention", &x);
                }
                _ => {
                    return Err(XrtError::Runtime(format!(
                        "Gemma4 layer {layer_index} has non-Gemma4 attention weights"
                    )));
                }
            }

            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            apply_rmsnorm(&x, &ffn_norm_weight, eps, normed);
            trace_stage!("ffn_norm", normed);
            match &layer.ffn {
                FfnWeights::Gemma4Dense {
                    gate: ffn_gate,
                    down: ffn_down,
                    up: ffn_up,
                    post_ffw_norm,
                    layer_output_scale,
                } => {
                    let ff_dim = ffn_gate.rows;
                    self.linear_resolved(ffn_gate, normed, &mut gate[..ff_dim])?;
                    self.linear_resolved(ffn_up, normed, &mut up[..ff_dim])?;
                    trace_stage!("ffn_gate", &gate[..ff_dim]);
                    trace_stage!("ffn_up", &up[..ff_dim]);
                    geglu_pytorch_tanh(&mut gate[..ff_dim], &up[..ff_dim]);
                    trace_stage!("ffn_hidden", &gate[..ff_dim]);
                    self.linear_resolved(ffn_down, &gate[..ff_dim], proj)?;
                    trace_stage!("ffn_down", proj);

                    let post_ffw_norm_w = self.load_vector(post_ffw_norm)?;
                    apply_rmsnorm(proj, &post_ffw_norm_w, eps, normed);
                    trace_stage!("post_ffw_norm", normed);
                    add_inplace(&mut x, normed);

                    if let Some(scale_name) = layer_output_scale {
                        let scale = self.load_vector(scale_name)?;
                        let scale_value = scale.first().copied().unwrap_or(1.0);
                        for value in &mut x {
                            *value *= scale_value;
                        }
                    }
                    trace_stage!("output", &x);
                }
                _ => {
                    return Err(XrtError::Runtime(format!(
                        "Gemma4 layer {layer_index} has non-Gemma4 FFN weights"
                    )));
                }
            }

            down[..dim].copy_from_slice(&x);
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        apply_rmsnorm(&x, &output_norm_weight, eps, normed);
        trace_stage!("final_norm", normed);
        self.linear_resolved(&self.output, normed, output_logits)?;

        if let Some(softcap) = gemma4.final_logit_softcapping {
            for value in output_logits.iter_mut() {
                *value = (*value / softcap).tanh() * softcap;
            }
        }
        trace_stage!("logits", output_logits);

        Ok(())
    }

    /// Compute attention output from Q and KV cache using online softmax.
    fn compute_attention<C: KvCache + Sync>(
        &self,
        q: &[f32],
        cache: &C,
        layer_index: usize,
        head_dim: usize,
        attn_out: &mut [f32],
    ) -> Result<()> {
        let seq_len = cache.len(layer_index);
        let n_kv_heads = self.config.attention_head_count_kv;
        let head_group = self.config.attention_head_count / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        attn_out.fill(0.0);

        let q_ref: &[f32] = q;
        let attn_out_ptr = SendPtr::new(attn_out.as_mut_ptr());

        global_pool().par_for(n_kv_heads, |kv_start, kv_end| {
            for kv_head in kv_start..kv_end {
                let q_start = kv_head * head_group;
                let q_end = q_start + head_group;

                for head in q_start..q_end {
                    let q_head = &q_ref[head * head_dim..(head + 1) * head_dim];
                    let out_offset = head * head_dim;
                    let out_head = unsafe {
                        std::slice::from_raw_parts_mut(
                            (attn_out_ptr.0 as *mut f32).add(out_offset),
                            head_dim,
                        )
                    };

                    let mut max_score = f32::NEG_INFINITY;
                    let mut sum_exp = 0.0f32;
                    let mut key_row_buf = vec![0.0f32; cache.width()];
                    let mut value_row_buf = vec![0.0f32; cache.width()];

                    for position_idx in 0..seq_len {
                        cache
                            .copy_key_into(layer_index, position_idx, &mut key_row_buf)
                            .expect("missing key cache entry");
                        let key_head = &key_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                        let score = dot(q_head, key_head) * scale;

                        if score > max_score {
                            let correction = (max_score - score).exp();
                            sum_exp *= correction;
                            for d in 0..head_dim {
                                out_head[d] *= correction;
                            }
                            max_score = score;
                        }

                        let weight = (score - max_score).exp();
                        sum_exp += weight;

                        cache
                            .copy_value_into(layer_index, position_idx, &mut value_row_buf)
                            .expect("missing value cache entry");
                        let value_head =
                            &value_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                        accumulate_scaled(out_head, value_head, weight);
                    }

                    if sum_exp > 0.0 {
                        let inv_sum = sum_exp.recip();
                        for d in 0..head_dim {
                            out_head[d] *= inv_sum;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Gemma4 attention uses per-layer dimensions, optional sliding-window masks,
    /// and an attention scale of 1.0 rather than the usual 1/sqrt(head_dim).
    fn compute_attention_gemma4<C: KvCache + Sync>(
        &self,
        q: &[f32],
        cache: &C,
        layer_index: usize,
        layer_config: &Gemma4LayerConfig,
        attn_out: &mut [f32],
    ) -> Result<()> {
        let seq_len = cache.len(layer_index);
        let n_kv_heads = layer_config.kv_head_count;
        let head_group = layer_config.head_count / n_kv_heads;
        let head_dim = layer_config.head_dim;
        let attend_start = layer_config
            .sliding_window
            .map(|window| seq_len.saturating_sub(window))
            .unwrap_or(0);

        attn_out.fill(0.0);

        let q_ref: &[f32] = q;
        let attn_out_ptr = SendPtr::new(attn_out.as_mut_ptr());

        global_pool().par_for(n_kv_heads, |kv_start, kv_end| {
            for kv_head in kv_start..kv_end {
                let q_start = kv_head * head_group;
                let q_end = q_start + head_group;

                for head in q_start..q_end {
                    let q_head = &q_ref[head * head_dim..(head + 1) * head_dim];
                    let out_offset = head * head_dim;
                    let out_head = unsafe {
                        std::slice::from_raw_parts_mut(
                            (attn_out_ptr.0 as *mut f32).add(out_offset),
                            head_dim,
                        )
                    };

                    let mut max_score = f32::NEG_INFINITY;
                    let mut sum_exp = 0.0f32;
                    let mut key_row_buf = vec![0.0f32; cache.width()];
                    let mut value_row_buf = vec![0.0f32; cache.width()];

                    for position_idx in attend_start..seq_len {
                        cache
                            .copy_key_into(layer_index, position_idx, &mut key_row_buf)
                            .expect("missing key cache entry");
                        let key_head = &key_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                        let score = dot(q_head, key_head);

                        if score > max_score {
                            let correction = (max_score - score).exp();
                            sum_exp *= correction;
                            for d in 0..head_dim {
                                out_head[d] *= correction;
                            }
                            max_score = score;
                        }

                        let weight = (score - max_score).exp();
                        sum_exp += weight;

                        cache
                            .copy_value_into(layer_index, position_idx, &mut value_row_buf)
                            .expect("missing value cache entry");
                        let value_head =
                            &value_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                        accumulate_scaled(out_head, value_head, weight);
                    }

                    if sum_exp > 0.0 {
                        let inv_sum = sum_exp.recip();
                        for d in 0..head_dim {
                            out_head[d] *= inv_sum;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    pub fn forward_batch<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        cache: &mut C,
    ) -> Result<Vec<f32>> {
        self.forward_batch_inner(token_ids, start_position, None, cache, None)
    }

    pub fn forward_batch_with_state<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
    ) -> Result<Vec<f32>> {
        self.forward_batch_inner(token_ids, start_position, recurrent_state, cache, None)
    }

    /// Like `forward_batch`, but with optional per-position embedding overrides.
    /// Used for multimodal models where image patch embeddings replace certain token positions.
    pub fn forward_batch_with_embeddings<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        cache: &mut C,
        embedding_overrides: std::collections::HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        self.forward_batch_inner(
            token_ids,
            start_position,
            None,
            cache,
            Some(embedding_overrides),
        )
    }

    pub fn forward_batch_with_embeddings_and_state<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
        embedding_overrides: std::collections::HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        self.forward_batch_inner(
            token_ids,
            start_position,
            recurrent_state,
            cache,
            Some(embedding_overrides),
        )
    }

    fn forward_batch_inner<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        mut recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
        embedding_overrides: Option<std::collections::HashMap<usize, Vec<f32>>>,
    ) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        // For single token, delegate to forward_token
        if seq_len == 1 {
            let mut logits = vec![0.0; self.config.vocab_size];
            self.forward_token_with_state(
                token_ids[0],
                start_position,
                recurrent_state,
                cache,
                &mut logits,
            )?;
            return Ok(logits);
        }

        // Hybrid and Gemma4 models use layer-specific state/widths, so process sequentially first.
        if self.config.is_hybrid() || self.config.is_gemma4() {
            let mut logits = vec![0.0; self.config.vocab_size];
            for (i, &token_id) in token_ids.iter().enumerate() {
                self.forward_token_with_state(
                    token_id,
                    start_position + i,
                    recurrent_state.as_deref_mut(),
                    cache,
                    &mut logits,
                )?;
            }
            return Ok(logits);
        }

        if cache.layers() < self.config.block_count {
            return Err(XrtError::Model(format!(
                "KV cache has {} layers, but model requires {}",
                cache.layers(),
                self.config.block_count
            )));
        }
        if cache.width() != self.config.kv_width() {
            return Err(XrtError::Model(format!(
                "KV cache width {} does not match model width {}",
                cache.width(),
                self.config.kv_width()
            )));
        }

        let dim = self.config.embedding_length;
        let q_width = self.config.q_width();
        let kv_width = self.config.kv_width();
        let head_dim = self.config.head_dim();
        let ff_dim = self.config.feed_forward_length;
        let n_heads = self.config.attention_head_count;
        let n_kv_heads = self.config.attention_head_count_kv;
        let head_group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;

        // Acquire pooled scratch buffers (XenoMind FieldPool pattern).
        // Take ownership to avoid holding the write lock across &self borrows.
        let mut batch = std::mem::replace(&mut *self.batch_scratch.write(), BatchScratch::new());
        batch.ensure_capacity(seq_len, &self.config);

        // Step 1: Batch embedding lookup
        // Supports optional embedding overrides for multimodal (vision) inputs.
        // Positions where embedding_overrides contains data skip the token lookup.
        for (t, &token_id) in token_ids.iter().enumerate() {
            if let Some(ref overrides) = embedding_overrides {
                if let Some(emb) = overrides.get(&t) {
                    batch.xs[t * dim..(t + 1) * dim].copy_from_slice(emb);
                    continue;
                }
            }
            let emb = self.embedding_lookup(token_id as usize)?;
            batch.xs[t * dim..(t + 1) * dim].copy_from_slice(&emb);
        }

        // RoPE sin/cos scratch (reused across positions)
        let rope_half = self.config.rope_dimension_count / 2;
        let mut sin_buf = vec![0.0f32; rope_half];
        let mut cos_buf = vec![0.0f32; rope_half];

        // Step 2: Layer loop
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if cache.len(layer_index) != start_position {
                return Err(XrtError::Runtime(format!(
                    "KV cache length mismatch at layer {layer_index}: expected {start_position}, found {}",
                    cache.len(layer_index)
                )));
            }

            // 2a: RMSNorm each token's hidden state
            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &attn_norm_weight, eps, normed_t);
            }

            // 2b: Batch QKV projections (read weight matrix ONCE for all tokens)
            // This code path only runs for Standard attention (hybrid models use sequential above)
            let (
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
            ) = match &layer.attn {
                AttnWeights::Standard {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    attn_q_bias,
                    attn_k_bias,
                    attn_v_bias,
                } => (
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm.as_ref(),
                    attn_k_norm.as_ref(),
                    attn_q_bias.as_ref(),
                    attn_k_bias.as_ref(),
                    attn_v_bias.as_ref(),
                ),
                _ => unreachable!(
                    "batch forward only handles Standard attention (hybrid uses sequential)"
                ),
            };

            self.linear_batch_resolved(
                attn_q,
                &batch.normed[..seq_len * dim],
                seq_len,
                &mut batch.q[..seq_len * q_width],
            )?;
            self.linear_batch_resolved(
                attn_k,
                &batch.normed[..seq_len * dim],
                seq_len,
                &mut batch.k[..seq_len * kv_width],
            )?;
            self.linear_batch_resolved(
                attn_v,
                &batch.normed[..seq_len * dim],
                seq_len,
                &mut batch.v[..seq_len * kv_width],
            )?;

            if let Some(q_bias_name) = attn_q_bias {
                let q_bias = self.load_vector(q_bias_name)?;
                Self::add_batch_bias(&mut batch.q[..seq_len * q_width], seq_len, q_width, &q_bias)?;
            }
            if let Some(k_bias_name) = attn_k_bias {
                let k_bias = self.load_vector(k_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.k[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &k_bias,
                )?;
            }
            if let Some(v_bias_name) = attn_v_bias {
                let v_bias = self.load_vector(v_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.v[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &v_bias,
                )?;
            }

            // 2c: Optional Qwen3 QK head normalization
            if let Some(q_norm_name) = attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(k_norm_name) = attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                for t in 0..seq_len {
                    let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                    self.apply_head_norm(k_t, n_kv_heads, head_dim, &k_norm_w);
                }
            }

            // 2d: RoPE for each token at its position (zero-alloc sin/cos)
            for t in 0..seq_len {
                let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                self.rope_freqs.precompute_sincos_into(
                    start_position + t,
                    &mut sin_buf,
                    &mut cos_buf,
                );
                self.rope_freqs
                    .apply_rotary_cached(q_t, n_heads, head_dim, &sin_buf, &cos_buf);
                self.rope_freqs
                    .apply_rotary_cached(k_t, n_kv_heads, head_dim, &sin_buf, &cos_buf);
            }

            // 2e: Batch KV cache append
            cache.append_batch(
                layer_index,
                &batch.k[..seq_len * kv_width],
                &batch.v[..seq_len * kv_width],
                seq_len,
            )?;

            // 2f: Flash attention with causal mask (online softmax, no scores buffer)
            batch.attn_out[..seq_len * q_width].fill(0.0);
            let q_sl = &batch.q[..seq_len * q_width];
            let attn_out_ptr = SendPtr::new(batch.attn_out.as_mut_ptr());

            global_pool().par_for(n_kv_heads, |kv_start, kv_end| {
                for kv_head in kv_start..kv_end {
                    let q_start_h = kv_head * head_group;
                    let q_end_h = q_start_h + head_group;
                    for head in q_start_h..q_end_h {
                        for t in 0..seq_len {
                            let attend_len = start_position + t + 1;
                            let q_head = &q_sl[t * q_width + head * head_dim
                                ..t * q_width + (head + 1) * head_dim];
                            let out_head = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (attn_out_ptr.0 as *mut f32).add(t * q_width + head * head_dim),
                                    head_dim,
                                )
                            };
                            let mut max_score = f32::NEG_INFINITY;
                            let mut sum_exp = 0.0f32;
                            let mut key_row_buf = vec![0.0f32; cache.width()];
                            let mut value_row_buf = vec![0.0f32; cache.width()];
                            for pos in 0..attend_len {
                                cache
                                    .copy_key_into(layer_index, pos, &mut key_row_buf)
                                    .expect("missing key");
                                let key_head =
                                    &key_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                                let score = dot(q_head, key_head) * scale;
                                if score > max_score {
                                    let correction = (max_score - score).exp();
                                    sum_exp *= correction;
                                    for d in out_head.iter_mut() {
                                        *d *= correction;
                                    }
                                    max_score = score;
                                }
                                let weight = (score - max_score).exp();
                                sum_exp += weight;
                                cache
                                    .copy_value_into(layer_index, pos, &mut value_row_buf)
                                    .expect("missing value");
                                let value_head =
                                    &value_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                                accumulate_scaled(out_head, value_head, weight);
                            }
                            if sum_exp > 0.0 {
                                let inv = sum_exp.recip();
                                for d in out_head.iter_mut() {
                                    *d *= inv;
                                }
                            }
                        }
                    }
                }
            });

            // 2g: Batch attention output projection
            self.linear_batch_resolved(
                attn_output,
                &batch.attn_out[..seq_len * q_width],
                seq_len,
                &mut batch.proj[..seq_len * dim],
            )?;

            // 2h: Residual add
            let xs_len = seq_len * dim;
            for i in 0..xs_len {
                batch.xs[i] += batch.proj[i];
            }

            // 2i: FFN norm
            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &ffn_norm_weight, eps, normed_t);
            }

            // 2j: Batch FFN (gate, up, swiglu, down)
            match &layer.ffn {
                FfnWeights::Dense {
                    gate: ffn_gate,
                    down: ffn_down,
                    up: ffn_up,
                } => {
                    self.linear_batch_resolved(
                        ffn_gate,
                        &batch.normed[..seq_len * dim],
                        seq_len,
                        &mut batch.gate[..seq_len * ff_dim],
                    )?;
                    self.linear_batch_resolved(
                        ffn_up,
                        &batch.normed[..seq_len * dim],
                        seq_len,
                        &mut batch.up[..seq_len * ff_dim],
                    )?;
                    for t in 0..seq_len {
                        swiglu(
                            &mut batch.gate[t * ff_dim..(t + 1) * ff_dim],
                            &batch.up[t * ff_dim..(t + 1) * ff_dim],
                        );
                    }
                    self.linear_batch_resolved(
                        ffn_down,
                        &batch.gate[..seq_len * ff_dim],
                        seq_len,
                        &mut batch.down[..seq_len * dim],
                    )?;
                }
                FfnWeights::Moe {
                    descriptor,
                    router,
                    experts,
                    shared,
                } => {
                    self.execute_moe_batch_configured(
                        descriptor, router, experts, seq_len, &mut batch,
                    )?;
                    if let Some(shared) = shared {
                        self.execute_shared_moe_batch(shared, seq_len, &mut batch)?;
                    }
                }
                FfnWeights::Gemma4Dense { .. } => {
                    unreachable!("Gemma4 batch prefill falls back to token-by-token execution")
                }
            }

            // 2k: Residual add
            for i in 0..xs_len {
                batch.xs[i] += batch.down[i];
            }
        }

        // Step 3: Output projection on LAST token only
        let last_x = &batch.xs[(seq_len - 1) * dim..seq_len * dim];
        let output_norm_weight = self.load_vector(&self.output_norm)?;
        let mut normed_last = vec![0.0f32; dim];
        apply_rmsnorm(last_x, &output_norm_weight, eps, &mut normed_last);
        let mut logits = vec![0.0f32; self.output.rows];
        self.linear_resolved(&self.output, &normed_last, &mut logits)?;

        // Return pooled buffers for reuse
        *self.batch_scratch.write() = batch;
        Ok(logits)
    }

    /// Like `forward_batch`, but returns logits for ALL positions (not just the last).
    /// Used for speculative decoding verification: the caller can check each position's
    /// predicted next token against the draft sequence.
    /// Returns a flat Vec of `seq_len * vocab_size` floats.
    pub fn forward_batch_all_logits<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        cache: &mut C,
    ) -> Result<Vec<f32>> {
        self.forward_batch_all_logits_with_state(token_ids, start_position, None, cache)
    }

    pub fn forward_batch_all_logits_with_state<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        mut recurrent_state: Option<&mut DeltaNetState>,
        cache: &mut C,
    ) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        if seq_len == 1 {
            let mut logits = vec![0.0; self.config.vocab_size];
            self.forward_token_with_state(
                token_ids[0],
                start_position,
                recurrent_state,
                cache,
                &mut logits,
            )?;
            return Ok(logits);
        }

        // Hybrid and Gemma4 models use layer-specific state/widths, so process sequentially first.
        if self.config.is_hybrid() || self.config.is_gemma4() {
            let vocab_size = self.config.vocab_size;
            let mut all_logits = vec![0.0f32; seq_len * vocab_size];
            let mut logits = vec![0.0; vocab_size];
            for (i, &token_id) in token_ids.iter().enumerate() {
                self.forward_token_with_state(
                    token_id,
                    start_position + i,
                    recurrent_state.as_deref_mut(),
                    cache,
                    &mut logits,
                )?;
                all_logits[i * vocab_size..(i + 1) * vocab_size].copy_from_slice(&logits);
            }
            return Ok(all_logits);
        }

        if cache.layers() < self.config.block_count {
            return Err(XrtError::Model(format!(
                "KV cache has {} layers, but model requires {}",
                cache.layers(),
                self.config.block_count
            )));
        }

        let dim = self.config.embedding_length;
        let q_width = self.config.q_width();
        let kv_width = self.config.kv_width();
        let head_dim = self.config.head_dim();
        let ff_dim = self.config.feed_forward_length;
        let n_heads = self.config.attention_head_count;
        let n_kv_heads = self.config.attention_head_count_kv;
        let head_group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let vocab_size = self.config.vocab_size;

        let mut batch = std::mem::replace(&mut *self.batch_scratch.write(), BatchScratch::new());
        batch.ensure_capacity(seq_len, &self.config);

        for (t, &token_id) in token_ids.iter().enumerate() {
            let emb = self.embedding_lookup(token_id as usize)?;
            batch.xs[t * dim..(t + 1) * dim].copy_from_slice(&emb);
        }

        let rope_half = self.config.rope_dimension_count / 2;
        let mut sin_buf = vec![0.0f32; rope_half];
        let mut cos_buf = vec![0.0f32; rope_half];

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if cache.len(layer_index) != start_position {
                return Err(XrtError::Runtime(format!(
                    "KV cache length mismatch at layer {layer_index}: expected {start_position}, found {}",
                    cache.len(layer_index)
                )));
            }

            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &attn_norm_weight, eps, normed_t);
            }

            let (
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
            ) = match &layer.attn {
                AttnWeights::Standard {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    attn_q_bias,
                    attn_k_bias,
                    attn_v_bias,
                } => (
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm.as_ref(),
                    attn_k_norm.as_ref(),
                    attn_q_bias.as_ref(),
                    attn_k_bias.as_ref(),
                    attn_v_bias.as_ref(),
                ),
                _ => unreachable!("batch all_logits only handles Standard attention"),
            };

            let normed_sl = &batch.normed[..seq_len * dim];
            self.linear_batch_resolved(
                attn_q,
                normed_sl,
                seq_len,
                &mut batch.q[..seq_len * q_width],
            )?;
            self.linear_batch_resolved(
                attn_k,
                normed_sl,
                seq_len,
                &mut batch.k[..seq_len * kv_width],
            )?;
            self.linear_batch_resolved(
                attn_v,
                normed_sl,
                seq_len,
                &mut batch.v[..seq_len * kv_width],
            )?;

            if let Some(q_bias_name) = attn_q_bias {
                let q_bias = self.load_vector(q_bias_name)?;
                Self::add_batch_bias(&mut batch.q[..seq_len * q_width], seq_len, q_width, &q_bias)?;
            }
            if let Some(k_bias_name) = attn_k_bias {
                let k_bias = self.load_vector(k_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.k[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &k_bias,
                )?;
            }
            if let Some(v_bias_name) = attn_v_bias {
                let v_bias = self.load_vector(v_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.v[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &v_bias,
                )?;
            }

            if let Some(q_norm_name) = attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(k_norm_name) = attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                for t in 0..seq_len {
                    let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                    self.apply_head_norm(k_t, n_kv_heads, head_dim, &k_norm_w);
                }
            }

            for t in 0..seq_len {
                let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                self.rope_freqs.precompute_sincos_into(
                    start_position + t,
                    &mut sin_buf,
                    &mut cos_buf,
                );
                self.rope_freqs
                    .apply_rotary_cached(q_t, n_heads, head_dim, &sin_buf, &cos_buf);
                self.rope_freqs
                    .apply_rotary_cached(k_t, n_kv_heads, head_dim, &sin_buf, &cos_buf);
            }

            cache.append_batch(
                layer_index,
                &batch.k[..seq_len * kv_width],
                &batch.v[..seq_len * kv_width],
                seq_len,
            )?;

            // Flash attention (online softmax, no scores buffer, parallel across KV heads)
            batch.attn_out[..seq_len * q_width].fill(0.0);
            {
                let q_sl = &batch.q[..seq_len * q_width];
                let attn_out_ptr = SendPtr::new(batch.attn_out.as_mut_ptr());
                global_pool().par_for(n_kv_heads, |kv_start, kv_end| {
                    for kv_head in kv_start..kv_end {
                        let q_start_h = kv_head * head_group;
                        let q_end_h = q_start_h + head_group;
                        for head in q_start_h..q_end_h {
                            for t in 0..seq_len {
                                let attend_len = start_position + t + 1;
                                let q_head = &q_sl[t * q_width + head * head_dim
                                    ..t * q_width + (head + 1) * head_dim];
                                let out_head = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        (attn_out_ptr.0 as *mut f32)
                                            .add(t * q_width + head * head_dim),
                                        head_dim,
                                    )
                                };
                                let mut max_score = f32::NEG_INFINITY;
                                let mut sum_exp = 0.0f32;
                                let mut key_row_buf = vec![0.0f32; cache.width()];
                                let mut value_row_buf = vec![0.0f32; cache.width()];
                                for pos in 0..attend_len {
                                    cache
                                        .copy_key_into(layer_index, pos, &mut key_row_buf)
                                        .expect("missing key");
                                    let key_head =
                                        &key_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                                    let score = dot(q_head, key_head) * scale;
                                    if score > max_score {
                                        let correction = (max_score - score).exp();
                                        sum_exp *= correction;
                                        for d in out_head.iter_mut() {
                                            *d *= correction;
                                        }
                                        max_score = score;
                                    }
                                    let weight = (score - max_score).exp();
                                    sum_exp += weight;
                                    cache
                                        .copy_value_into(layer_index, pos, &mut value_row_buf)
                                        .expect("missing value");
                                    let value_head = &value_row_buf
                                        [kv_head * head_dim..(kv_head + 1) * head_dim];
                                    accumulate_scaled(out_head, value_head, weight);
                                }
                                if sum_exp > 0.0 {
                                    let inv = sum_exp.recip();
                                    for d in out_head.iter_mut() {
                                        *d *= inv;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            self.linear_batch_resolved(
                attn_output,
                &batch.attn_out[..seq_len * q_width],
                seq_len,
                &mut batch.proj[..seq_len * dim],
            )?;
            let xs_len = seq_len * dim;
            for i in 0..xs_len {
                batch.xs[i] += batch.proj[i];
            }

            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &ffn_norm_weight, eps, normed_t);
            }

            let normed_sl = &batch.normed[..seq_len * dim];
            match &layer.ffn {
                FfnWeights::Dense {
                    gate: ffn_gate,
                    down: ffn_down,
                    up: ffn_up,
                } => {
                    self.linear_batch_resolved(
                        ffn_gate,
                        normed_sl,
                        seq_len,
                        &mut batch.gate[..seq_len * ff_dim],
                    )?;
                    self.linear_batch_resolved(
                        ffn_up,
                        normed_sl,
                        seq_len,
                        &mut batch.up[..seq_len * ff_dim],
                    )?;
                    for t in 0..seq_len {
                        swiglu(
                            &mut batch.gate[t * ff_dim..(t + 1) * ff_dim],
                            &batch.up[t * ff_dim..(t + 1) * ff_dim],
                        );
                    }
                    self.linear_batch_resolved(
                        ffn_down,
                        &batch.gate[..seq_len * ff_dim],
                        seq_len,
                        &mut batch.down[..seq_len * dim],
                    )?;
                }
                FfnWeights::Moe {
                    descriptor,
                    router,
                    experts,
                    shared,
                } => {
                    self.execute_moe_batch_configured(
                        descriptor, router, experts, seq_len, &mut batch,
                    )?;
                    if let Some(shared) = shared {
                        self.execute_shared_moe_batch(shared, seq_len, &mut batch)?;
                    }
                }
                FfnWeights::Gemma4Dense { .. } => {
                    unreachable!("Gemma4 all-logits prefill falls back to token-by-token execution")
                }
            }
            for i in 0..xs_len {
                batch.xs[i] += batch.down[i];
            }
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        let mut all_logits = vec![0.0f32; seq_len * vocab_size];
        let mut normed_buf = vec![0.0f32; dim];

        for t in 0..seq_len {
            let x_t = &batch.xs[t * dim..(t + 1) * dim];
            apply_rmsnorm(x_t, &output_norm_weight, eps, &mut normed_buf);
            let logits_t = &mut all_logits[t * vocab_size..(t + 1) * vocab_size];
            self.linear_resolved(&self.output, &normed_buf, logits_t)?;
        }

        *self.batch_scratch.write() = batch;
        Ok(all_logits)
    }

    /// Batch prefill through only the first `n_layers` layers to populate a draft KV cache.
    /// No output logits are computed — this only builds the KV cache entries.
    /// Used to warm-start the draft cache for self-speculative decoding.
    pub fn forward_batch_draft<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        n_layers: usize,
        cache: &mut C,
    ) -> Result<()> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Ok(());
        }
        if self.config.is_hybrid() {
            return Err(XrtError::Unsupported(
                "partial-layer drafting is not supported for hybrid recurrent models".to_string(),
            ));
        }
        if cache.layers() < n_layers {
            return Err(XrtError::Model(format!(
                "KV cache has {} layers, but draft requires {}",
                cache.layers(),
                n_layers
            )));
        }
        if self.config.is_gemma4() {
            let mut logits = vec![0.0; self.config.vocab_size];
            for (i, &token_id) in token_ids.iter().enumerate() {
                self.forward_draft(token_id, start_position + i, n_layers, cache, &mut logits)?;
            }
            return Ok(());
        }
        if cache.width() != self.config.kv_width() {
            return Err(XrtError::Model(format!(
                "KV cache width {} does not match model width {}",
                cache.width(),
                self.config.kv_width()
            )));
        }

        let dim = self.config.embedding_length;
        let q_width = self.config.q_width();
        let kv_width = self.config.kv_width();
        let head_dim = self.config.head_dim();
        let ff_dim = self.config.feed_forward_length;
        let n_heads = self.config.attention_head_count;
        let n_kv_heads = self.config.attention_head_count_kv;
        let head_group = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;

        let mut batch = std::mem::replace(&mut *self.batch_scratch.write(), BatchScratch::new());
        batch.ensure_capacity(seq_len, &self.config);

        for (t, &token_id) in token_ids.iter().enumerate() {
            let emb = self.embedding_lookup(token_id as usize)?;
            batch.xs[t * dim..(t + 1) * dim].copy_from_slice(&emb);
        }

        let rope_half = self.config.rope_dimension_count / 2;
        let mut sin_buf = vec![0.0f32; rope_half];
        let mut cos_buf = vec![0.0f32; rope_half];

        // Only iterate through first n_layers
        for (layer_index, layer) in self.layers[..n_layers].iter().enumerate() {
            if cache.len(layer_index) != start_position {
                return Err(XrtError::Runtime(format!(
                    "KV cache length mismatch at layer {layer_index}: expected {start_position}, found {}",
                    cache.len(layer_index)
                )));
            }

            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &attn_norm_weight, eps, normed_t);
            }

            let (
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
            ) = match &layer.attn {
                AttnWeights::Standard {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    attn_q_bias,
                    attn_k_bias,
                    attn_v_bias,
                } => (
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm.as_ref(),
                    attn_k_norm.as_ref(),
                    attn_q_bias.as_ref(),
                    attn_k_bias.as_ref(),
                    attn_v_bias.as_ref(),
                ),
                _ => unreachable!("forward_batch_draft only handles Standard attention"),
            };

            let normed_sl = &batch.normed[..seq_len * dim];
            self.linear_batch_resolved(
                attn_q,
                normed_sl,
                seq_len,
                &mut batch.q[..seq_len * q_width],
            )?;
            self.linear_batch_resolved(
                attn_k,
                normed_sl,
                seq_len,
                &mut batch.k[..seq_len * kv_width],
            )?;
            self.linear_batch_resolved(
                attn_v,
                normed_sl,
                seq_len,
                &mut batch.v[..seq_len * kv_width],
            )?;

            if let Some(q_bias_name) = attn_q_bias {
                let q_bias = self.load_vector(q_bias_name)?;
                Self::add_batch_bias(&mut batch.q[..seq_len * q_width], seq_len, q_width, &q_bias)?;
            }
            if let Some(k_bias_name) = attn_k_bias {
                let k_bias = self.load_vector(k_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.k[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &k_bias,
                )?;
            }
            if let Some(v_bias_name) = attn_v_bias {
                let v_bias = self.load_vector(v_bias_name)?;
                Self::add_batch_bias(
                    &mut batch.v[..seq_len * kv_width],
                    seq_len,
                    kv_width,
                    &v_bias,
                )?;
            }

            if let Some(q_norm_name) = attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(k_norm_name) = attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                for t in 0..seq_len {
                    let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                    self.apply_head_norm(k_t, n_kv_heads, head_dim, &k_norm_w);
                }
            }

            for t in 0..seq_len {
                let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                self.rope_freqs.precompute_sincos_into(
                    start_position + t,
                    &mut sin_buf,
                    &mut cos_buf,
                );
                self.rope_freqs
                    .apply_rotary_cached(q_t, n_heads, head_dim, &sin_buf, &cos_buf);
                self.rope_freqs
                    .apply_rotary_cached(k_t, n_kv_heads, head_dim, &sin_buf, &cos_buf);
            }

            cache.append_batch(
                layer_index,
                &batch.k[..seq_len * kv_width],
                &batch.v[..seq_len * kv_width],
                seq_len,
            )?;

            // Flash attention (online softmax, no scores buffer, parallel across KV heads)
            batch.attn_out[..seq_len * q_width].fill(0.0);
            {
                let q_sl = &batch.q[..seq_len * q_width];
                let attn_out_ptr = SendPtr::new(batch.attn_out.as_mut_ptr());
                global_pool().par_for(n_kv_heads, |kv_start, kv_end| {
                    for kv_head in kv_start..kv_end {
                        let q_start_h = kv_head * head_group;
                        let q_end_h = q_start_h + head_group;
                        for head in q_start_h..q_end_h {
                            for t in 0..seq_len {
                                let attend_len = start_position + t + 1;
                                let q_head = &q_sl[t * q_width + head * head_dim
                                    ..t * q_width + (head + 1) * head_dim];
                                let out_head = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        (attn_out_ptr.0 as *mut f32)
                                            .add(t * q_width + head * head_dim),
                                        head_dim,
                                    )
                                };
                                let mut max_score = f32::NEG_INFINITY;
                                let mut sum_exp = 0.0f32;
                                let mut key_row_buf = vec![0.0f32; cache.width()];
                                let mut value_row_buf = vec![0.0f32; cache.width()];
                                for pos in 0..attend_len {
                                    cache
                                        .copy_key_into(layer_index, pos, &mut key_row_buf)
                                        .expect("missing key");
                                    let key_head =
                                        &key_row_buf[kv_head * head_dim..(kv_head + 1) * head_dim];
                                    let score = dot(q_head, key_head) * scale;
                                    if score > max_score {
                                        let correction = (max_score - score).exp();
                                        sum_exp *= correction;
                                        for d in out_head.iter_mut() {
                                            *d *= correction;
                                        }
                                        max_score = score;
                                    }
                                    let weight = (score - max_score).exp();
                                    sum_exp += weight;
                                    cache
                                        .copy_value_into(layer_index, pos, &mut value_row_buf)
                                        .expect("missing value");
                                    let value_head = &value_row_buf
                                        [kv_head * head_dim..(kv_head + 1) * head_dim];
                                    accumulate_scaled(out_head, value_head, weight);
                                }
                                if sum_exp > 0.0 {
                                    let inv = sum_exp.recip();
                                    for d in out_head.iter_mut() {
                                        *d *= inv;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            self.linear_batch_resolved(
                attn_output,
                &batch.attn_out[..seq_len * q_width],
                seq_len,
                &mut batch.proj[..seq_len * dim],
            )?;
            let xs_len = seq_len * dim;
            for i in 0..xs_len {
                batch.xs[i] += batch.proj[i];
            }

            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            for t in 0..seq_len {
                let x_t = &batch.xs[t * dim..(t + 1) * dim];
                let normed_t = &mut batch.normed[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &ffn_norm_weight, eps, normed_t);
            }
            let normed_sl = &batch.normed[..seq_len * dim];
            match &layer.ffn {
                FfnWeights::Dense {
                    gate: ffn_gate,
                    down: ffn_down,
                    up: ffn_up,
                } => {
                    self.linear_batch_resolved(
                        ffn_gate,
                        normed_sl,
                        seq_len,
                        &mut batch.gate[..seq_len * ff_dim],
                    )?;
                    self.linear_batch_resolved(
                        ffn_up,
                        normed_sl,
                        seq_len,
                        &mut batch.up[..seq_len * ff_dim],
                    )?;
                    for t in 0..seq_len {
                        swiglu(
                            &mut batch.gate[t * ff_dim..(t + 1) * ff_dim],
                            &batch.up[t * ff_dim..(t + 1) * ff_dim],
                        );
                    }
                    self.linear_batch_resolved(
                        ffn_down,
                        &batch.gate[..seq_len * ff_dim],
                        seq_len,
                        &mut batch.down[..seq_len * dim],
                    )?;
                }
                FfnWeights::Moe {
                    descriptor,
                    router,
                    experts,
                    shared,
                } => {
                    self.execute_moe_batch_configured(
                        descriptor, router, experts, seq_len, &mut batch,
                    )?;
                    if let Some(shared) = shared {
                        self.execute_shared_moe_batch(shared, seq_len, &mut batch)?;
                    }
                }
                FfnWeights::Gemma4Dense { .. } => {
                    unreachable!("Gemma4 draft prefill falls back to token-by-token execution")
                }
            }
            for i in 0..xs_len {
                batch.xs[i] += batch.down[i];
            }
        }

        // No output projection — we only needed to populate the KV cache
        *self.batch_scratch.write() = batch;
        Ok(())
    }

    /// Batch linear projection using pre-resolved weight (zero HashMap lookups).
    fn linear_batch_resolved(
        &self,
        w: &ResolvedWeight,
        inputs: &[f32],
        seq_len: usize,
        outputs: &mut [f32],
    ) -> Result<()> {
        if !w.dtype.is_quantized() {
            for token_index in 0..seq_len {
                let input = &inputs[token_index * w.cols..(token_index + 1) * w.cols];
                let output = &mut outputs[token_index * w.rows..(token_index + 1) * w.rows];
                self.linear_resolved(w, input, output)?;
            }
            return Ok(());
        }
        let bytes = self.gguf.tensor_data_raw(w.data_offset, w.nbytes);
        matvec_quantized_batch(bytes, w.rows, w.cols, w.dtype, inputs, seq_len, outputs)
    }

    fn add_bias(output: &mut [f32], bias: &[f32]) -> Result<()> {
        if output.len() != bias.len() {
            return Err(XrtError::Model(format!(
                "bias length mismatch: output has {}, bias has {}",
                output.len(),
                bias.len()
            )));
        }
        for (value, &bias_value) in output.iter_mut().zip(bias.iter()) {
            *value += bias_value;
        }
        Ok(())
    }

    fn add_batch_bias(
        output: &mut [f32],
        seq_len: usize,
        row_width: usize,
        bias: &[f32],
    ) -> Result<()> {
        if bias.len() != row_width {
            return Err(XrtError::Model(format!(
                "batch bias length mismatch: row width is {row_width}, bias has {}",
                bias.len()
            )));
        }
        let expected_len = seq_len * row_width;
        if output.len() != expected_len {
            return Err(XrtError::Model(format!(
                "batch output length mismatch: expected {expected_len}, found {}",
                output.len()
            )));
        }
        for row in output.chunks_exact_mut(row_width) {
            for (value, &bias_value) in row.iter_mut().zip(bias.iter()) {
                *value += bias_value;
            }
        }
        Ok(())
    }

    /// Apply RMSNorm independently to each head's slice.
    /// Used by Qwen3 for QK normalization with per-head-dim weight vectors.
    fn apply_head_norm(&self, tensor: &mut [f32], n_heads: usize, head_dim: usize, weight: &[f32]) {
        debug_assert_eq!(tensor.len(), n_heads * head_dim);
        debug_assert_eq!(weight.len(), head_dim);
        let eps = self.config.rms_norm_eps;
        for head in 0..n_heads {
            let head_slice = &mut tensor[head * head_dim..(head + 1) * head_dim];
            let mut sum_sq = 0.0f32;
            for &val in head_slice.iter() {
                sum_sq += val * val;
            }
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
            for (val, &w) in head_slice.iter_mut().zip(weight.iter()) {
                *val = *val * inv_rms * w;
            }
        }
    }

    fn apply_head_rmsnorm_unweighted(&self, tensor: &mut [f32], n_heads: usize, head_dim: usize) {
        debug_assert_eq!(tensor.len(), n_heads * head_dim);
        let eps = self.config.rms_norm_eps;
        for head in 0..n_heads {
            let head_slice = &mut tensor[head * head_dim..(head + 1) * head_dim];
            let mut sum_sq = 0.0f32;
            for &val in head_slice.iter() {
                sum_sq += val * val;
            }
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
            for val in head_slice.iter_mut() {
                *val *= inv_rms;
            }
        }
    }

    /// Look up the embedding vector for a token ID.
    pub fn embedding_lookup(&self, token_id: usize) -> Result<Vec<f32>> {
        let info = self.gguf.require_tensor(&self.token_embedding)?;
        if token_id >= info.rows() {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                info.rows()
            )));
        }

        let bytes = self.gguf.tensor_data(&self.token_embedding)?;
        let mut output = vec![0.0f32; info.row_len()];
        self.decode_row_into(info, bytes, token_id, &mut output)?;
        Ok(output)
    }

    /// Linear projection using pre-resolved weight metadata (zero HashMap lookups).
    /// If a LoRA adapter is loaded and has weights for this tensor, applies the delta.
    fn linear_resolved(&self, w: &ResolvedWeight, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), w.cols);
        debug_assert_eq!(output.len(), w.rows);
        let bytes = self.gguf.tensor_data_raw(w.data_offset, w.nbytes);
        match w.dtype {
            DType::F32 => {
                for (row, output_value) in output.iter_mut().enumerate().take(w.rows) {
                    let row_start = row * w.cols * 4;
                    let mut sum = 0.0f32;
                    for (col, input_value) in input.iter().enumerate().take(w.cols) {
                        let offset = row_start + col * 4;
                        let weight = f32::from_le_bytes([
                            bytes[offset],
                            bytes[offset + 1],
                            bytes[offset + 2],
                            bytes[offset + 3],
                        ]);
                        sum += weight * input_value;
                    }
                    *output_value = sum;
                }
            }
            DType::F16 => {
                for (row, output_value) in output.iter_mut().enumerate().take(w.rows) {
                    let row_start = row * w.cols * 2;
                    let mut sum = 0.0f32;
                    for (col, input_value) in input.iter().enumerate().take(w.cols) {
                        let offset = row_start + col * 2;
                        let weight = decode_f16(&bytes[offset..offset + 2])?;
                        sum += weight * input_value;
                    }
                    *output_value = sum;
                }
            }
            DType::BF16 => {
                for (row, output_value) in output.iter_mut().enumerate().take(w.rows) {
                    let row_start = row * w.cols * 2;
                    let mut sum = 0.0f32;
                    for (col, input_value) in input.iter().enumerate().take(w.cols) {
                        let offset = row_start + col * 2;
                        let weight = decode_bf16(&bytes[offset..offset + 2])?;
                        sum += weight * input_value;
                    }
                    *output_value = sum;
                }
            }
            _ => matvec_quantized(bytes, w.rows, w.cols, w.dtype, input, output)?,
        }
        // Apply LoRA delta if adapter is loaded and has weights for this tensor
        if let Some(ref lora) = self.lora {
            if lora.has_weight(&w.name) {
                lora.apply(&w.name, input, output)?;
            }
        }
        Ok(())
    }

    fn linear_pair_resolved(
        &self,
        first: &ResolvedWeight,
        second: &ResolvedWeight,
        input: &[f32],
        first_output: &mut [f32],
        second_output: &mut [f32],
    ) -> Result<()> {
        let can_fuse = first.dtype == second.dtype
            && first.cols == second.cols
            && first.dtype.is_quantized()
            && input.len() == first.cols
            && first_output.len() == first.rows
            && second_output.len() == second.rows;
        if !can_fuse {
            self.linear_resolved(first, input, first_output)?;
            return self.linear_resolved(second, input, second_output);
        }

        let first_bytes = self.gguf.tensor_data_raw(first.data_offset, first.nbytes);
        let second_bytes = self.gguf.tensor_data_raw(second.data_offset, second.nbytes);
        matvec_quantized_fused(
            &[first_bytes, second_bytes],
            &[first.rows, second.rows],
            first.cols,
            first.dtype,
            input,
            &mut [first_output, second_output],
        )?;
        if let Some(lora) = &self.lora {
            if lora.has_weight(&first.name) {
                lora.apply(&first.name, input, first_output)?;
            }
            if lora.has_weight(&second.name) {
                lora.apply(&second.name, input, second_output)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_moe_token(
        &self,
        descriptor: &MoeLayerDescriptor,
        router: &ResolvedWeight,
        experts: &[MoeExpertWeights],
        input: &[f32],
        router_logits: &mut [f32],
        gate: &mut [f32],
        up: &mut [f32],
        expert_out: &mut [f32],
        output: &mut [f32],
    ) -> Result<MoeRoutingRow> {
        if experts.len() != descriptor.expert_count()
            || router_logits.len() != descriptor.expert_count()
            || input.len() != descriptor.hidden_size()
            || output.len() != descriptor.hidden_size()
            || expert_out.len() != descriptor.hidden_size()
            || gate.len() != descriptor.intermediate_size()
            || up.len() != descriptor.intermediate_size()
        {
            return Err(XrtError::Runtime(format!(
                "MoE layer {} execution scratch does not match its validated descriptor",
                descriptor.layer_index()
            )));
        }

        self.linear_resolved(router, input, router_logits)?;
        let mut route = MoeRoutingRow::default();
        route_top_k(router_logits, descriptor.selected_per_token(), &mut route)?;
        self.moe_telemetry.record_route(&route);
        #[cfg(feature = "moe-route-trace")]
        self.moe_telemetry
            .record_route_trace(descriptor.layer_index(), &route);

        output.fill(0.0);
        for (expert_id, weight) in route.iter() {
            let expert = experts.get(expert_id).ok_or_else(|| {
                XrtError::Runtime(format!(
                    "MoE layer {} selected missing logical expert {expert_id}",
                    descriptor.layer_index()
                ))
            })?;
            self.linear_pair_resolved(&expert.gate, &expert.up, input, gate, up)?;
            swiglu(gate, up);
            self.linear_resolved(&expert.down, gate, expert_out)?;
            accumulate_scaled(output, expert_out, weight);
        }
        Ok(route)
    }

    fn execute_shared_moe_token(
        &self,
        shared: &MoeSharedExpertWeights,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        expert_out: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        if input.len() != shared.gate_selector.cols
            || gate.len() != shared.intermediate_size
            || up.len() != shared.intermediate_size
            || expert_out.len() != output.len()
            || output.len() != shared.down.rows
        {
            return Err(XrtError::Runtime(
                "MoE shared-expert scratch does not match validated tensor geometry".to_string(),
            ));
        }
        let mut selector = [0.0f32; 1];
        self.linear_resolved(&shared.gate_selector, input, &mut selector)?;
        let shared_weight = 1.0 / (1.0 + (-selector[0]).exp());
        if !shared_weight.is_finite() {
            return Err(XrtError::Runtime(
                "MoE shared-expert sigmoid gate produced a non-finite value".to_string(),
            ));
        }
        self.linear_pair_resolved(&shared.gate, &shared.up, input, gate, up)?;
        swiglu(gate, up);
        self.linear_resolved(&shared.down, gate, expert_out)?;
        accumulate_scaled(output, expert_out, shared_weight);
        Ok(())
    }

    fn execute_shared_moe_batch(
        &self,
        shared: &MoeSharedExpertWeights,
        seq_len: usize,
        batch: &mut BatchScratch,
    ) -> Result<()> {
        let hidden_size = shared.gate_selector.cols;
        let intermediate_size = shared.intermediate_size;
        if batch.normed.len() < seq_len.saturating_mul(hidden_size)
            || batch.down.len() < seq_len.saturating_mul(hidden_size)
            || batch.proj.len() < seq_len.saturating_mul(hidden_size)
            || batch.gate.len() < seq_len.saturating_mul(intermediate_size)
            || batch.up.len() < seq_len.saturating_mul(intermediate_size)
            || batch.moe_shared_gate.len() < seq_len
        {
            return Err(XrtError::Runtime(
                "MoE shared-expert batch scratch does not match validated tensor geometry"
                    .to_string(),
            ));
        }

        self.linear_batch_resolved(
            &shared.gate_selector,
            &batch.normed[..seq_len * hidden_size],
            seq_len,
            &mut batch.moe_shared_gate[..seq_len],
        )?;
        self.linear_batch_resolved(
            &shared.gate,
            &batch.normed[..seq_len * hidden_size],
            seq_len,
            &mut batch.gate[..seq_len * intermediate_size],
        )?;
        self.linear_batch_resolved(
            &shared.up,
            &batch.normed[..seq_len * hidden_size],
            seq_len,
            &mut batch.up[..seq_len * intermediate_size],
        )?;
        for token in 0..seq_len {
            swiglu(
                &mut batch.gate[token * intermediate_size..(token + 1) * intermediate_size],
                &batch.up[token * intermediate_size..(token + 1) * intermediate_size],
            );
        }
        self.linear_batch_resolved(
            &shared.down,
            &batch.gate[..seq_len * intermediate_size],
            seq_len,
            &mut batch.proj[..seq_len * hidden_size],
        )?;
        for token in 0..seq_len {
            let shared_weight = 1.0 / (1.0 + (-batch.moe_shared_gate[token]).exp());
            if !shared_weight.is_finite() {
                return Err(XrtError::Runtime(format!(
                    "MoE shared-expert sigmoid gate produced a non-finite value for token {token}"
                )));
            }
            accumulate_scaled(
                &mut batch.down[token * hidden_size..(token + 1) * hidden_size],
                &batch.proj[token * hidden_size..(token + 1) * hidden_size],
                shared_weight,
            );
        }
        Ok(())
    }

    fn execute_moe_batch_configured(
        &self,
        descriptor: &MoeLayerDescriptor,
        router: &ResolvedWeight,
        experts: &[MoeExpertWeights],
        seq_len: usize,
        batch: &mut BatchScratch,
    ) -> Result<()> {
        let grouped_work = seq_len
            .checked_mul(descriptor.selected_per_token())
            .and_then(|value| value.checked_mul(descriptor.hidden_size()))
            .and_then(|value| value.checked_mul(descriptor.intermediate_size()))
            .unwrap_or(usize::MAX);
        // Small layers lose more to grouping and two dispatch barriers than
        // they gain from expert locality. This rollout guard keeps the exact
        // legacy executor for undersized batches while real MoE layers enter
        // the grouped path.
        if self.moe_cpu_execution == MoeCpuExecution::Optimized
            && seq_len > 1
            && grouped_work >= (1 << 20)
        {
            self.moe_telemetry.record_grouped_batch(seq_len);
            return self.execute_moe_batch_optimized(descriptor, router, experts, seq_len, batch);
        }

        self.moe_telemetry.record_legacy_batch();
        let hidden_size = descriptor.hidden_size();
        let intermediate_size = descriptor.intermediate_size();
        for token in 0..seq_len {
            self.execute_moe_token(
                descriptor,
                router,
                experts,
                &batch.normed[token * hidden_size..(token + 1) * hidden_size],
                &mut batch.moe_router_logits[..experts.len()],
                &mut batch.gate[token * intermediate_size..(token + 1) * intermediate_size],
                &mut batch.up[token * intermediate_size..(token + 1) * intermediate_size],
                &mut batch.proj[token * hidden_size..(token + 1) * hidden_size],
                &mut batch.down[token * hidden_size..(token + 1) * hidden_size],
            )?;
        }
        Ok(())
    }

    fn execute_moe_batch_optimized(
        &self,
        descriptor: &MoeLayerDescriptor,
        router: &ResolvedWeight,
        experts: &[MoeExpertWeights],
        seq_len: usize,
        batch: &mut BatchScratch,
    ) -> Result<()> {
        let expert_count = descriptor.expert_count();
        let hidden_size = descriptor.hidden_size();
        let intermediate_size = descriptor.intermediate_size();
        if experts.len() != expert_count
            || batch.moe_routes.len() < seq_len
            || batch.moe_router_logits.len() < seq_len.saturating_mul(expert_count)
            || batch.moe_expert_counts.len() < expert_count
            || batch.moe_expert_offsets.len() < expert_count.saturating_add(1)
            || batch.moe_expert_cursors.len() < expert_count
            || batch.moe_token_indices.len() < seq_len
            || batch.moe_inputs.len() < seq_len.saturating_mul(hidden_size)
        {
            return Err(XrtError::Runtime(format!(
                "MoE layer {} batch scratch does not match its validated descriptor",
                descriptor.layer_index()
            )));
        }

        self.linear_batch_resolved(
            router,
            &batch.normed[..seq_len * hidden_size],
            seq_len,
            &mut batch.moe_router_logits[..seq_len * expert_count],
        )?;
        for token in 0..seq_len {
            route_top_k(
                &batch.moe_router_logits[token * expert_count..(token + 1) * expert_count],
                descriptor.selected_per_token(),
                &mut batch.moe_routes[token],
            )?;
            self.moe_telemetry.record_route(&batch.moe_routes[token]);
            #[cfg(feature = "moe-route-trace")]
            self.moe_telemetry
                .record_route_trace(descriptor.layer_index(), &batch.moe_routes[token]);
        }

        batch.down[..seq_len * hidden_size].fill(0.0);
        for route_slot in 0..descriptor.selected_per_token() {
            group_route_slot_by_expert(
                &batch.moe_routes[..seq_len],
                route_slot,
                expert_count,
                &mut batch.moe_expert_counts,
                &mut batch.moe_expert_offsets,
                &mut batch.moe_expert_cursors,
                &mut batch.moe_token_indices,
            )?;

            for expert_id in 0..expert_count {
                let group_start = batch.moe_expert_offsets[expert_id];
                let group_end = batch.moe_expert_offsets[expert_id + 1];
                for grouped_index in group_start..group_end {
                    let token = batch.moe_token_indices[grouped_index];
                    batch.moe_inputs
                        [grouped_index * hidden_size..(grouped_index + 1) * hidden_size]
                        .copy_from_slice(
                            &batch.normed[token * hidden_size..(token + 1) * hidden_size],
                        );
                }
            }

            let input_address = batch.moe_inputs.as_ptr() as usize;
            let gate_address = batch.gate.as_mut_ptr() as usize;
            let up_address = batch.up.as_mut_ptr() as usize;
            let projection_address = batch.proj.as_mut_ptr() as usize;
            let errors = std::sync::Mutex::new(None);
            let worker_result = global_expert_pool()
                .submit_scoped(expert_count, |start_expert, end_expert| {
                    for expert_id in start_expert..end_expert {
                        if errors.lock().expect("MoE error lock poisoned").is_some() {
                            return;
                        }
                        let group_start = batch.moe_expert_offsets[expert_id];
                        let group_end = batch.moe_expert_offsets[expert_id + 1];
                        let group_len = group_end - group_start;
                        if group_len == 0 {
                            continue;
                        }
                        let expert = &experts[expert_id];
                        // SAFETY: grouping assigns each expert a disjoint
                        // [group_start, group_end) span in every mutable buffer.
                        // The buffers remain borrowed for this complete joined
                        // dispatch, and nested dense kernels run serially within
                        // each bounded expert worker.
                        let result = unsafe {
                            let inputs = std::slice::from_raw_parts(
                                (input_address as *const f32).add(group_start * hidden_size),
                                group_len * hidden_size,
                            );
                            let gate = std::slice::from_raw_parts_mut(
                                (gate_address as *mut f32).add(group_start * intermediate_size),
                                group_len * intermediate_size,
                            );
                            let up = std::slice::from_raw_parts_mut(
                                (up_address as *mut f32).add(group_start * intermediate_size),
                                group_len * intermediate_size,
                            );
                            let projection = std::slice::from_raw_parts_mut(
                                (projection_address as *mut f32).add(group_start * hidden_size),
                                group_len * hidden_size,
                            );
                            (|| -> Result<()> {
                                self.linear_batch_resolved(&expert.gate, inputs, group_len, gate)?;
                                self.linear_batch_resolved(&expert.up, inputs, group_len, up)?;
                                for local_index in 0..group_len {
                                    swiglu(
                                        &mut gate[local_index * intermediate_size
                                            ..(local_index + 1) * intermediate_size],
                                        &up[local_index * intermediate_size
                                            ..(local_index + 1) * intermediate_size],
                                    );
                                }
                                self.linear_batch_resolved(
                                    &expert.down,
                                    gate,
                                    group_len,
                                    projection,
                                )
                            })()
                        };
                        if let Err(error) = result {
                            let mut first_error = errors.lock().expect("MoE error lock poisoned");
                            if first_error.is_none() {
                                *first_error = Some(error);
                            }
                            return;
                        }
                    }
                })
                .join();
            if let Err(error) = worker_result {
                self.moe_telemetry.record_worker_failure();
                return Err(error);
            }
            if let Some(error) = errors
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
            {
                self.moe_telemetry.record_worker_failure();
                return Err(error);
            }

            for expert_id in 0..expert_count {
                let group_start = batch.moe_expert_offsets[expert_id];
                let group_end = batch.moe_expert_offsets[expert_id + 1];
                for grouped_index in group_start..group_end {
                    let token = batch.moe_token_indices[grouped_index];
                    let weight = batch.moe_routes[token].weights()[route_slot];
                    accumulate_scaled(
                        &mut batch.down[token * hidden_size..(token + 1) * hidden_size],
                        &batch.proj[grouped_index * hidden_size..(grouped_index + 1) * hidden_size],
                        weight,
                    );
                }
            }
        }
        Ok(())
    }

    fn linear_resolved_float_reference(
        &self,
        w: &ResolvedWeight,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if !w.dtype.is_quantized() {
            return self.linear_resolved(w, input, output);
        }
        if w.rows == 0 || w.nbytes % w.rows != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "tensor {} byte length {} is not divisible by row count {}",
                w.name, w.nbytes, w.rows
            )));
        }
        let bytes = self.gguf.tensor_data_raw(w.data_offset, w.nbytes);
        let row_bytes = w.nbytes / w.rows;
        for (row, output_value) in output.iter_mut().enumerate().take(w.rows) {
            let start = row * row_bytes;
            *output_value = quantized_row_dot(w.dtype, &bytes[start..start + row_bytes], input)?;
        }
        if let Some(ref lora) = self.lora {
            if lora.has_weight(&w.name) {
                lora.apply(&w.name, input, output)?;
            }
        }
        Ok(())
    }

    fn load_vector(&self, tensor_name: &str) -> Result<Arc<Vec<f32>>> {
        if let Some(cached) = self.vector_cache.read().get(tensor_name).cloned() {
            return Ok(cached);
        }

        let info = self.gguf.require_tensor(tensor_name)?;
        if info.rows() != 1 {
            return Err(XrtError::Model(format!(
                "tensor {tensor_name} is not a vector (rows = {})",
                info.rows()
            )));
        }

        let bytes = self.gguf.tensor_data(tensor_name)?;
        let mut values = vec![0.0f32; info.row_len()];
        self.decode_row_into(info, bytes, 0, &mut values)?;
        let values = Arc::new(values);
        self.vector_cache
            .write()
            .insert(tensor_name.to_string(), values.clone());
        Ok(values)
    }

    fn decode_row_into(
        &self,
        info: &TensorInfo,
        bytes: &[u8],
        row: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let rows = info.rows();
        let cols = info.row_len();
        if row >= rows {
            return Err(XrtError::InvalidTensor(format!(
                "row {row} is out of range for tensor {} with {rows} rows",
                info.name
            )));
        }
        if output.len() != cols {
            return Err(XrtError::InvalidTensor(format!(
                "output row length {} does not match tensor {} row width {cols}",
                output.len(),
                info.name
            )));
        }

        let row_bytes = info.nbytes / rows;
        let start = row * row_bytes;
        let end = start + row_bytes;
        let row_bytes = &bytes[start..end];

        match info.dtype {
            DType::F32 => {
                for (index, chunk) in row_bytes.chunks_exact(4).enumerate() {
                    output[index] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            DType::F16 => {
                for (index, chunk) in row_bytes.chunks_exact(2).enumerate() {
                    output[index] = decode_f16(chunk)?;
                }
            }
            DType::BF16 => {
                for (index, chunk) in row_bytes.chunks_exact(2).enumerate() {
                    output[index] = decode_bf16(chunk)?;
                }
            }
            DType::Q8_0 => dequantize_q8_0_row(row_bytes, output)?,
            DType::Q4_0 => dequantize_q4_0_row(row_bytes, output)?,
            DType::Q4_K => dequantize_q4_k_row(row_bytes, output)?,
            DType::Q5_K => dequantize_q5_k_row(row_bytes, output)?,
            DType::Q6_K => dequantize_q6_k_row(row_bytes, output)?,
            DType::MXFP4 => dequantize_mxfp4_row(row_bytes, output)?,
        }

        Ok(())
    }
}

fn qwen35_delta_qk_group(v_head: usize, v_heads: usize, num_groups: usize) -> usize {
    debug_assert!(v_heads > 0);
    debug_assert!(num_groups > 0);
    (v_head * num_groups) / v_heads
}

fn expand_layer_usizes(
    values: Option<Vec<usize>>,
    default_value: usize,
    layer_count: usize,
    name: &str,
) -> Result<Vec<usize>> {
    match values {
        Some(values) if values.len() == layer_count => Ok(values),
        Some(values) if values.len() == 1 => Ok(vec![values[0]; layer_count]),
        Some(values) => Err(XrtError::InvalidMetadata(format!(
            "{name} has {} entries, expected {layer_count}",
            values.len()
        ))),
        None => Ok(vec![default_value; layer_count]),
    }
}

fn expand_layer_bools(
    values: Option<Vec<bool>>,
    default_value: bool,
    layer_count: usize,
    name: &str,
) -> Result<Vec<bool>> {
    match values {
        Some(values) if values.len() == layer_count => Ok(values),
        Some(values) if values.len() == 1 => Ok(vec![values[0]; layer_count]),
        Some(values) => Err(XrtError::InvalidMetadata(format!(
            "{name} has {} entries, expected {layer_count}",
            values.len()
        ))),
        None => Ok(vec![default_value; layer_count]),
    }
}

#[cfg(test)]
mod tests {
    use super::{qwen35_delta_qk_group, LlamaConfig};
    use xrt_safetensors::HfModelConfig;

    #[test]
    fn hf_qwen2_config_maps_to_standard_dense_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2",
                "hidden_size": 2048,
                "intermediate_size": 11008,
                "max_position_embeddings": 131072,
                "num_attention_heads": 16,
                "num_hidden_layers": 36,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "rope_scaling": null,
                "use_sliding_window": false,
                "tie_word_embeddings": true,
                "hidden_act": "silu",
                "torch_dtype": "bfloat16",
                "vocab_size": 151936
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.feed_forward_length, 11008);
        assert_eq!(config.context_length, 131072);
        assert_eq!(config.block_count, 36);
        assert_eq!(config.attention_head_count, 16);
        assert_eq!(config.attention_head_count_kv, 2);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.q_width(), 2048);
        assert_eq!(config.kv_width(), 256);
        assert_eq!(config.rope_freq_base, 1000000.0);
        assert_eq!(config.rms_norm_eps, 0.000001);
        assert!(!config.is_gemma4());
        assert!(!config.is_hybrid());
    }

    #[test]
    fn hf_qwen25_vl_default_mrope_maps_text_only_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2_5_vl",
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "max_position_embeddings": 128000,
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "mrope_section": [16, 24, 24],
                    "rope_type": "default",
                    "type": "default"
                },
                "use_sliding_window": false,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "dtype": "bfloat16",
                "vocab_size": 152064
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen2_5_vl");
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.kv_width(), 512);
    }

    #[test]
    fn hf_qwen25_vl_rejects_nondefault_mrope() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2_5_vl",
                "hidden_size": 128,
                "intermediate_size": 256,
                "max_position_embeddings": 128,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "mrope_section": [16, 24, 24],
                    "rope_type": "dynamic",
                    "type": "dynamic"
                },
                "use_sliding_window": false,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "dtype": "bfloat16",
                "vocab_size": 256
            }"#,
        )
        .unwrap();
        assert!(LlamaConfig::from_hf(&hf).is_err());
    }

    #[test]
    fn hf_qwen2_autoawq_reuses_standard_model_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2",
                "hidden_size": 32,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "use_sliding_window": false,
                "tie_word_embeddings": true,
                "hidden_act": "silu",
                "torch_dtype": "float16",
                "vocab_size": 16,
                "quantization_config": {
                    "quant_method": "awq",
                    "w_bit": 4,
                    "q_group_size": 32,
                    "zero_point": true,
                    "version": "GEMM"
                }
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.embedding_length, 32);
        assert_eq!(config.feed_forward_length, 64);
        assert_eq!(config.block_count, 1);
    }

    #[test]
    fn hf_qwen3_autoawq_gemv_reuses_standard_model_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "max_position_embeddings": 32768,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "rope_scaling": null,
                "use_sliding_window": false,
                "tie_word_embeddings": true,
                "hidden_act": "silu",
                "torch_dtype": "float16",
                "vocab_size": 151936,
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": true,
                    "version": "gemv"
                }
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen3");
        assert_eq!(config.embedding_length, 1024);
        assert_eq!(config.feed_forward_length, 3072);
        assert_eq!(config.block_count, 28);
        assert_eq!(config.attention_head_count, 16);
        assert_eq!(config.attention_head_count_kv, 8);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.q_width(), 2048);
        assert_eq!(config.kv_width(), 1024);
    }

    #[test]
    fn hf_qwen2_gptq_v1_reuses_standard_model_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2",
                "hidden_size": 32,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "use_sliding_window": false,
                "tie_word_embeddings": true,
                "hidden_act": "silu",
                "torch_dtype": "float16",
                "vocab_size": 16,
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 32,
                    "sym": true,
                    "desc_act": false,
                    "exllama_config": {"version": 1}
                }
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.embedding_length, 32);
        assert_eq!(config.feed_forward_length, 64);
        assert_eq!(config.block_count, 1);
    }

    #[test]
    fn hf_qwen2_compressed_tensors_reuses_standard_model_geometry() {
        let hf = HfModelConfig::from_json_bytes(
            br#"{
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "max_position_embeddings": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "use_sliding_window": false,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "torch_dtype": "bfloat16",
                "vocab_size": 16,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "quantization_status": "compressed",
                    "config_groups": {
                        "group_0": {
                            "targets": ["Linear"],
                            "input_activations": null,
                            "output_activations": null,
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
                    }
                }
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf(&hf).unwrap();
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.embedding_length, 64);
        assert_eq!(config.feed_forward_length, 128);
        assert_eq!(config.block_count, 1);
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.kv_width(), 32);
    }

    #[test]
    fn qwen35_delta_group_mapping_handles_more_v_heads_than_qk_groups() {
        let mapping: Vec<usize> = (0..32)
            .map(|v_head| qwen35_delta_qk_group(v_head, 32, 16))
            .collect();
        let expected: Vec<usize> = (0..16).flat_map(|group| [group, group]).collect();
        assert_eq!(mapping, expected);
    }

    #[test]
    fn qwen35_delta_group_mapping_is_identity_when_counts_match() {
        let mapping: Vec<usize> = (0..16)
            .map(|v_head| qwen35_delta_qk_group(v_head, 16, 16))
            .collect();
        let expected: Vec<usize> = (0..16).collect();
        assert_eq!(mapping, expected);
    }
}
