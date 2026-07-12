use parking_lot::RwLock;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::info;
use xrt_core::{decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_kernels::cpu::{
    accumulate_scaled, add_inplace, apply_rmsnorm, delta_rule_group, dequantize_q4_0_row,
    dequantize_q4_k_row, dequantize_q5_k_row, dequantize_q6_k_row, dequantize_q8_0_row, dot,
    gated_rmsnorm, geglu_pytorch_tanh, global_pool, l2_normalize, matvec_quantized,
    matvec_quantized_batch, matvec_quantized_fused, matvec_quantized_fused_mixed,
    quantized_row_dot, silu_inplace_fast, swiglu, RopeFreqs,
};
use xrt_safetensors::HfModelConfig;

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
        "qwen3" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen3,
            metadata_prefixes: &["qwen3"],
        }),
        "qwen35" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen35", "qwen3_5", "qwen3_5_moe", "qwen3_next"],
        }),
        "qwen3_5" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen3_5", "qwen35", "qwen3_5_moe", "qwen3_next"],
        }),
        "qwen3_5_moe" => Ok(ArchitectureDescriptor {
            family: ArchitectureFamily::Qwen35Like,
            metadata_prefixes: &["qwen3_5_moe", "qwen3_5", "qwen35", "qwen3_next"],
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
            "xrt-models supports llama, qwen2/qwen2.5, qwen3, qwen35/qwen3.5, qwen3-next, and gemma4 architectures, found {architecture}"
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
    pub block_count: usize,
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
    // Qwen3.5 DeltaNet SSM parameters (None for standard transformer models)
    pub ssm_conv_kernel: Option<usize>,
    pub ssm_state_size: Option<usize>,
    pub ssm_group_count: Option<usize>,
    pub ssm_inner_size: Option<usize>,
    pub ssm_dt_rank: Option<usize>,
    gemma4: Option<Gemma4Config>,
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
        let feed_forward_length = required_usize_any(gguf, prefixes, "feed_forward_length")?;
        let block_count = required_usize_any(gguf, prefixes, "block_count")?;
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
        let expert_count = metadata_usize_any(gguf, prefixes, "expert_count");
        let expert_used_count = metadata_usize_any(gguf, prefixes, "expert_used_count");

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

        Ok(Self {
            architecture,
            architecture_family: descriptor.family,
            vocab_size,
            context_length,
            embedding_length,
            feed_forward_length,
            block_count,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count,
            rms_norm_eps,
            rope_freq_base,
            rope_freq_scale,
            head_dim_override,
            expert_count,
            expert_used_count,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_inner_size,
            ssm_dt_rank,
            gemma4,
        })
    }

    pub fn from_hf(config: &HfModelConfig) -> Result<Self> {
        let model_type = config.model_type.trim().to_ascii_lowercase();
        if model_type != "qwen2" {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors CUDA decode currently supports dense Qwen2 models, found model_type `{}`",
                config.model_type
            )));
        }
        if config.quantization.is_some() {
            return Err(XrtError::Unsupported(
                "SafeTensors CUDA decode currently supports dense F32/F16/BF16 weights; AWQ, GPTQ, and compressed-tensors kernels are not wired yet"
                    .to_string(),
            ));
        }
        if !matches!(
            config.hidden_act.trim().to_ascii_lowercase().as_str(),
            "silu" | "swish"
        ) {
            return Err(XrtError::Unsupported(format!(
                "SafeTensors Qwen2 CUDA decode requires SiLU activation, found `{}`",
                config.hidden_act
            )));
        }
        if config.use_sliding_window {
            return Err(XrtError::Unsupported(
                "SafeTensors Qwen2 sliding-window attention is not wired into the standard dense CUDA path"
                    .to_string(),
            ));
        }
        if config
            .raw
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return Err(XrtError::Unsupported(
                "SafeTensors Qwen2 rope_scaling variants are not wired into the CUDA path"
                    .to_string(),
            ));
        }

        let descriptor = describe_architecture(&model_type)?;
        let default_head_dim = config.hidden_size / config.num_attention_heads;
        let actual_head_dim = config.head_dim.unwrap_or(default_head_dim);
        if actual_head_dim == 0 {
            return Err(XrtError::InvalidMetadata(
                "SafeTensors Qwen2 head dimension must be greater than zero".to_string(),
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
            attention_head_count: config.num_attention_heads,
            attention_head_count_kv: config.num_key_value_heads,
            rope_dimension_count: actual_head_dim,
            rms_norm_eps: config.rms_norm_eps,
            rope_freq_base: config.rope_theta,
            rope_freq_scale: 1.0,
            head_dim_override: (actual_head_dim != default_head_dim).then_some(actual_head_dim),
            expert_count: None,
            expert_used_count: None,
            ssm_conv_kernel: None,
            ssm_state_size: None,
            ssm_group_count: None,
            ssm_inner_size: None,
            ssm_dt_rank: None,
            gemma4: None,
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

    pub fn is_qwen35_family(&self) -> bool {
        self.architecture_family == ArchitectureFamily::Qwen35Like
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

/// DeltaNet recurrent state per layer: conv1d sliding window + state matrix.
struct DeltaNetLayerState {
    /// Conv1d sliding window: (conv_kernel - 1) previous QKV vectors.
    /// Layout: [conv_kernel - 1][conv_channels], newest at the end.
    conv_state: Vec<f32>,
    /// Recurrent state matrices: [num_v_heads][head_v_dim][head_k_dim].
    recurrent_state: Vec<f32>,
}

/// DeltaNet state for all recurrent layers in a hybrid model.
struct DeltaNetState {
    /// Per-layer state. None for full-attention layers.
    layers: Vec<Option<DeltaNetLayerState>>,
    /// Current sequence position (how many tokens have been processed).
    position: usize,
}

impl DeltaNetState {
    fn new(config: &LlamaConfig) -> Self {
        let conv_kernel = config.ssm_conv_kernel.unwrap_or(4);
        let state_size = config.ssm_state_size.unwrap_or(128);
        let group_count = config.ssm_group_count.unwrap_or(16);
        let inner_size = config.ssm_inner_size.unwrap_or(2048);
        let dt_rank = config.ssm_dt_rank.unwrap_or(16);
        let head_v_dim = inner_size / dt_rank;
        let conv_channels = state_size * group_count * 2 + head_v_dim * dt_rank;
        let layers = (0..config.block_count)
            .map(|i| {
                if config.is_recurrent(i) {
                    Some(DeltaNetLayerState {
                        conv_state: vec![0.0; (conv_kernel - 1) * conv_channels],
                        recurrent_state: vec![0.0; dt_rank * head_v_dim * state_size],
                    })
                } else {
                    None
                }
            })
            .collect();
        Self {
            layers,
            position: 0,
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            if let Some(ref mut state) = layer {
                state.conv_state.fill(0.0);
                state.recurrent_state.fill(0.0);
            }
        }
        self.position = 0;
    }

    /// Save a snapshot of all layer states for speculative rollback.
    fn save_snapshot(&self) -> Vec<Option<(Vec<f32>, Vec<f32>)>> {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .as_ref()
                    .map(|s| (s.conv_state.clone(), s.recurrent_state.clone()))
            })
            .collect()
    }

    /// Restore a saved snapshot, rolling back all recurrent state.
    fn restore_snapshot(&mut self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]) {
        for (layer, snap) in self.layers.iter_mut().zip(snapshot.iter()) {
            if let (Some(state), Some((conv, recur))) = (layer.as_mut(), snap.as_ref()) {
                state.conv_state.copy_from_slice(conv);
                state.recurrent_state.copy_from_slice(recur);
            }
        }
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
        router: ResolvedWeight,
        experts: Vec<MoeExpertWeights>,
    },
}

#[derive(Debug, Clone)]
struct MoeExpertWeights {
    gate: ResolvedWeight,
    down: ResolvedWeight,
    up: ResolvedWeight,
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
        Self {
            normed: vec![0.0; config.embedding_length],
            q: vec![0.0; config.q_width().max(qg_size / 2)],
            k: vec![0.0; config.kv_width()],
            v: vec![0.0; config.kv_width()],
            gate: vec![0.0; config.feed_forward_length],
            up: vec![0.0; config.feed_forward_length],
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
            self.gate[..seq_len * config.feed_forward_length].fill(0.0);
            self.up[..seq_len * config.feed_forward_length].fill(0.0);
            self.attn_out[..seq_len * config.q_width()].fill(0.0);
            self.proj[..seq_len * dim].fill(0.0);
            self.down[..seq_len * dim].fill(0.0);
            return;
        }
        let dim = config.embedding_length;
        self.xs = vec![0.0; seq_len * dim];
        self.normed = vec![0.0; seq_len * dim];
        self.q = vec![0.0; seq_len * config.q_width()];
        self.k = vec![0.0; seq_len * config.kv_width()];
        self.v = vec![0.0; seq_len * config.kv_width()];
        self.gate = vec![0.0; seq_len * config.feed_forward_length];
        self.up = vec![0.0; seq_len * config.feed_forward_length];
        self.attn_out = vec![0.0; seq_len * config.q_width()];
        self.proj = vec![0.0; seq_len * dim];
        self.down = vec![0.0; seq_len * dim];
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
    lora: Option<crate::lora::LoraAdapter>,
    vector_cache: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    rope_freqs: RopeFreqs,
    gemma4_rope_freqs: Vec<RopeFreqs>,
    scratch: RwLock<ForwardScratch>,
    batch_scratch: RwLock<BatchScratch>,
    deltanet_state: RwLock<Option<DeltaNetState>>,
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

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Self> {
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
                AttnWeights::DeltaNet {
                    attn_qkv: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_qkv.weight"))?,
                    attn_gate: Self::resolve_weight(
                        &gguf,
                        &format!("blk.{index}.attn_gate.weight"),
                    )?,
                    ssm_alpha: Self::resolve_weight(
                        &gguf,
                        &format!("blk.{index}.ssm_alpha.weight"),
                    )?,
                    ssm_beta: Self::resolve_weight(&gguf, &format!("blk.{index}.ssm_beta.weight"))?,
                    ssm_a: format!("blk.{index}.ssm_a"),
                    ssm_dt_bias: format!("blk.{index}.ssm_dt.bias"),
                    ssm_norm: format!("blk.{index}.ssm_norm.weight"),
                    ssm_out: Self::resolve_weight(&gguf, &format!("blk.{index}.ssm_out.weight"))?,
                }
            } else if config.is_hybrid() {
                // Qwen3.5 full attention layer (Q+gate interleaved)
                AttnWeights::Qwen35Attn {
                    attn_qg: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_q.weight"))?,
                    attn_k: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_k.weight"))?,
                    attn_v: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_v.weight"))?,
                    attn_output: Self::resolve_weight(
                        &gguf,
                        &format!("blk.{index}.attn_output.weight"),
                    )?,
                    attn_q_norm: format!("blk.{index}.attn_q_norm.weight"),
                    attn_k_norm: format!("blk.{index}.attn_k_norm.weight"),
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
                let n_experts = config.expert_count.unwrap();
                let router =
                    Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_gate_inp.weight"))?;
                let mut experts = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    experts.push(MoeExpertWeights {
                        gate: Self::resolve_weight(
                            &gguf,
                            &format!("blk.{index}.ffn_gate.{e}.weight"),
                        )?,
                        down: Self::resolve_weight(
                            &gguf,
                            &format!("blk.{index}.ffn_down.{e}.weight"),
                        )?,
                        up: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_up.{e}.weight"))?,
                    });
                }
                FfnWeights::Moe { router, experts }
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
                let bytes = gguf.tensor_data(&conv_name)?;
                let total = info.rows() * info.row_len();
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

        let deltanet_state = if config.is_hybrid() {
            Some(DeltaNetState::new(&config))
        } else {
            None
        };

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
            lora: None,
            vector_cache: RwLock::new(HashMap::new()),
            rope_freqs,
            gemma4_rope_freqs,
            scratch,
            batch_scratch: RwLock::new(BatchScratch::new()),
            deltanet_state: RwLock::new(deltanet_state),
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

    /// Clear DeltaNet recurrent state (for hybrid models). No-op for standard transformers.
    pub fn clear_state(&self) {
        if let Some(ref mut state) = *self.deltanet_state.write() {
            state.clear();
        }
    }

    /// Save DeltaNet recurrent state snapshot for speculative rollback.
    /// Returns an opaque snapshot that can be passed to `restore_state`.
    pub fn save_state(&self) -> Option<Vec<Option<(Vec<f32>, Vec<f32>)>>> {
        self.deltanet_state
            .read()
            .as_ref()
            .map(|s| s.save_snapshot())
    }

    /// Restore a previously saved DeltaNet state snapshot.
    pub fn restore_state(&self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]) {
        if let Some(ref mut state) = *self.deltanet_state.write() {
            state.restore_snapshot(snapshot);
        }
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
        self.forward_token_inner(token_id, position, Some(n_layers), cache, output_logits)
    }

    pub fn forward_token<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_token_inner(token_id, position, None, cache, output_logits)
    }

    fn forward_token_inner<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        max_layers: Option<usize>,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        let n_layers = max_layers.unwrap_or(self.config.block_count);
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
        let mut dn_state = self.deltanet_state.write();
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
                    let dn = dn_state.as_mut().ok_or_else(|| {
                        XrtError::Runtime("DeltaNet state not initialized".to_string())
                    })?;
                    let layer_state = dn.layers[layer_index].as_mut().ok_or_else(|| {
                        XrtError::Runtime(format!("layer {layer_index} is not a DeltaNet layer"))
                    })?;

                    let num_groups = self.config.ssm_group_count.unwrap();
                    let state_size = self.config.ssm_state_size.unwrap(); // head_k_dim
                    let inner_size = self.config.ssm_inner_size.unwrap();
                    let dt_rank = self.config.ssm_dt_rank.unwrap(); // num_v_heads
                    let head_v_dim = inner_size / dt_rank;
                    let conv_kernel = self.config.ssm_conv_kernel.unwrap();
                    let conv_channels = state_size * num_groups * 2 + inner_size;

                    // 1. Fused QKV + gate + alpha + beta projections (all share normed input)
                    // Single dispatch: quantize input once, all 4 projections in one par_for
                    {
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
                        let cs = &layer_state.conv_state;
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

                    // Now shift state left and insert current input
                    layer_state.conv_state.copy_within(conv_channels.., 0);
                    layer_state.conv_state[(history - 1) * conv_channels..history * conv_channels]
                        .copy_from_slice(&dn_qkv[..conv_channels]);

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

                        let state_g = &mut layer_state.recurrent_state
                            [state_off..state_off + head_v_dim * state_size];

                        unsafe {
                            delta_rule_group(
                                state_g,
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
                FfnWeights::Moe { router, experts } => {
                    let n_experts_used = self.config.expert_used_count.unwrap_or(2);
                    let rl = &mut moe_router_logits[..experts.len()];
                    self.linear_resolved(router, normed, rl)?;

                    // Select top-K experts by score
                    let selected = top_k_indices(rl, n_experts_used);

                    // Softmax over selected expert logits for routing weights
                    let mut routing_weights = vec![0.0f32; n_experts_used];
                    let max_logit = selected
                        .iter()
                        .map(|&(_, s)| s)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for (i, &(_, logit)) in selected.iter().enumerate() {
                        let e = (logit - max_logit).exp();
                        routing_weights[i] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for w in &mut routing_weights {
                        *w *= inv_sum;
                    }

                    // Compute weighted sum of expert outputs
                    down.fill(0.0);
                    let dim = self.config.embedding_length;
                    for (i, &(expert_idx, _)) in selected.iter().enumerate() {
                        let expert = &experts[expert_idx];
                        self.linear_resolved(&expert.gate, normed, gate)?;
                        self.linear_resolved(&expert.up, normed, up)?;
                        swiglu(gate, up);
                        let eout = &mut moe_expert_out[..dim];
                        self.linear_resolved(&expert.down, gate, eout)?;
                        accumulate_scaled(down, eout, routing_weights[i]);
                    }
                    add_inplace(&mut x, down);
                }
                FfnWeights::Gemma4Dense { .. } => unreachable!(
                    "Gemma4 FFN uses forward_gemma4_token_inner and never reaches this path"
                ),
            }
        }

        // Update DeltaNet position
        if let Some(ref mut dn) = *dn_state {
            dn.position += 1;
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        apply_rmsnorm(&x, &output_norm_weight, eps, normed);

        // Output projection directly into caller's buffer (zero alloc per token).
        self.linear_resolved(&self.output, normed, output_logits)?;
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
        self.forward_batch_inner(token_ids, start_position, cache, None)
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
        self.forward_batch_inner(token_ids, start_position, cache, Some(embedding_overrides))
    }

    fn forward_batch_inner<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
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
            self.forward_token(token_ids[0], start_position, cache, &mut logits)?;
            return Ok(logits);
        }

        // Hybrid and Gemma4 models use layer-specific state/widths, so process sequentially first.
        if self.config.is_hybrid() || self.config.is_gemma4() {
            let mut logits = vec![0.0; self.config.vocab_size];
            for (i, &token_id) in token_ids.iter().enumerate() {
                self.forward_token(token_id, start_position + i, cache, &mut logits)?;
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
                FfnWeights::Moe { router, experts } => {
                    let n_experts_used = self.config.expert_used_count.unwrap_or(2);
                    // Process each token independently through MoE
                    batch.down[..seq_len * dim].fill(0.0);
                    let mut router_logits = vec![0.0f32; experts.len()];
                    let mut expert_gate = vec![0.0f32; ff_dim];
                    let mut expert_up = vec![0.0f32; ff_dim];
                    let mut expert_out = vec![0.0f32; dim];
                    for t in 0..seq_len {
                        let normed_t = &batch.normed[t * dim..(t + 1) * dim];
                        let down_t = &mut batch.down[t * dim..(t + 1) * dim];

                        self.linear_resolved(router, normed_t, &mut router_logits)?;
                        let selected = top_k_indices(&router_logits, n_experts_used);

                        // Softmax over selected
                        let max_l = selected
                            .iter()
                            .map(|&(_, s)| s)
                            .fold(f32::NEG_INFINITY, f32::max);
                        let mut weights = vec![0.0f32; n_experts_used];
                        let mut sum_exp = 0.0f32;
                        for (i, &(_, logit)) in selected.iter().enumerate() {
                            let e = (logit - max_l).exp();
                            weights[i] = e;
                            sum_exp += e;
                        }
                        let inv_sum = 1.0 / sum_exp;
                        for w in &mut weights {
                            *w *= inv_sum;
                        }

                        for (i, &(expert_idx, _)) in selected.iter().enumerate() {
                            let expert = &experts[expert_idx];
                            self.linear_resolved(&expert.gate, normed_t, &mut expert_gate)?;
                            self.linear_resolved(&expert.up, normed_t, &mut expert_up)?;
                            swiglu(&mut expert_gate, &expert_up);
                            self.linear_resolved(&expert.down, &expert_gate, &mut expert_out)?;
                            accumulate_scaled(down_t, &expert_out, weights[i]);
                        }
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
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        if seq_len == 1 {
            let mut logits = vec![0.0; self.config.vocab_size];
            self.forward_token(token_ids[0], start_position, cache, &mut logits)?;
            return Ok(logits);
        }

        // Hybrid and Gemma4 models use layer-specific state/widths, so process sequentially first.
        if self.config.is_hybrid() || self.config.is_gemma4() {
            let vocab_size = self.config.vocab_size;
            let mut all_logits = vec![0.0f32; seq_len * vocab_size];
            let mut logits = vec![0.0; vocab_size];
            for (i, &token_id) in token_ids.iter().enumerate() {
                self.forward_token(token_id, start_position + i, cache, &mut logits)?;
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
                FfnWeights::Moe { router, experts } => {
                    let n_experts_used = self.config.expert_used_count.unwrap_or(2);
                    batch.down[..seq_len * dim].fill(0.0);
                    let mut router_logits = vec![0.0f32; experts.len()];
                    let mut expert_gate = vec![0.0f32; ff_dim];
                    let mut expert_up = vec![0.0f32; ff_dim];
                    let mut expert_out = vec![0.0f32; dim];
                    for t in 0..seq_len {
                        let normed_t = &batch.normed[t * dim..(t + 1) * dim];
                        let down_t = &mut batch.down[t * dim..(t + 1) * dim];
                        self.linear_resolved(router, normed_t, &mut router_logits)?;
                        let selected = top_k_indices(&router_logits, n_experts_used);
                        let max_l = selected
                            .iter()
                            .map(|&(_, s)| s)
                            .fold(f32::NEG_INFINITY, f32::max);
                        let mut weights = vec![0.0f32; n_experts_used];
                        let mut sum_exp = 0.0f32;
                        for (i, &(_, logit)) in selected.iter().enumerate() {
                            let e = (logit - max_l).exp();
                            weights[i] = e;
                            sum_exp += e;
                        }
                        let inv_sum = 1.0 / sum_exp;
                        for w in &mut weights {
                            *w *= inv_sum;
                        }
                        for (i, &(expert_idx, _)) in selected.iter().enumerate() {
                            let expert = &experts[expert_idx];
                            self.linear_resolved(&expert.gate, normed_t, &mut expert_gate)?;
                            self.linear_resolved(&expert.up, normed_t, &mut expert_up)?;
                            swiglu(&mut expert_gate, &expert_up);
                            self.linear_resolved(&expert.down, &expert_gate, &mut expert_out)?;
                            accumulate_scaled(down_t, &expert_out, weights[i]);
                        }
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
                FfnWeights::Moe { router, experts } => {
                    let n_experts_used = self.config.expert_used_count.unwrap_or(2);
                    batch.down[..seq_len * dim].fill(0.0);
                    let mut router_logits = vec![0.0f32; experts.len()];
                    let mut expert_gate = vec![0.0f32; ff_dim];
                    let mut expert_up = vec![0.0f32; ff_dim];
                    let mut expert_out = vec![0.0f32; dim];
                    for t in 0..seq_len {
                        let normed_t = &batch.normed[t * dim..(t + 1) * dim];
                        let down_t = &mut batch.down[t * dim..(t + 1) * dim];
                        self.linear_resolved(router, normed_t, &mut router_logits)?;
                        let selected = top_k_indices(&router_logits, n_experts_used);
                        let max_l = selected
                            .iter()
                            .map(|&(_, s)| s)
                            .fold(f32::NEG_INFINITY, f32::max);
                        let mut weights = vec![0.0f32; n_experts_used];
                        let mut sum_exp = 0.0f32;
                        for (i, &(_, logit)) in selected.iter().enumerate() {
                            let e = (logit - max_l).exp();
                            weights[i] = e;
                            sum_exp += e;
                        }
                        let inv_sum = 1.0 / sum_exp;
                        for w in &mut weights {
                            *w *= inv_sum;
                        }
                        for (i, &(expert_idx, _)) in selected.iter().enumerate() {
                            let expert = &experts[expert_idx];
                            self.linear_resolved(&expert.gate, normed_t, &mut expert_gate)?;
                            self.linear_resolved(&expert.up, normed_t, &mut expert_up)?;
                            swiglu(&mut expert_gate, &expert_up);
                            self.linear_resolved(&expert.down, &expert_gate, &mut expert_out)?;
                            accumulate_scaled(down_t, &expert_out, weights[i]);
                        }
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

/// Return the top-K (index, score) pairs from `logits`, sorted by score descending.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indices: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let n = k.min(indices.len());
    if n == 0 {
        return Vec::new();
    }
    // Partial sort: move top-k to front
    let nth = n - 1;
    indices.select_nth_unstable_by(nth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(n);
    indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indices
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
