use serde::Deserialize;
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub fish_schema_version: u32,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub eos_token_id: u32,
    pub audio_pad_token_id: u32,
    pub speech_template_version: String,
    pub text_config: TextConfig,
    pub audio_decoder_config: AudioConfig,
    pub quantization_config: QuantizationConfig,
}
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_act: String,
    pub attn_output_gate: bool,
    pub use_qk_norm: bool,
    pub layernorm_zero_centered_gamma: bool,
    pub shared_expert_gate: bool,
    pub mlp_layer_types: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeConfig,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
}
fn yes() -> bool {
    true
}
#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rope_type: String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub layernorm_zero_centered_gamma: bool,
    pub attention_output_gate: bool,
    pub rotary_interleaved: bool,
    pub dim: usize,
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub shared_enrichment_layers: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub vq_backbone_codebooks: usize,
    pub decoder_type: String,
    pub rope_base: f32,
    pub rotary_percent: f32,
}
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub weight_block_size: [usize; 2],
    pub activation_scheme: String,
}

impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1024 * 1024 {
            return Err(XrtError::InvalidMetadata(
                "Fish config exceeds 1 MiB".into(),
            ));
        }
        let config: Self =
            serde_json::from_slice(bytes).map_err(|e| XrtError::InvalidMetadata(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        let t = &self.text_config;
        let a = &self.audio_decoder_config;
        let bad = || XrtError::InvalidMetadata("unsupported or invalid Fish Omni geometry".into());
        if t.model_type != "fish_omni_text"
            || t.hidden_act != "silu"
            || !t.attn_output_gate
            || !t.use_qk_norm
            || !t.layernorm_zero_centered_gamma
            || !t.shared_expert_gate
            || t.mlp_layer_types.len() != t.num_hidden_layers
            || t.mlp_layer_types.iter().any(|s| s != "moe")
            || !a.layernorm_zero_centered_gamma
            || !a.attention_output_gate
            || !a.rotary_interleaved
        {
            return Err(bad());
        }
        if self.model_type != "fish_omni"
            || self.fish_schema_version != 1
            || self.speech_template_version != "reference-bracket-v1"
            || self.quantization_config.quant_method != "bsgemm-moe"
            || self.quantization_config.weight_block_size != [128, 128]
            || self.quantization_config.activation_scheme != "dynamic"
            || a.decoder_type != "codebook_mtp_shared"
        {
            return Err(bad());
        }
        for n in [
            t.hidden_size,
            t.num_hidden_layers,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            t.vocab_size,
            t.num_experts,
            t.num_experts_per_tok,
            t.moe_intermediate_size,
            t.shared_expert_intermediate_size,
            t.max_position_embeddings,
            t.linear_num_key_heads,
            t.linear_num_value_heads,
            t.linear_key_head_dim,
            t.linear_value_head_dim,
            t.linear_conv_kernel_dim,
            a.dim,
            a.num_codebooks,
            a.vocab_size,
            a.shared_enrichment_layers,
            a.n_head,
            a.n_local_heads,
            a.head_dim,
        ] {
            if n == 0 {
                return Err(bad());
            }
        }
        if t.hidden_size > 65536
            || t.num_hidden_layers > 256
            || t.num_experts > 4096
            || t.vocab_size > 1_000_000
            || t.max_position_embeddings > 1_048_576
            || a.num_codebooks > 64
            || a.vocab_size > 1_000_000
            || t.head_dim > 2048
            || a.head_dim > 2048
            || t.moe_intermediate_size > 65536
            || t.shared_expert_intermediate_size > 65536
            || t.linear_conv_kernel_dim > 64
            || t.num_attention_heads > 1024
            || a.n_head > 1024
            || t.linear_key_head_dim > 2048
            || t.linear_value_head_dim > 2048
            || t.linear_num_value_heads > 1024
            || t.num_experts_per_tok > t.num_experts
            || t.layer_types.len() != t.num_hidden_layers
            || t.layer_types
                .iter()
                .any(|s| s != "linear_attention" && s != "full_attention")
            || t.num_attention_heads % t.num_key_value_heads != 0
            || t.linear_num_value_heads % t.linear_num_key_heads != 0
            || a.n_head % a.n_local_heads != 0
            || a.dim != t.hidden_size
            || a.vq_backbone_codebooks >= a.num_codebooks
            || !t.rms_norm_eps.is_finite()
            || t.rms_norm_eps <= 0.0
            || !a.norm_eps.is_finite()
            || a.norm_eps <= 0.0
            || self
                .semantic_end_token_id
                .checked_sub(self.semantic_start_token_id)
                != Some(a.vocab_size as u32 - 1)
            || self.semantic_end_token_id as usize >= t.vocab_size
            || self.eos_token_id as usize >= t.vocab_size
            || self.audio_pad_token_id as usize >= t.vocab_size
            || t.rope_parameters.rope_type != "default"
            || !t.rope_parameters.rope_theta.is_finite()
            || t.rope_parameters.rope_theta <= 0.0
            || !t.rope_parameters.partial_rotary_factor.is_finite()
            || !(0.0..=1.0).contains(&t.rope_parameters.partial_rotary_factor)
            || ((t.head_dim as f32 * t.rope_parameters.partial_rotary_factor) as usize) % 2 != 0
            || !a.rope_base.is_finite()
            || a.rope_base <= 0.0
            || !a.rotary_percent.is_finite()
            || !(0.0..=1.0).contains(&a.rotary_percent)
            || ((a.head_dim as f32 * a.rotary_percent) as usize) % 2 != 0
        {
            return Err(bad());
        }
        Ok(())
    }
    /// Session state only; weights, codec, convolution state, scratch and
    /// staging caches need separate reservations. No GPU fit claim is made.
    pub fn state_bytes(&self, rows: usize) -> Result<u64> {
        if rows == 0 || rows > self.text_config.max_position_embeddings {
            return Err(XrtError::Shape(
                "Fish context exceeds configured bounds".into(),
            ));
        }
        let mul = |a: u64, b: u64| {
            a.checked_mul(b)
                .ok_or_else(|| XrtError::Shape("Fish state budget overflow".into()))
        };
        let t = &self.text_config;
        let a = &self.audio_decoder_config;
        let full = t
            .layer_types
            .iter()
            .filter(|x| *x == "full_attention")
            .count() as u64;
        let linear = t.num_hidden_layers as u64 - full;
        let kv = mul(
            mul(
                mul(mul(full, rows as u64)?, t.num_key_value_heads as u64)?,
                t.head_dim as u64,
            )?,
            8,
        )?;
        let delta = mul(
            mul(
                mul(
                    mul(linear, t.linear_num_value_heads as u64)?,
                    t.linear_key_head_dim as u64,
                )?,
                t.linear_value_head_dim as u64,
            )?,
            4,
        )?;
        let tower_layers = a
            .shared_enrichment_layers
            .checked_add(a.num_codebooks - 1)
            .ok_or_else(|| XrtError::Shape("tower layer overflow".into()))?;
        let tower = mul(
            mul(
                mul(
                    mul(tower_layers as u64, rows as u64)?,
                    a.n_local_heads as u64,
                )?,
                a.head_dim as u64,
            )?,
            8,
        )?;
        kv.checked_add(delta)
            .and_then(|v| v.checked_add(tower))
            .ok_or_else(|| XrtError::Shape("Fish state sum overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_config_parses_and_oversized_context_is_refused() {
        let c = Config::parse(include_bytes!(
            "../../../../reference/audio/drama3/config.json"
        ))
        .unwrap();
        assert_eq!(c.text_config.num_experts, 256);
        assert_eq!(c.audio_decoder_config.num_codebooks, 10);
        assert!(c.state_bytes(8192).unwrap() > 60 * 1024 * 1024);
        assert!(c.state_bytes(usize::MAX).is_err());
    }
    #[test]
    fn invalid_heads_and_quantization_fail_before_loading() {
        let mut c = Config::parse(include_bytes!(
            "../../../../reference/audio/drama3/config.json"
        ))
        .unwrap();
        c.text_config.linear_num_value_heads = 31;
        assert!(c.validate().is_err());
        c.text_config.linear_num_value_heads = 32;
        c.quantization_config.weight_block_size = [64, 64];
        assert!(c.validate().is_err());
    }
}
