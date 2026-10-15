//! Request-owned Fish hybrid backbone. Precision: FP32 activations/state with
//! encoded BF16/FP8 weights. This is a CPU reference, not BF16 bit parity.
use super::checkpoint::Checkpoint;
use crate::control::InferenceControl;
use xrt_core::{Result, XrtError};
use xrt_kernels::cpu::{
    delta_state::{GatedDeltaState, HeadExpansion},
    softmax_inplace,
};

enum LayerState {
    Full {
        keys: Vec<f32>,
        values: Vec<f32>,
    },
    Delta {
        recurrent: GatedDeltaState,
        conv: Vec<f32>,
    },
}
pub struct BackboneSession<'a> {
    checkpoint: &'a Checkpoint,
    layers: Vec<LayerState>,
    position: usize,
    max_rows: usize,
    reserved_bytes: u64,
}

pub fn rmsnorm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    zero_centered: bool,
) -> Result<Vec<f32>> {
    if input.is_empty()
        || input.len() != weight.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.iter().chain(weight).any(|x| !x.is_finite())
    {
        return Err(XrtError::InvalidTensor(
            "invalid normalization input".into(),
        ));
    }
    let mean = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    if !mean.is_finite() {
        return Err(XrtError::InvalidTensor("normalization sum overflow".into()));
    }
    let inv = 1.0 / (mean + epsilon).sqrt();
    let result: Vec<_> = input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * inv * (w + if zero_centered { 1.0 } else { 0.0 }))
        .collect();
    finite(&result)?;
    Ok(result)
}
fn finite(input: &[f32]) -> Result<()> {
    if input.iter().any(|x| !x.is_finite()) {
        Err(XrtError::InvalidTensor(
            "non-finite Fish graph activation".into(),
        ))
    } else {
        Ok(())
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(XrtError::Shape("Fish residual dimensions".into()));
    }
    let out: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
    finite(&out)?;
    Ok(out)
}
fn rope(x: &mut [f32], head_dim: usize, rotary: usize, position: usize, theta: f32) {
    if rotary == 0 {
        return;
    }
    for head in x.chunks_exact_mut(head_dim) {
        for pair in 0..rotary / 2 {
            let angle = position as f32 / theta.powf((2 * pair) as f32 / rotary as f32);
            let (sin, cos) = angle.sin_cos();
            let a = head[pair];
            let b = head[pair + rotary / 2];
            head[pair] = a * cos - b * sin;
            head[pair + rotary / 2] = b * cos + a * sin;
        }
    }
}

impl<'a> BackboneSession<'a> {
    pub fn new(checkpoint: &'a Checkpoint, max_rows: usize, budget_bytes: u64) -> Result<Self> {
        checkpoint.config.validate()?;
        let t = &checkpoint.config.text_config;
        let c = t
            .linear_num_key_heads
            .checked_mul(t.linear_key_head_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(t.linear_num_value_heads * t.linear_value_head_dim))
            .ok_or_else(|| XrtError::Shape("Fish convolution overflow".into()))?;
        let convolution_bytes = (c as u64)
            .checked_mul(t.linear_conv_kernel_dim as u64)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| {
                v.checked_mul(
                    t.layer_types
                        .iter()
                        .filter(|x| *x == "linear_attention")
                        .count() as u64,
                )
            })
            .ok_or_else(|| XrtError::Shape("convolution state budget overflow".into()))?;
        let reserved_bytes = checkpoint
            .config
            .state_bytes(max_rows)?
            .checked_add(convolution_bytes)
            .ok_or_else(|| XrtError::Shape("Fish session budget overflow".into()))?;
        if reserved_bytes > budget_bytes {
            return Err(XrtError::Runtime(format!(
                "Fish state needs {reserved_bytes} bytes, budget {budget_bytes}"
            )));
        }
        let mut layers = Vec::new();
        for kind in &t.layer_types {
            layers.push(if kind == "full_attention" {
                LayerState::Full {
                    keys: reserve_cache(
                        max_rows
                            .checked_mul(t.num_key_value_heads)
                            .and_then(|n| n.checked_mul(t.head_dim))
                            .ok_or_else(|| XrtError::Shape("Fish KV cache size overflow".into()))?,
                    )?,
                    values: reserve_cache(
                        max_rows
                            .checked_mul(t.num_key_value_heads)
                            .and_then(|n| n.checked_mul(t.head_dim))
                            .ok_or_else(|| XrtError::Shape("Fish KV cache size overflow".into()))?,
                    )?,
                }
            } else {
                LayerState::Delta {
                    recurrent: GatedDeltaState::new(
                        t.linear_num_key_heads,
                        t.linear_num_value_heads,
                        t.linear_key_head_dim,
                        t.linear_value_head_dim,
                        HeadExpansion::RepeatInterleave,
                        budget_bytes.try_into().unwrap_or(usize::MAX),
                    )?,
                    conv: vec![0.0; c * (t.linear_conv_kernel_dim - 1)],
                }
            });
        }
        Ok(Self {
            checkpoint,
            layers,
            position: 0,
            max_rows,
            reserved_bytes,
        })
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        for state in &mut self.layers {
            match state {
                LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
                LayerState::Delta { recurrent, conv } => {
                    recurrent.reset();
                    conv.fill(0.0);
                }
            }
        }
        self.position = 0;
    }
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        self.checkpoint
            .row("model.language_model.embed_tokens.weight", token as usize)
    }
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.checkpoint.linear("lm_head.weight", hidden)
    }
    pub fn semantic_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        // The trained generation domain permits only EOS and the semantic
        // band. Uncomputed vocabulary rows are explicitly masked, never zero.
        let cfg = &self.checkpoint.config;
        let mut out = vec![f32::NEG_INFINITY; cfg.text_config.vocab_size];
        let band = self.checkpoint.linear_rows(
            "lm_head.weight",
            hidden,
            cfg.semantic_start_token_id as usize,
            cfg.audio_decoder_config.vocab_size,
        )?;
        out[cfg.semantic_start_token_id as usize..=cfg.semantic_end_token_id as usize]
            .copy_from_slice(&band);
        out[cfg.eos_token_id as usize] =
            self.checkpoint
                .linear_rows("lm_head.weight", hidden, cfg.eos_token_id as usize, 1)?[0];
        Ok(out)
    }
    pub fn step(&mut self, embedding: &[f32], control: &InferenceControl) -> Result<Vec<f32>> {
        self.step_traced(embedding, control, |_, _, _| {})
    }
    /// Caller-owned numerical qualification trace; no telemetry/text is emitted
    /// by the ordinary execution path. Each callback sees one layer's residual.
    pub fn step_traced(
        &mut self,
        embedding: &[f32],
        control: &InferenceControl,
        mut trace: impl FnMut(usize, bool, &[f32]),
    ) -> Result<Vec<f32>> {
        if self.position >= self.max_rows
            || embedding.len() != self.checkpoint.config.text_config.hidden_size
        {
            return Err(XrtError::Shape(
                "Fish backbone context/input dimensions".into(),
            ));
        }
        finite(embedding)?;
        let result = self.forward(embedding, control, &mut trace);
        match result {
            Ok(hidden) => {
                self.position += 1;
                Ok(hidden)
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }
    fn forward(
        &mut self,
        input: &[f32],
        control: &InferenceControl,
        trace: &mut impl FnMut(usize, bool, &[f32]),
    ) -> Result<Vec<f32>> {
        let t = &self.checkpoint.config.text_config;
        let mut hidden = input.to_vec();
        self.checkpoint.numerics.round_slice(&mut hidden);
        let store = self.checkpoint;
        for (layer, state) in self.layers.iter_mut().enumerate() {
            trace(layer, false, &hidden);
            control
                .check()
                .map_err(|e| XrtError::Runtime(e.to_string()))?;
            let prefix = format!("model.language_model.layers.{layer}");
            let normalized = store.norm(
                &hidden,
                &store.vector(&format!("{prefix}.input_layernorm.weight"))?,
                t.rms_norm_eps,
                true,
            )?;
            let attention = match state {
                LayerState::Full { keys, values } => {
                    let p = format!("{prefix}.self_attn");
                    let packed = store.linear(&format!("{p}.q_proj.weight"), &normalized)?;
                    if packed.len() != t.num_attention_heads * t.head_dim * 2 {
                        return Err(XrtError::Shape("Fish query/gate packing".into()));
                    }
                    let mut query = Vec::with_capacity(packed.len() / 2);
                    let mut gate = Vec::with_capacity(packed.len() / 2);
                    for pair in packed.chunks_exact(t.head_dim * 2) {
                        query.extend_from_slice(&pair[..t.head_dim]);
                        gate.extend_from_slice(&pair[t.head_dim..]);
                    }
                    let mut key = store.linear(&format!("{p}.k_proj.weight"), &normalized)?;
                    let value = store.linear(&format!("{p}.v_proj.weight"), &normalized)?;
                    if key.len() != t.num_key_value_heads * t.head_dim || value.len() != key.len() {
                        return Err(XrtError::Shape("Fish attention KV geometry".into()));
                    }
                    let q_weight = store.vector(&format!("{p}.q_norm.weight"))?;
                    let k_weight = store.vector(&format!("{p}.k_norm.weight"))?;
                    for q in query.chunks_exact_mut(t.head_dim) {
                        q.copy_from_slice(&store.norm(q, &q_weight, t.rms_norm_eps, true)?);
                    }
                    for k in key.chunks_exact_mut(t.head_dim) {
                        k.copy_from_slice(&store.norm(k, &k_weight, t.rms_norm_eps, true)?);
                    }
                    let rotary =
                        (t.head_dim as f32 * t.rope_parameters.partial_rotary_factor) as usize;
                    rope(
                        &mut query,
                        t.head_dim,
                        rotary,
                        self.position,
                        t.rope_parameters.rope_theta,
                    );
                    rope(
                        &mut key,
                        t.head_dim,
                        rotary,
                        self.position,
                        t.rope_parameters.rope_theta,
                    );
                    store.numerics.round_slice(&mut query);
                    store.numerics.round_slice(&mut key);
                    keys.extend_from_slice(&key);
                    values.extend_from_slice(&value);
                    let width = t.num_key_value_heads * t.head_dim;
                    let mut attended = vec![0.0; query.len()];
                    for head in 0..t.num_attention_heads {
                        let kh = head / (t.num_attention_heads / t.num_key_value_heads);
                        let q = &query[head * t.head_dim..][..t.head_dim];
                        let mut scores: Vec<_> = (0..=self.position)
                            .map(|row| {
                                q.iter()
                                    .zip(&keys[row * width + kh * t.head_dim..][..t.head_dim])
                                    .map(|(a, b)| a * b)
                                    .sum::<f32>()
                                    / (t.head_dim as f32).sqrt()
                            })
                            .collect();
                        finite(&scores)?;
                        softmax_inplace(&mut scores);
                        for (row, score) in scores.iter().enumerate() {
                            for d in 0..t.head_dim {
                                attended[head * t.head_dim + d] +=
                                    score * values[row * width + kh * t.head_dim + d];
                            }
                        }
                    }
                    store.attention_gate(&mut attended, &gate)?;
                    store.linear(&format!("{p}.o_proj.weight"), &attended)?
                }
                LayerState::Delta { recurrent, conv } => {
                    let p = format!("{prefix}.linear_attn");
                    let mixed = store.linear(&format!("{p}.in_proj_qkv.weight"), &normalized)?;
                    let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
                    let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;
                    let channels = 2 * key_dim + value_dim;
                    if mixed.len() != channels {
                        return Err(XrtError::Shape("Fish DeltaNet QKV geometry".into()));
                    }
                    let kernel = store.values(&format!("{p}.conv1d.weight"))?;
                    let history = t.linear_conv_kernel_dim - 1;
                    if kernel.len() != channels * t.linear_conv_kernel_dim {
                        return Err(XrtError::Shape("Fish convolution weight geometry".into()));
                    }
                    let mut filtered = vec![0.0; channels];
                    for channel in 0..channels {
                        let past = &mut conv[channel * history..][..history];
                        let weights = &kernel[channel * t.linear_conv_kernel_dim..]
                            [..t.linear_conv_kernel_dim];
                        let mixed_value = past.iter().zip(weights).map(|(a, b)| a * b).sum::<f32>()
                            + mixed[channel] * weights[history];
                        filtered[channel] = store
                            .numerics
                            .round(silu(store.numerics.round(mixed_value)));
                        if history > 0 {
                            past.rotate_left(1);
                            past[history - 1] = mixed[channel];
                        }
                    }
                    let z = store.linear(&format!("{p}.in_proj_z.weight"), &normalized)?;
                    let b = store.linear(&format!("{p}.in_proj_b.weight"), &normalized)?;
                    let a = store.linear(&format!("{p}.in_proj_a.weight"), &normalized)?;
                    let bias = store.vector(&format!("{p}.dt_bias"))?;
                    let log_a = store.vector(&format!("{p}.A_log"))?;
                    if b.len() != t.linear_num_value_heads
                        || a.len() != b.len()
                        || bias.len() != b.len()
                        || log_a.len() != b.len()
                        || z.len() != value_dim
                    {
                        return Err(XrtError::Shape("Fish delta control geometry".into()));
                    }
                    let beta: Vec<_> = b.iter().map(|x| sigmoid(*x)).collect();
                    let decay: Vec<_> = a
                        .iter()
                        .zip(bias)
                        .zip(log_a)
                        .map(|((x, b), a)| -a.exp() * softplus(x + b))
                        .collect();
                    let mut core = vec![0.0; value_dim];
                    recurrent.step(
                        &filtered[..key_dim],
                        &filtered[key_dim..2 * key_dim],
                        &filtered[2 * key_dim..],
                        &decay,
                        &beta,
                        &mut core,
                    )?;
                    store.numerics.round_slice(&mut core);
                    let weights = store.vector(&format!("{p}.norm.weight"))?;
                    for (row, g) in core
                        .chunks_exact_mut(t.linear_value_head_dim)
                        .zip(z.chunks_exact(t.linear_value_head_dim))
                    {
                        if store.numerics == super::numerics::Numerics::Bf16W8a8 {
                            // Upstream rounds the normalized value before the BF16
                            // weight multiply, then rounds that product separately.
                            row.copy_from_slice(&rmsnorm(
                                row,
                                &vec![1.0; row.len()],
                                t.rms_norm_eps,
                                false,
                            )?);
                            store.numerics.round_checked(row)?;
                            for (x, w) in row.iter_mut().zip(&weights) {
                                *x = store.numerics.round(*x * w);
                            }
                        } else {
                            row.copy_from_slice(&rmsnorm(row, &weights, t.rms_norm_eps, false)?);
                        }
                        for (x, g) in row.iter_mut().zip(g) {
                            *x *= silu(*g);
                        }
                    }
                    store.numerics.round_slice(&mut core);
                    store.linear(&format!("{p}.out_proj.weight"), &core)?
                }
            };
            hidden = add(&hidden, &attention)?;
            store.numerics.round_slice(&mut hidden);
            let normalized = store.norm(
                &hidden,
                &store.vector(&format!("{prefix}.post_attention_layernorm.weight"))?,
                t.rms_norm_eps,
                true,
            )?;
            hidden = add(
                &hidden,
                &moe(store, &format!("{prefix}.mlp"), &normalized, control)?,
            )?;
            store.numerics.round_slice(&mut hidden);
            trace(layer, true, &hidden);
        }
        store.norm(
            &hidden,
            &store.vector("model.language_model.norm.weight")?,
            t.rms_norm_eps,
            true,
        )
    }
}
pub(super) fn reserve_cache(elements: usize) -> Result<Vec<f32>> {
    let mut cache = Vec::new();
    cache
        .try_reserve_exact(elements)
        .map_err(|e| XrtError::Runtime(format!("Fish cache allocation: {e}")))?;
    Ok(cache)
}
fn dense_mlp(store: &Checkpoint, prefix: &str, input: &[f32]) -> Result<Vec<f32>> {
    let gate = store.linear(&format!("{prefix}.gate_proj.weight"), input)?;
    let up = store.linear(&format!("{prefix}.up_proj.weight"), input)?;
    if gate.len() != up.len() {
        return Err(XrtError::Shape("Fish MLP gate/up dimensions".into()));
    }
    let hidden: Vec<_> = if store.numerics == super::numerics::Numerics::Fp32Reference {
        gate.iter().zip(up).map(|(g, u)| silu(*g) * u).collect()
    } else {
        store.mlp_hidden(&gate, &up, prefix.contains(".experts."))?
    };
    store.linear(&format!("{prefix}.down_proj.weight"), &hidden)
}
pub fn moe(
    store: &Checkpoint,
    prefix: &str,
    input: &[f32],
    control: &InferenceControl,
) -> Result<Vec<f32>> {
    let cfg = &store.config.text_config;
    let mut probabilities = store.linear(&format!("{prefix}.gate.weight"), input)?;
    if probabilities.len() != cfg.num_experts {
        return Err(XrtError::Shape("Fish router expert count".into()));
    }
    softmax_inplace(&mut probabilities);
    let mut order: Vec<_> = (0..probabilities.len()).collect();
    order.sort_by(|a, b| {
        probabilities[*b]
            .total_cmp(&probabilities[*a])
            .then(a.cmp(b))
    });
    order.truncate(cfg.num_experts_per_tok);
    let denominator = if cfg.norm_topk_prob {
        order.iter().map(|i| probabilities[*i]).sum::<f32>()
    } else {
        1.0
    };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(XrtError::InvalidTensor("Fish routing normalization".into()));
    }
    control
        .check()
        .map_err(|e| XrtError::Runtime(e.to_string()))?;
    #[cfg(feature = "cuda")]
    let gpu = store.routed_experts(
        prefix,
        &order,
        input,
        &order.iter().map(|i| probabilities[*i]).collect::<Vec<_>>(),
        denominator,
    )?;
    #[cfg(not(feature = "cuda"))]
    let gpu: Option<Vec<f32>> = None;
    let mut result = if let Some(out) = gpu {
        out
    } else {
        let mut result = vec![0.0; input.len()];
        for expert in order {
            control
                .check()
                .map_err(|e| XrtError::Runtime(e.to_string()))?;
            let out = dense_mlp(store, &format!("{prefix}.experts.{expert}"), input)?;
            if out.len() != result.len() {
                return Err(XrtError::Shape("Fish expert output dimensions".into()));
            }
            for (sum, x) in result.iter_mut().zip(out) {
                if store.numerics == super::numerics::Numerics::Bf16W8a8 {
                    *sum += x * store.numerics.round(probabilities[expert] / denominator);
                } else {
                    *sum += x * probabilities[expert] / denominator;
                }
            }
        }
        result
    };
    control
        .check()
        .map_err(|e| XrtError::Runtime(e.to_string()))?;
    store.numerics.round_slice(&mut result);
    let gate = store.linear(&format!("{prefix}.shared_expert_gate.weight"), input)?;
    if gate.len() != 1 {
        return Err(XrtError::Shape("Fish shared expert gate".into()));
    }
    let shared = dense_mlp(store, &format!("{prefix}.shared_expert"), input)?;
    if shared.len() != result.len() {
        return Err(XrtError::Shape("Fish shared expert dimensions".into()));
    }
    for (sum, x) in result.iter_mut().zip(shared) {
        *sum = store.numerics.round(
            *sum + store
                .numerics
                .round(store.numerics.round(sigmoid(gate[0])) * x),
        );
    }
    finite(&result)?;
    Ok(result)
}
