//! Request-owned Fish hybrid backbone. The FP32 reference keeps its arithmetic;
//! BF16/W8A8 CUDA follows the pinned serving precision profile.
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
        #[cfg(feature = "cuda")]
        gpu: Option<Box<xrt_cuda::GpuOrderedDeltaState>>,
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
    rmsnorm_impl::<false>(input, weight, epsilon, zero_centered)
}

/// Stable FP32 mean for reduced-precision activation boundaries. Accumulating
/// the squared BF16 values in FP64 avoids serial-sum error crossing a BF16 tie;
/// the mean returns to FP32 before rsqrt and scaling, as in the reference graph.
pub(super) fn rmsnorm_stable(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    zero_centered: bool,
) -> Result<Vec<f32>> {
    rmsnorm_impl::<true>(input, weight, epsilon, zero_centered)
}

fn rmsnorm_impl<const STABLE: bool>(
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
    let mean = if STABLE {
        (input.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / input.len() as f64) as f32
    } else {
        input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32
    };
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

/// One causal decode row. BF16 serving attention rounds the unnormalized
/// exponential operand for the value product, retaining the FP32 denominator.
/// Rounding normalized probabilities instead is a different numerical graph.
pub(super) fn attention_core(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    numerics: super::numerics::Numerics,
) -> Result<Vec<f32>> {
    let width = kv_heads
        .checked_mul(dim)
        .filter(|v| *v > 0)
        .ok_or_else(|| XrtError::Shape("attention KV width".into()))?;
    let query_width = heads
        .checked_mul(dim)
        .filter(|v| *v > 0)
        .ok_or_else(|| XrtError::Shape("attention query width".into()))?;
    if heads % kv_heads != 0
        || query.len() != query_width
        || keys.is_empty()
        || keys.len() % width != 0
        || values.len() != keys.len()
    {
        return Err(XrtError::Shape("attention input geometry".into()));
    }
    let mut out = vec![0.0; query.len()];
    for head in 0..heads {
        let kh = head / (heads / kv_heads);
        let q = &query[head * dim..][..dim];
        let mut scores: Vec<f32> = keys
            .chunks_exact(width)
            .map(|row| {
                q.iter()
                    .zip(&row[kh * dim..][..dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (dim as f32).sqrt()
            })
            .collect();
        finite(&scores)?;
        let denominator = if numerics == super::numerics::Numerics::Bf16W8a8 {
            for s in &mut scores {
                *s *= std::f32::consts::LOG2_E;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = (*s - maximum).exp2();
            }
            let denominator = scores.iter().sum::<f32>();
            numerics.round_checked(&mut scores)?;
            denominator
        } else {
            softmax_inplace(&mut scores);
            1.0
        };
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(XrtError::InvalidTensor("attention denominator".into()));
        }
        for (row, score) in values.chunks_exact(width).zip(&scores) {
            for d in 0..dim {
                out[head * dim + d] += score * row[kh * dim + d];
            }
        }
        if numerics == super::numerics::Numerics::Bf16W8a8 {
            let inverse = 1.0 / denominator;
            for x in &mut out[head * dim..][..dim] {
                *x *= inverse;
            }
        }
    }
    numerics.round_checked(&mut out)?;
    Ok(out)
}

impl<'a> BackboneSession<'a> {
    pub fn new(checkpoint: &'a Checkpoint, max_rows: usize, budget_bytes: u64) -> Result<Self> {
        checkpoint.config.validate()?;
        if checkpoint
            .attention_context_limit()
            .is_some_and(|limit| max_rows > limit)
        {
            return Err(XrtError::Shape(
                "BF16 CUDA attention supports at most 8192 combined prompt and generation rows"
                    .into(),
            ));
        }
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
                    #[cfg(feature = "cuda")]
                    gpu: checkpoint.ordered_delta_state(budget_bytes)?,
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
                LayerState::Delta {
                    recurrent,
                    conv,
                    #[cfg(feature = "cuda")]
                    gpu,
                } => {
                    recurrent.reset();
                    conv.fill(0.0);
                    #[cfg(feature = "cuda")]
                    if let Some(gpu) = gpu {
                        if let Err(e) = gpu.reset() {
                            tracing::warn!(error=%e,"native delta reset failed");
                        }
                    }
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
                    let rotary =
                        (t.head_dim as f32 * t.rope_parameters.partial_rotary_factor) as usize;
                    if !store.normalize_rotate_qk(
                        &mut query,
                        &mut key,
                        &q_weight,
                        &k_weight,
                        xrt_cuda::Bf16RopeSpec {
                            head_dim: t.head_dim,
                            rotary_dim: rotary,
                            theta: t.rope_parameters.rope_theta,
                            position: self.position,
                            interleaved: false,
                            epsilon: t.rms_norm_eps,
                        },
                    )? {
                        for q in query.chunks_exact_mut(t.head_dim) {
                            q.copy_from_slice(&store.norm_with_rows(
                                q,
                                &q_weight,
                                t.rms_norm_eps,
                                true,
                                t.num_attention_heads,
                            )?);
                        }
                        for k in key.chunks_exact_mut(t.head_dim) {
                            k.copy_from_slice(&store.norm_with_rows(
                                k,
                                &k_weight,
                                t.rms_norm_eps,
                                true,
                                t.num_key_value_heads,
                            )?);
                        }
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
                    }
                    store.numerics.round_slice(&mut query);
                    store.numerics.round_slice(&mut key);
                    keys.extend_from_slice(&key);
                    values.extend_from_slice(&value);
                    let mut attended = store.attention_core(
                        &query,
                        keys,
                        values,
                        t.num_attention_heads,
                        t.num_key_value_heads,
                        t.head_dim,
                    )?;
                    store.attention_gate(&mut attended, &gate)?;
                    store.linear(&format!("{p}.o_proj.weight"), &attended)?
                }
                LayerState::Delta {
                    recurrent,
                    conv,
                    #[cfg(feature = "cuda")]
                    gpu,
                } => {
                    let p = format!("{prefix}.linear_attn");
                    let mixed = store.linear(&format!("{p}.in_proj_qkv.weight"), &normalized)?;
                    let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
                    let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;
                    let channels = 2 * key_dim + value_dim;
                    if mixed.len() != channels {
                        return Err(XrtError::Shape("Fish DeltaNet QKV geometry".into()));
                    }
                    let kernel = store.values(&format!("{p}.conv1d.weight"))?;
                    #[cfg(feature = "drama3-diagnostics")]
                    let conv_input = [mixed.as_slice(), conv.as_slice()].concat();
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
                    #[cfg(feature = "drama3-diagnostics")]
                    store.record(
                        "gdn_conv",
                        &p,
                        &conv_input,
                        &kernel,
                        &filtered,
                        serde_json::json!({"channels":channels,"kernel":t.linear_conv_kernel_dim}),
                    )?;
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
                    #[allow(unused_mut)]
                    let mut beta: Vec<_> = b.iter().map(|x| sigmoid(*x)).collect();
                    #[allow(unused_mut)]
                    let mut decay: Vec<_> = a
                        .iter()
                        .zip(&bias)
                        .zip(&log_a)
                        .map(|((x, b), a)| -a.exp() * softplus(x + b))
                        .collect();
                    #[cfg(feature = "drama3-diagnostics")]
                    let previous_state = if store.is_recording() {
                        #[cfg(feature = "cuda")]
                        let snapshot = if let Some(gpu) = gpu.as_ref() {
                            gpu.snapshot_value_major()?
                        } else {
                            recurrent.snapshot().to_vec()
                        };
                        #[cfg(not(feature = "cuda"))]
                        let snapshot = recurrent.snapshot().to_vec();
                        Some(snapshot)
                    } else {
                        None
                    };
                    let mut core = vec![0.0; value_dim];
                    #[cfg(feature = "cuda")]
                    let accelerated = if let Some(gpu) = gpu.as_mut() {
                        let result = gpu.step(&filtered, &a, &b, &bias, &log_a)?;
                        core = result.core;
                        beta = result.beta;
                        decay = result.log_decay;
                        true
                    } else {
                        false
                    };
                    #[cfg(not(feature = "cuda"))]
                    let accelerated = false;
                    if !accelerated {
                        let step = if cfg!(feature = "drama3-diagnostics")
                            && std::env::var_os("XRT_DRAMA3_DIAGNOSTIC_ORDERED_STATE").as_deref()
                                == Some(std::ffi::OsStr::new("1"))
                            && store.numerics == super::numerics::Numerics::Bf16W8a8
                        {
                            GatedDeltaState::step_ordered
                        } else {
                            GatedDeltaState::step
                        };
                        step(
                            recurrent,
                            &filtered[..key_dim],
                            &filtered[key_dim..2 * key_dim],
                            &filtered[2 * key_dim..],
                            &decay,
                            &beta,
                            &mut core,
                        )?;
                    }
                    store.numerics.round_slice(&mut core);
                    #[cfg(feature = "drama3-diagnostics")]
                    if let Some(previous) = previous_state {
                        store.record(
                            "gdn_controls",
                            &p,
                            &[a.as_slice(), b.as_slice()].concat(),
                            &[bias.as_slice(), log_a.as_slice()].concat(),
                            &[beta.as_slice(), decay.as_slice()].concat(),
                            serde_json::json!({}),
                        )?;
                        #[cfg(feature = "cuda")]
                        let snapshot = if let Some(gpu) = gpu.as_ref() {
                            gpu.snapshot_value_major()?
                        } else {
                            recurrent.snapshot().to_vec()
                        };
                        #[cfg(not(feature = "cuda"))]
                        let snapshot = recurrent.snapshot().to_vec();
                        store.record("gdn_recurrent",&p,&filtered,&[beta.as_slice(),decay.as_slice(),previous.as_slice()].concat(),
                            &[core.as_slice(),snapshot.as_slice()].concat(),serde_json::json!({"position":self.position,
                            "key_heads":t.linear_num_key_heads,"value_heads":t.linear_num_value_heads,"key_dim":t.linear_key_head_dim,"value_dim":t.linear_value_head_dim}))?;
                    }
                    let weights = store.vector(&format!("{p}.norm.weight"))?;
                    #[cfg(feature = "drama3-diagnostics")]
                    let gated_input = core.clone();
                    if let Some(normalized) =
                        store.gated_norm_cuda(&core, &z, &weights, t.rms_norm_eps)?
                    {
                        core = normalized;
                    } else {
                        for (row, g) in core
                            .chunks_exact_mut(t.linear_value_head_dim)
                            .zip(z.chunks_exact(t.linear_value_head_dim))
                        {
                            if store.numerics == super::numerics::Numerics::Bf16W8a8 {
                                // Upstream rounds the normalized value before the BF16
                                // weight multiply, then rounds that product separately.
                                row.copy_from_slice(&store.norm_with_rows(
                                    row,
                                    &vec![1.0; row.len()],
                                    t.rms_norm_eps,
                                    false,
                                    t.linear_num_value_heads,
                                )?);
                                store.numerics.round_checked(row)?;
                                for (x, w) in row.iter_mut().zip(&weights) {
                                    *x = store.numerics.round(*x * w);
                                }
                            } else {
                                row.copy_from_slice(&rmsnorm(
                                    row,
                                    &weights,
                                    t.rms_norm_eps,
                                    false,
                                )?);
                            }
                            for (x, g) in row.iter_mut().zip(g) {
                                *x *= if store.numerics == super::numerics::Numerics::Bf16W8a8 {
                                    *g / (1.0 + (-*g).exp())
                                } else {
                                    silu(*g)
                                };
                            }
                        }
                    }
                    store.numerics.round_slice(&mut core);
                    #[cfg(feature = "drama3-diagnostics")]
                    store.record("gdn_gate",&p,&gated_input,&[weights.as_slice(),z.as_slice()].concat(),&core,
                        serde_json::json!({"epsilon":t.rms_norm_eps,"head_dim":t.linear_value_head_dim}))?;
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
    if let Some(out) = store.router_softmax(&probabilities)? {
        probabilities = out;
    } else {
        softmax_inplace(&mut probabilities);
    }
    let mut order: Vec<_> = (0..probabilities.len()).collect();
    order.sort_by(|a, b| {
        probabilities[*b]
            .total_cmp(&probabilities[*a])
            .then(a.cmp(b))
    });
    order.truncate(cfg.num_experts_per_tok);
    let denominator = if cfg.norm_topk_prob {
        if store.numerics == super::numerics::Numerics::Bf16W8a8 && order.len() == 8 {
            let mut sums: Vec<_> = order.iter().map(|i| probabilities[*i]).collect();
            for stride in [4, 2, 1] {
                for i in 0..stride {
                    sums[i] += sums[i + stride];
                }
            }
            sums[0]
        } else {
            order.iter().map(|i| probabilities[*i]).sum::<f32>()
        }
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
    if let Some(out) = store.sigmoid_gate_add_cuda(&shared, &gate, Some(&result))? {
        result = out;
    } else {
        for (sum, x) in result.iter_mut().zip(shared) {
            *sum = store.numerics.round(
                *sum + store
                    .numerics
                    .round(store.numerics.round(sigmoid(gate[0])) * x),
            );
        }
    }
    finite(&result)?;
    Ok(result)
}

#[cfg(test)]
mod attention_tests {
    use super::super::numerics::Numerics;
    use super::attention_core;
    #[test]
    fn bf16_attention_matches_pinned_cuda_operand_rounding() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../reference/audio/drama3/bf16-attention-fixture.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let values = |name: &str| {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let actual = attention_core(
                &values("query"),
                &values("keys"),
                &values("values"),
                case["heads"].as_u64().unwrap() as usize,
                case["kv_heads"].as_u64().unwrap() as usize,
                case["dim"].as_u64().unwrap() as usize,
                Numerics::Bf16W8a8,
            )
            .unwrap();
            let expected = values("expected");
            let squared: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum();
            let reference: f64 = expected.iter().map(|v| (*v as f64).powi(2)).sum();
            assert!(
                (squared / reference).sqrt() < 2e-4,
                "CUDA attention rounding mismatch: rows {}",
                case["rows"]
            );
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() <= 0.002 + 0.001 * b.abs());
            }
        }
        assert!(attention_core(&[], &[], &[], usize::MAX, 2, 4, Numerics::Bf16W8a8).is_err());
        assert!(attention_core(
            &[0.0; 2],
            &[0.0; 2],
            &[f32::NAN; 2],
            1,
            1,
            2,
            Numerics::Bf16W8a8
        )
        .is_err());
    }
}
