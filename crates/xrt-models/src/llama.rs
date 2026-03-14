use parking_lot::RwLock;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::info;
use xrt_core::{decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_kernels::cpu::{
    accumulate_scaled, add_inplace, apply_rmsnorm, dequantize_q4_0_row, dequantize_q4_k_row,
    dequantize_q5_k_row, dequantize_q6_k_row, dequantize_q8_0_row, dot, global_pool,
    matvec_quantized, matvec_quantized_batch, matvec_quantized_fused, softmax_inplace, swiglu,
    RopeFreqs,
};

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
pub struct LlamaConfig {
    pub architecture: String,
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
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .unwrap_or("llama")
            .to_string();
        if architecture != "llama" && architecture != "qwen3" {
            return Err(XrtError::Unsupported(format!(
                "xrt-models supports llama and qwen3 architectures, found {architecture}"
            )));
        }

        let prefix = architecture.as_str();
        let vocab_size = gguf
            .metadata_usize(&format!("{prefix}.vocab_size"))
            .or_else(|| {
                gguf.metadata_array("tokenizer.ggml.tokens")
                    .map(|array| array.len())
            })
            .ok_or_else(|| {
                XrtError::InvalidMetadata("missing llama vocab size metadata".to_string())
            })?;
        let context_length = required_usize(gguf, &format!("{prefix}.context_length"))?;
        let embedding_length = required_usize(gguf, &format!("{prefix}.embedding_length"))?;
        let feed_forward_length = required_usize(gguf, &format!("{prefix}.feed_forward_length"))?;
        let block_count = required_usize(gguf, &format!("{prefix}.block_count"))?;
        let attention_head_count = required_usize(gguf, &format!("{prefix}.attention.head_count"))?;
        let attention_head_count_kv = gguf
            .metadata_usize(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(attention_head_count);
        if attention_head_count == 0 || attention_head_count_kv == 0 {
            return Err(XrtError::InvalidMetadata(
                "attention head counts must be non-zero".to_string(),
            ));
        }
        if embedding_length % attention_head_count != 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "embedding length {embedding_length} is not divisible by attention head count {attention_head_count}"
            )));
        }
        if attention_head_count % attention_head_count_kv != 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "attention head count {attention_head_count} is not divisible by KV head count {attention_head_count_kv}"
            )));
        }

        let default_head_dim = embedding_length / attention_head_count;
        let head_dim_override = gguf
            .metadata_usize(&format!("{prefix}.attention.key_length"))
            .filter(|&dim| dim != default_head_dim);
        let actual_head_dim = head_dim_override.unwrap_or(default_head_dim);
        let rope_dimension_count = gguf
            .metadata_usize(&format!("{prefix}.rope.dimension_count"))
            .unwrap_or(actual_head_dim);
        let rms_norm_eps = gguf
            .metadata_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.metadata_f32(&format!("{prefix}.attention.layer_norm_epsilon")))
            .unwrap_or(1e-5);
        let rope_freq_base = gguf
            .metadata_f32(&format!("{prefix}.rope.freq_base"))
            .or_else(|| gguf.metadata_f32(&format!("{prefix}.rope.freq_base_train")))
            .unwrap_or(10000.0);
        let rope_freq_scale = gguf
            .metadata_f32(&format!("{prefix}.rope.scale_linear"))
            .or_else(|| gguf.metadata_f32(&format!("{prefix}.rope.scaling.factor")))
            .unwrap_or(1.0);

        Ok(Self {
            architecture,
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
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.embedding_length / self.attention_head_count)
    }

    pub fn q_width(&self) -> usize {
        self.attention_head_count * self.head_dim()
    }

    pub fn kv_width(&self) -> usize {
        self.attention_head_count_kv * self.head_dim()
    }
}

/// Pre-resolved tensor metadata to avoid HashMap lookups during forward pass.
/// Each forward_token call does 7 linear projections × 28 layers = 196 calls,
/// each requiring 2 HashMap lookups (require_tensor + tensor_data). Pre-resolving
/// eliminates ~400 string hash+compare operations per token.
#[derive(Debug, Clone, Copy)]
struct ResolvedWeight {
    /// Byte offset of this tensor's data within the GGUF data section.
    data_offset: usize,
    /// Total byte size of this tensor's data.
    nbytes: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm: String,
    attn_q: ResolvedWeight,
    attn_k: ResolvedWeight,
    attn_v: ResolvedWeight,
    attn_output: ResolvedWeight,
    ffn_norm: String,
    ffn_gate: ResolvedWeight,
    ffn_down: ResolvedWeight,
    ffn_up: ResolvedWeight,
    attn_q_norm: Option<String>,
    attn_k_norm: Option<String>,
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
}

impl ForwardScratch {
    fn new(config: &LlamaConfig) -> Self {
        let rope_dim = config.rope_dimension_count / 2;
        Self {
            normed: vec![0.0; config.embedding_length],
            q: vec![0.0; config.q_width()],
            k: vec![0.0; config.kv_width()],
            v: vec![0.0; config.kv_width()],
            gate: vec![0.0; config.feed_forward_length],
            up: vec![0.0; config.feed_forward_length],
            attn_out: vec![0.0; config.q_width()],
            proj: vec![0.0; config.embedding_length],
            down: vec![0.0; config.embedding_length],
            sin_cache: vec![0.0; rope_dim],
            cos_cache: vec![0.0; rope_dim],
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
    vector_cache: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    rope_freqs: RopeFreqs,
    scratch: RwLock<ForwardScratch>,
    batch_scratch: RwLock<BatchScratch>,
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
        for index in 0..config.block_count {
            let q_norm_name = format!("blk.{index}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{index}.attn_k_norm.weight");
            let has_qk_norm = gguf.tensor_info(&q_norm_name).is_some();
            let layer = LayerWeights {
                attn_norm: format!("blk.{index}.attn_norm.weight"),
                attn_q: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_q.weight"))?,
                attn_k: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_k.weight"))?,
                attn_v: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_v.weight"))?,
                attn_output: Self::resolve_weight(&gguf, &format!("blk.{index}.attn_output.weight"))?,
                ffn_norm: format!("blk.{index}.ffn_norm.weight"),
                ffn_gate: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_gate.weight"))?,
                ffn_down: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_down.weight"))?,
                ffn_up: Self::resolve_weight(&gguf, &format!("blk.{index}.ffn_up.weight"))?,
                attn_q_norm: if has_qk_norm { Some(q_norm_name) } else { None },
                attn_k_norm: if has_qk_norm { Some(k_norm_name) } else { None },
            };
            gguf.require_tensor(&layer.attn_norm)?;
            gguf.require_tensor(&layer.ffn_norm)?;
            if let Some(ref name) = layer.attn_q_norm {
                gguf.require_tensor(name)?;
            }
            if let Some(ref name) = layer.attn_k_norm {
                gguf.require_tensor(name)?;
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
        let scratch = RwLock::new(ForwardScratch::new(&config));

        info!(
            "loaded llama model {} with {} layers, {} heads, {} kv heads",
            model_name,
            config.block_count,
            config.attention_head_count,
            config.attention_head_count_kv
        );

        Ok(Self {
            gguf,
            config,
            token_embedding,
            output_norm,
            output,
            layers,
            model_name,
            vector_cache: RwLock::new(HashMap::new()),
            rope_freqs,
            scratch,
            batch_scratch: RwLock::new(BatchScratch::new()),
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn forward_token<C: KvCache + Sync>(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut C,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
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

        let mut scratch = self.scratch.write();
        let head_dim = self.config.head_dim();

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
            ..
        } = &mut *scratch;

        let mut x = self.embedding_lookup(token_id as usize)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if cache.len(layer_index) != position {
                return Err(XrtError::Runtime(format!(
                    "KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                    cache.len(layer_index)
                )));
            }

            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            apply_rmsnorm(&x, &attn_norm_weight, self.config.rms_norm_eps, normed);

            // Fused QKV projection: single parallel dispatch instead of 3 separate ones.
            // Only fuse when all matrices share the same dtype and cols (Q4_K_M uses
            // mixed quant: Q/K are Q4_K but V is often Q6_K, so we fuse Q+K and do V separately).
            {
                let can_fuse_all = layer.attn_q.dtype == layer.attn_k.dtype
                    && layer.attn_q.dtype == layer.attn_v.dtype
                    && layer.attn_q.cols == layer.attn_k.cols
                    && layer.attn_q.cols == layer.attn_v.cols;

                if can_fuse_all {
                    let q_data = self.gguf.tensor_data_raw(layer.attn_q.data_offset, layer.attn_q.nbytes);
                    let k_data = self.gguf.tensor_data_raw(layer.attn_k.data_offset, layer.attn_k.nbytes);
                    let v_data = self.gguf.tensor_data_raw(layer.attn_v.data_offset, layer.attn_v.nbytes);
                    matvec_quantized_fused(
                        &[q_data, k_data, v_data],
                        &[layer.attn_q.rows, layer.attn_k.rows, layer.attn_v.rows],
                        layer.attn_q.cols,
                        layer.attn_q.dtype,
                        normed,
                        &mut [&mut q[..], &mut k[..], &mut v[..]],
                    )?;
                } else {
                    // Fuse Q+K if they match, V runs separately
                    let can_fuse_qk = layer.attn_q.dtype == layer.attn_k.dtype
                        && layer.attn_q.cols == layer.attn_k.cols;
                    if can_fuse_qk {
                        let q_data = self.gguf.tensor_data_raw(layer.attn_q.data_offset, layer.attn_q.nbytes);
                        let k_data = self.gguf.tensor_data_raw(layer.attn_k.data_offset, layer.attn_k.nbytes);
                        matvec_quantized_fused(
                            &[q_data, k_data],
                            &[layer.attn_q.rows, layer.attn_k.rows],
                            layer.attn_q.cols,
                            layer.attn_q.dtype,
                            normed,
                            &mut [&mut q[..], &mut k[..]],
                        )?;
                    } else {
                        self.linear_resolved(&layer.attn_q, normed, q)?;
                        self.linear_resolved(&layer.attn_k, normed, k)?;
                    }
                    self.linear_resolved(&layer.attn_v, normed, v)?;
                }
            }

            // Qwen3-style per-head QK normalization (before RoPE)
            if let Some(ref q_norm_name) = layer.attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                self.apply_head_norm(q, self.config.attention_head_count, head_dim, &q_norm_w);
            }
            if let Some(ref k_norm_name) = layer.attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                self.apply_head_norm(k, self.config.attention_head_count_kv, head_dim, &k_norm_w);
            }

            self.rope_freqs.precompute_sincos_into(position, sin_cache, cos_cache);
            self.rope_freqs.apply_rotary_cached(q, self.config.attention_head_count, head_dim, sin_cache, cos_cache);
            self.rope_freqs.apply_rotary_cached(k, self.config.attention_head_count_kv, head_dim, sin_cache, cos_cache);

            cache.append(layer_index, k, v)?;
            let seq_len = cache.len(layer_index);
            let n_kv_heads = self.config.attention_head_count_kv;
            let head_group = self.config.attention_head_count / n_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            attn_out.fill(0.0);

            // Online softmax attention: single pass over the KV cache per head.
            // For each position, compute score, update running max/sum, and accumulate
            // weighted values incrementally. Eliminates the scores buffer and fuses
            // score computation + softmax + value accumulation into one pass.
            {
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

                            for position_idx in 0..seq_len {
                                let key_row = cache
                                    .key(layer_index, position_idx)
                                    .expect("missing key cache entry");
                                let key_head = &key_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                                let score = dot(q_head, key_head) * scale;

                                if score > max_score {
                                    // Rescale all previous accumulations
                                    let correction = (max_score - score).exp();
                                    sum_exp *= correction;
                                    for d in 0..head_dim {
                                        out_head[d] *= correction;
                                    }
                                    max_score = score;
                                }

                                let weight = (score - max_score).exp();
                                sum_exp += weight;

                                let value_row = cache
                                    .value(layer_index, position_idx)
                                    .expect("missing value cache entry");
                                let value_head =
                                    &value_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                                accumulate_scaled(out_head, value_head, weight);
                            }

                            // Final normalization
                            if sum_exp > 0.0 {
                                let inv_sum = sum_exp.recip();
                                for d in 0..head_dim {
                                    out_head[d] *= inv_sum;
                                }
                            }
                        }
                    }
                });
            }

            self.linear_resolved(&layer.attn_output, attn_out, proj)?;
            add_inplace(&mut x, proj);

            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            apply_rmsnorm(&x, &ffn_norm_weight, self.config.rms_norm_eps, normed);

            // Fused gate+up projection: single dispatch instead of 2.
            // Only fuse when gate and up share dtype and cols.
            if layer.ffn_gate.dtype == layer.ffn_up.dtype && layer.ffn_gate.cols == layer.ffn_up.cols {
                let gate_data = self.gguf.tensor_data_raw(layer.ffn_gate.data_offset, layer.ffn_gate.nbytes);
                let up_data = self.gguf.tensor_data_raw(layer.ffn_up.data_offset, layer.ffn_up.nbytes);
                matvec_quantized_fused(
                    &[gate_data, up_data],
                    &[layer.ffn_gate.rows, layer.ffn_up.rows],
                    layer.ffn_gate.cols,
                    layer.ffn_gate.dtype,
                    normed,
                    &mut [&mut gate[..], &mut up[..]],
                )?;
            } else {
                self.linear_resolved(&layer.ffn_gate, normed, gate)?;
                self.linear_resolved(&layer.ffn_up, normed, up)?;
            }
            swiglu(gate, up);
            self.linear_resolved(&layer.ffn_down, gate, down)?;
            add_inplace(&mut x, down);
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        apply_rmsnorm(
            &x,
            &output_norm_weight,
            self.config.rms_norm_eps,
            normed,
        );

        // Output projection directly into caller's buffer (zero alloc per token).
        self.linear_resolved(&self.output, normed, output_logits)?;
        Ok(())
    }

    pub fn forward_batch<C: KvCache + Sync>(
        &self,
        token_ids: &[u32],
        start_position: usize,
        cache: &mut C,
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
        for (t, &token_id) in token_ids.iter().enumerate() {
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
            // Slice buffers to actual seq_len (batch_scratch may be larger from previous calls)
            self.linear_batch_resolved(&layer.attn_q, &batch.normed[..seq_len * dim], seq_len, &mut batch.q[..seq_len * q_width])?;
            self.linear_batch_resolved(&layer.attn_k, &batch.normed[..seq_len * dim], seq_len, &mut batch.k[..seq_len * kv_width])?;
            self.linear_batch_resolved(&layer.attn_v, &batch.normed[..seq_len * dim], seq_len, &mut batch.v[..seq_len * kv_width])?;

            // 2c: Optional Qwen3 QK head normalization
            if let Some(ref q_norm_name) = layer.attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(ref k_norm_name) = layer.attn_k_norm {
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
                self.rope_freqs.precompute_sincos_into(start_position + t, &mut sin_buf, &mut cos_buf);
                self.rope_freqs.apply_rotary_cached(q_t, n_heads, head_dim, &sin_buf, &cos_buf);
                self.rope_freqs.apply_rotary_cached(k_t, n_kv_heads, head_dim, &sin_buf, &cos_buf);
            }

            // 2e: Batch KV cache append
            cache.append_batch(layer_index, &batch.k[..seq_len * kv_width], &batch.v[..seq_len * kv_width], seq_len)?;

            // 2f: Attention with causal mask
            let total_seq = cache.len(layer_index);

            batch.attn_out[..seq_len * q_width].fill(0.0);

            let max_attend = total_seq;
            let mut scores = vec![0.0f32; max_attend];

            for t in 0..seq_len {
                let attend_len = start_position + t + 1;

                for head in 0..n_heads {
                    let q_head = &batch.q
                        [t * q_width + head * head_dim..t * q_width + (head + 1) * head_dim];
                    let kv_head = head / head_group;

                    // Compute attention scores
                    for pos in 0..attend_len {
                        let key_row = cache.key(layer_index, pos).ok_or_else(|| {
                            XrtError::Runtime("missing key cache entry".to_string())
                        })?;
                        let key_head =
                            &key_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                        scores[pos] = dot(q_head, key_head) * scale;
                    }

                    softmax_inplace(&mut scores[..attend_len]);

                    // Weighted sum of values
                    let out_head = &mut batch.attn_out
                        [t * q_width + head * head_dim..t * q_width + (head + 1) * head_dim];
                    out_head.fill(0.0);
                    for (pos, &weight) in scores.iter().enumerate().take(attend_len) {
                        let value_row = cache.value(layer_index, pos).ok_or_else(|| {
                            XrtError::Runtime("missing value cache entry".to_string())
                        })?;
                        let value_head =
                            &value_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                        for (dst, src) in out_head.iter_mut().zip(value_head.iter()) {
                            *dst += weight * src;
                        }
                    }
                }
            }

            // 2g: Batch attention output projection
            self.linear_batch_resolved(
                &layer.attn_output,
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
            self.linear_batch_resolved(&layer.ffn_gate, &batch.normed[..seq_len * dim], seq_len, &mut batch.gate[..seq_len * ff_dim])?;
            self.linear_batch_resolved(&layer.ffn_up, &batch.normed[..seq_len * dim], seq_len, &mut batch.up[..seq_len * ff_dim])?;

            // SwiGLU per token
            for t in 0..seq_len {
                let gate_t = &mut batch.gate[t * ff_dim..(t + 1) * ff_dim];
                let up_t = &batch.up[t * ff_dim..(t + 1) * ff_dim];
                swiglu(gate_t, up_t);
            }

            self.linear_batch_resolved(&layer.ffn_down, &batch.gate[..seq_len * ff_dim], seq_len, &mut batch.down[..seq_len * dim])?;

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

            // Slice buffers to actual seq_len (batch_scratch may be larger from previous calls)
            let normed_sl = &batch.normed[..seq_len * dim];
            let q_sl = &mut batch.q[..seq_len * q_width];
            let k_sl = &mut batch.k[..seq_len * kv_width];
            let v_sl = &mut batch.v[..seq_len * kv_width];

            self.linear_batch_resolved(&layer.attn_q, normed_sl, seq_len, q_sl)?;
            self.linear_batch_resolved(&layer.attn_k, normed_sl, seq_len, k_sl)?;
            self.linear_batch_resolved(&layer.attn_v, normed_sl, seq_len, v_sl)?;

            if let Some(ref q_norm_name) = layer.attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(ref k_norm_name) = layer.attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                for t in 0..seq_len {
                    let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                    self.apply_head_norm(k_t, n_kv_heads, head_dim, &k_norm_w);
                }
            }

            for t in 0..seq_len {
                let q_t = &mut batch.q[t * q_width..(t + 1) * q_width];
                let k_t = &mut batch.k[t * kv_width..(t + 1) * kv_width];
                self.rope_freqs.precompute_sincos_into(start_position + t, &mut sin_buf, &mut cos_buf);
                self.rope_freqs.apply_rotary_cached(q_t, n_heads, head_dim, &sin_buf, &cos_buf);
                self.rope_freqs.apply_rotary_cached(k_t, n_kv_heads, head_dim, &sin_buf, &cos_buf);
            }

            cache.append_batch(layer_index, &batch.k[..seq_len * kv_width], &batch.v[..seq_len * kv_width], seq_len)?;

            let total_seq = cache.len(layer_index);
            batch.attn_out[..seq_len * q_width].fill(0.0);
            let mut scores = vec![0.0f32; total_seq];

            for t in 0..seq_len {
                let attend_len = start_position + t + 1;
                for head in 0..n_heads {
                    let q_head = &batch.q
                        [t * q_width + head * head_dim..t * q_width + (head + 1) * head_dim];
                    let kv_head = head / head_group;
                    for pos in 0..attend_len {
                        let key_row = cache.key(layer_index, pos).ok_or_else(|| {
                            XrtError::Runtime("missing key cache entry".to_string())
                        })?;
                        let key_head = &key_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                        scores[pos] = dot(q_head, key_head) * scale;
                    }
                    softmax_inplace(&mut scores[..attend_len]);
                    let out_head = &mut batch.attn_out
                        [t * q_width + head * head_dim..t * q_width + (head + 1) * head_dim];
                    out_head.fill(0.0);
                    for (pos, &weight) in scores.iter().enumerate().take(attend_len) {
                        let value_row = cache.value(layer_index, pos).ok_or_else(|| {
                            XrtError::Runtime("missing value cache entry".to_string())
                        })?;
                        let value_head = &value_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                        for (dst, src) in out_head.iter_mut().zip(value_head.iter()) {
                            *dst += weight * src;
                        }
                    }
                }
            }

            self.linear_batch_resolved(&layer.attn_output, &batch.attn_out[..seq_len * q_width], seq_len, &mut batch.proj[..seq_len * dim])?;
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
            self.linear_batch_resolved(&layer.ffn_gate, normed_sl, seq_len, &mut batch.gate[..seq_len * ff_dim])?;
            self.linear_batch_resolved(&layer.ffn_up, normed_sl, seq_len, &mut batch.up[..seq_len * ff_dim])?;
            for t in 0..seq_len {
                let gate_t = &mut batch.gate[t * ff_dim..(t + 1) * ff_dim];
                let up_t = &batch.up[t * ff_dim..(t + 1) * ff_dim];
                swiglu(gate_t, up_t);
            }
            self.linear_batch_resolved(&layer.ffn_down, &batch.gate[..seq_len * ff_dim], seq_len, &mut batch.down[..seq_len * dim])?;
            for i in 0..xs_len {
                batch.xs[i] += batch.down[i];
            }
        }

        // Step 3: Output projection on ALL positions
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

    /// Batch linear projection using pre-resolved weight (zero HashMap lookups).
    fn linear_batch_resolved(
        &self,
        w: &ResolvedWeight,
        inputs: &[f32],
        seq_len: usize,
        outputs: &mut [f32],
    ) -> Result<()> {
        let bytes = self.gguf.tensor_data_raw(w.data_offset, w.nbytes);
        matvec_quantized_batch(bytes, w.rows, w.cols, w.dtype, inputs, seq_len, outputs)
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

    fn embedding_lookup(&self, token_id: usize) -> Result<Vec<f32>> {
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
    fn linear_resolved(&self, w: &ResolvedWeight, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), w.cols);
        debug_assert_eq!(output.len(), w.rows);
        let bytes = self.gguf.tensor_data_raw(w.data_offset, w.nbytes);
        matvec_quantized(bytes, w.rows, w.cols, w.dtype, input, output)
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

fn required_usize(gguf: &GgufFile, key: &str) -> Result<usize> {
    gguf.metadata_usize(key).ok_or_else(|| {
        XrtError::InvalidMetadata(format!("missing or invalid usize metadata key {key}"))
    })
}
