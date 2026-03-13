use parking_lot::RwLock;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::info;
use xrt_core::{decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_kernels::cpu::{
    add_inplace, apply_rmsnorm, dequantize_q4_0_row, dequantize_q4_k_row,
    dequantize_q5_k_row, dequantize_q6_k_row, dequantize_q8_0_row, dot, matvec_quantized,
    matvec_quantized_batch, softmax_inplace, swiglu, RopeFreqs,
};

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

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_down: String,
    ffn_up: String,
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
    scores: Vec<f32>,
    proj: Vec<f32>,
    down: Vec<f32>,
}

impl ForwardScratch {
    fn new(config: &LlamaConfig) -> Self {
        Self {
            normed: vec![0.0; config.embedding_length],
            q: vec![0.0; config.q_width()],
            k: vec![0.0; config.kv_width()],
            v: vec![0.0; config.kv_width()],
            gate: vec![0.0; config.feed_forward_length],
            up: vec![0.0; config.feed_forward_length],
            attn_out: vec![0.0; config.q_width()],
            scores: Vec::new(),
            proj: vec![0.0; config.embedding_length],
            down: vec![0.0; config.embedding_length],
        }
    }
}

pub struct LlamaModel {
    gguf: Arc<GgufFile>,
    config: LlamaConfig,
    token_embedding: String,
    output_norm: String,
    output: String,
    layers: Vec<LayerWeights>,
    model_name: String,
    vector_cache: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    rope_freqs: RopeFreqs,
    scratch: RwLock<ForwardScratch>,
}

impl LlamaModel {
    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Self> {
        let config = LlamaConfig::from_gguf(&gguf)?;
        let token_embedding = "token_embd.weight".to_string();
        let output_norm = "output_norm.weight".to_string();
        let output = if gguf.tensor_info("output.weight").is_some() {
            "output.weight".to_string()
        } else {
            token_embedding.clone()
        };

        gguf.require_tensor(&token_embedding)?;
        gguf.require_tensor(&output_norm)?;
        gguf.require_tensor(&output)?;

        let mut layers = Vec::with_capacity(config.block_count);
        for index in 0..config.block_count {
            let q_norm_name = format!("blk.{index}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{index}.attn_k_norm.weight");
            let has_qk_norm = gguf.tensor_info(&q_norm_name).is_some();
            let layer = LayerWeights {
                attn_norm: format!("blk.{index}.attn_norm.weight"),
                attn_q: format!("blk.{index}.attn_q.weight"),
                attn_k: format!("blk.{index}.attn_k.weight"),
                attn_v: format!("blk.{index}.attn_v.weight"),
                attn_output: format!("blk.{index}.attn_output.weight"),
                ffn_norm: format!("blk.{index}.ffn_norm.weight"),
                ffn_gate: format!("blk.{index}.ffn_gate.weight"),
                ffn_down: format!("blk.{index}.ffn_down.weight"),
                ffn_up: format!("blk.{index}.ffn_up.weight"),
                attn_q_norm: if has_qk_norm { Some(q_norm_name) } else { None },
                attn_k_norm: if has_qk_norm { Some(k_norm_name) } else { None },
            };
            gguf.require_tensor(&layer.attn_norm)?;
            gguf.require_tensor(&layer.attn_q)?;
            gguf.require_tensor(&layer.attn_k)?;
            gguf.require_tensor(&layer.attn_v)?;
            gguf.require_tensor(&layer.attn_output)?;
            gguf.require_tensor(&layer.ffn_norm)?;
            gguf.require_tensor(&layer.ffn_gate)?;
            gguf.require_tensor(&layer.ffn_down)?;
            gguf.require_tensor(&layer.ffn_up)?;
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
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn forward_token<C: KvCache>(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut C,
    ) -> Result<Vec<f32>> {
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

        // Destructure so the borrow checker can track each field independently.
        let ForwardScratch {
            normed,
            q,
            k,
            v,
            gate,
            up,
            attn_out,
            scores,
            proj,
            down,
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

            self.linear_into(&layer.attn_q, normed, q)?;
            self.linear_into(&layer.attn_k, normed, k)?;
            self.linear_into(&layer.attn_v, normed, v)?;

            // Qwen3-style per-head QK normalization (before RoPE)
            if let Some(ref q_norm_name) = layer.attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                self.apply_head_norm(q, self.config.attention_head_count, head_dim, &q_norm_w);
            }
            if let Some(ref k_norm_name) = layer.attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                self.apply_head_norm(k, self.config.attention_head_count_kv, head_dim, &k_norm_w);
            }

            self.rope_freqs.apply_qk(
                q,
                k,
                self.config.attention_head_count,
                self.config.attention_head_count_kv,
                head_dim,
                position,
            );

            cache.append(layer_index, k, v)?;
            let seq_len = cache.len(layer_index);
            let head_group = self.config.attention_head_count / self.config.attention_head_count_kv;
            let scale = 1.0 / (head_dim as f32).sqrt();

            if scores.len() < seq_len {
                scores.resize(seq_len, 0.0);
            }
            attn_out.fill(0.0);

            for head in 0..self.config.attention_head_count {
                let q_head = &q[head * head_dim..(head + 1) * head_dim];
                let kv_head = head / head_group;
                for (position_idx, score) in scores.iter_mut().enumerate().take(seq_len) {
                    let key_row = cache
                        .key(layer_index, position_idx)
                        .ok_or_else(|| XrtError::Runtime("missing key cache entry".to_string()))?;
                    let key_head = &key_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                    *score = dot(q_head, key_head) * scale;
                }
                softmax_inplace(&mut scores[..seq_len]);

                let out_head = &mut attn_out[head * head_dim..(head + 1) * head_dim];
                out_head.fill(0.0);
                for (position_idx, &weight) in scores.iter().enumerate().take(seq_len) {
                    let value_row = cache
                        .value(layer_index, position_idx)
                        .ok_or_else(|| XrtError::Runtime("missing value cache entry".to_string()))?;
                    let value_head = &value_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                    for (dst, src) in out_head.iter_mut().zip(value_head.iter()) {
                        *dst += weight * src;
                    }
                }
            }

            self.linear_into(&layer.attn_output, attn_out, proj)?;
            add_inplace(&mut x, proj);

            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            apply_rmsnorm(&x, &ffn_norm_weight, self.config.rms_norm_eps, normed);

            self.linear_into(&layer.ffn_gate, normed, gate)?;
            self.linear_into(&layer.ffn_up, normed, up)?;
            swiglu(gate, up);
            self.linear_into(&layer.ffn_down, gate, down)?;
            add_inplace(&mut x, down);
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        apply_rmsnorm(
            &x,
            &output_norm_weight,
            self.config.rms_norm_eps,
            normed,
        );
        self.linear(&self.output, normed)
    }

    pub fn forward_batch<C: KvCache>(
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
            return self.forward_token(token_ids[0], start_position, cache);
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

        // Step 1: Batch embedding lookup
        let mut xs = vec![0.0f32; seq_len * dim];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let emb = self.embedding_lookup(token_id as usize)?;
            xs[t * dim..(t + 1) * dim].copy_from_slice(&emb);
        }

        // Batch scratch buffers (2D: seq_len * width)
        let mut normed_batch = vec![0.0f32; seq_len * dim];
        let mut q_batch = vec![0.0f32; seq_len * q_width];
        let mut k_batch = vec![0.0f32; seq_len * kv_width];
        let mut v_batch = vec![0.0f32; seq_len * kv_width];
        let mut gate_batch = vec![0.0f32; seq_len * ff_dim];
        let mut up_batch = vec![0.0f32; seq_len * ff_dim];
        let mut attn_out_batch = vec![0.0f32; seq_len * q_width];
        let mut proj_batch = vec![0.0f32; seq_len * dim];
        let mut down_batch = vec![0.0f32; seq_len * dim];

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
                let x_t = &xs[t * dim..(t + 1) * dim];
                let normed_t = &mut normed_batch[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &attn_norm_weight, eps, normed_t);
            }

            // 2b: Batch QKV projections (read weight matrix ONCE for all tokens)
            self.linear_batch_into(&layer.attn_q, &normed_batch, seq_len, &mut q_batch)?;
            self.linear_batch_into(&layer.attn_k, &normed_batch, seq_len, &mut k_batch)?;
            self.linear_batch_into(&layer.attn_v, &normed_batch, seq_len, &mut v_batch)?;

            // 2c: Optional Qwen3 QK head normalization
            if let Some(ref q_norm_name) = layer.attn_q_norm {
                let q_norm_w = self.load_vector(q_norm_name)?;
                for t in 0..seq_len {
                    let q_t = &mut q_batch[t * q_width..(t + 1) * q_width];
                    self.apply_head_norm(q_t, n_heads, head_dim, &q_norm_w);
                }
            }
            if let Some(ref k_norm_name) = layer.attn_k_norm {
                let k_norm_w = self.load_vector(k_norm_name)?;
                for t in 0..seq_len {
                    let k_t = &mut k_batch[t * kv_width..(t + 1) * kv_width];
                    self.apply_head_norm(k_t, n_kv_heads, head_dim, &k_norm_w);
                }
            }

            // 2d: RoPE for each token at its position
            for t in 0..seq_len {
                let q_t = &mut q_batch[t * q_width..(t + 1) * q_width];
                let k_t = &mut k_batch[t * kv_width..(t + 1) * kv_width];
                self.rope_freqs
                    .apply_qk(q_t, k_t, n_heads, n_kv_heads, head_dim, start_position + t);
            }

            // 2e: Batch KV cache append
            cache.append_batch(layer_index, &k_batch, &v_batch, seq_len)?;

            // 2f: Attention with causal mask
            let total_seq = cache.len(layer_index);

            attn_out_batch.fill(0.0);

            let max_attend = total_seq;
            let mut scores = vec![0.0f32; max_attend];

            for t in 0..seq_len {
                let attend_len = start_position + t + 1;

                for head in 0..n_heads {
                    let q_head = &q_batch
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
                    let out_head = &mut attn_out_batch
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
            self.linear_batch_into(
                &layer.attn_output,
                &attn_out_batch,
                seq_len,
                &mut proj_batch,
            )?;

            // 2h: Residual add
            for i in 0..xs.len() {
                xs[i] += proj_batch[i];
            }

            // 2i: FFN norm
            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            for t in 0..seq_len {
                let x_t = &xs[t * dim..(t + 1) * dim];
                let normed_t = &mut normed_batch[t * dim..(t + 1) * dim];
                apply_rmsnorm(x_t, &ffn_norm_weight, eps, normed_t);
            }

            // 2j: Batch FFN (gate, up, swiglu, down)
            self.linear_batch_into(&layer.ffn_gate, &normed_batch, seq_len, &mut gate_batch)?;
            self.linear_batch_into(&layer.ffn_up, &normed_batch, seq_len, &mut up_batch)?;

            // SwiGLU per token
            for t in 0..seq_len {
                let gate_t = &mut gate_batch[t * ff_dim..(t + 1) * ff_dim];
                let up_t = &up_batch[t * ff_dim..(t + 1) * ff_dim];
                swiglu(gate_t, up_t);
            }

            self.linear_batch_into(&layer.ffn_down, &gate_batch, seq_len, &mut down_batch)?;

            // 2k: Residual add
            for i in 0..xs.len() {
                xs[i] += down_batch[i];
            }
        }

        // Step 3: Output projection on LAST token only
        let last_x = &xs[(seq_len - 1) * dim..seq_len * dim];
        let output_norm_weight = self.load_vector(&self.output_norm)?;
        let mut normed_last = vec![0.0f32; dim];
        apply_rmsnorm(last_x, &output_norm_weight, eps, &mut normed_last);
        self.linear(&self.output, &normed_last)
    }

    fn linear_batch_into(
        &self,
        tensor_name: &str,
        inputs: &[f32],
        seq_len: usize,
        outputs: &mut [f32],
    ) -> Result<()> {
        let info = self.gguf.require_tensor(tensor_name)?;
        let rows = info.rows();
        let cols = info.row_len();
        let bytes = self.gguf.tensor_data(tensor_name)?;

        match info.dtype {
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
                matvec_quantized_batch(bytes, rows, cols, info.dtype, inputs, seq_len, outputs)?;
            }
            _ => {
                // Fallback: process token by token
                let mut scratch = vec![0.0f32; cols];
                for t in 0..seq_len {
                    let input = &inputs[t * cols..(t + 1) * cols];
                    for (row, out) in outputs[t * rows..(t + 1) * rows].iter_mut().enumerate() {
                        self.decode_row_into(info, bytes, row, &mut scratch)?;
                        *out = dot(input, &scratch);
                    }
                }
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

    fn linear_into(&self, tensor_name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        let info = self.gguf.require_tensor(tensor_name)?;
        if input.len() != info.row_len() {
            return Err(XrtError::Model(format!(
                "tensor {tensor_name} expects input width {}, received {}",
                info.row_len(),
                input.len()
            )));
        }
        let rows = info.rows();
        if output.len() != rows {
            return Err(XrtError::Model(format!(
                "tensor {tensor_name} expects output length {rows}, received {}",
                output.len()
            )));
        }

        let bytes = self.gguf.tensor_data(tensor_name)?;
        match info.dtype {
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
                matvec_quantized(bytes, rows, info.row_len(), info.dtype, input, output)?;
            }
            _ => {
                let mut scratch = vec![0.0f32; info.row_len()];
                for (row, out) in output.iter_mut().enumerate().take(rows) {
                    self.decode_row_into(info, bytes, row, &mut scratch)?;
                    *out = dot(&scratch, input);
                }
            }
        }
        Ok(())
    }

    fn linear(&self, tensor_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let info = self.gguf.require_tensor(tensor_name)?;
        if input.len() != info.row_len() {
            return Err(XrtError::Model(format!(
                "tensor {tensor_name} expects input width {}, received {}",
                info.row_len(),
                input.len()
            )));
        }

        let rows = info.rows();
        let bytes = self.gguf.tensor_data(tensor_name)?;
        let mut output = vec![0.0f32; rows];
        match info.dtype {
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
                matvec_quantized(bytes, rows, info.row_len(), info.dtype, input, &mut output)?;
            }
            _ => {
                let mut scratch = vec![0.0f32; info.row_len()];
                for (row, out) in output.iter_mut().enumerate().take(rows) {
                    self.decode_row_into(info, bytes, row, &mut scratch)?;
                    *out = dot(&scratch, input);
                }
            }
        }
        Ok(output)
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
