use parking_lot::RwLock;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::info;
use xrt_core::{decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};
use xrt_kernels::cpu::{
    add_inplace, apply_rmsnorm, apply_rotary_qk, dequantize_q4_0_row, dequantize_q4_k_row,
    dequantize_q5_k_row, dequantize_q6_k_row, dequantize_q8_0_row, dot, matvec_quantized,
    softmax_inplace, swiglu,
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
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .unwrap_or("llama")
            .to_string();
        if architecture != "llama" {
            return Err(XrtError::Unsupported(format!(
                "xrt-models currently supports only llama GGUF models, found architecture {architecture}"
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

        let head_dim = embedding_length / attention_head_count;
        let rope_dimension_count = gguf
            .metadata_usize(&format!("{prefix}.rope.dimension_count"))
            .unwrap_or(head_dim);
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
        })
    }

    pub fn head_dim(&self) -> usize {
        self.embedding_length / self.attention_head_count
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

        let mut x = self.embedding_lookup(token_id as usize)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if cache.len(layer_index) != position {
                return Err(XrtError::Runtime(format!(
                    "KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                    cache.len(layer_index)
                )));
            }

            let residual = x.clone();
            let attn_norm_weight = self.load_vector(&layer.attn_norm)?;
            let mut normed = vec![0.0f32; self.config.embedding_length];
            apply_rmsnorm(&x, &attn_norm_weight, self.config.rms_norm_eps, &mut normed);

            let mut q = self.linear(&layer.attn_q, &normed)?;
            let mut k = self.linear(&layer.attn_k, &normed)?;
            let v = self.linear(&layer.attn_v, &normed)?;
            apply_rotary_qk(
                &mut q,
                &mut k,
                self.config.attention_head_count,
                self.config.attention_head_count_kv,
                self.config.head_dim(),
                position,
                self.config.rope_dimension_count,
                self.config.rope_freq_base,
                self.config.rope_freq_scale,
            );

            let attention = self.attention(layer_index, &q, &k, &v, cache)?;
            let projected = self.linear(&layer.attn_output, &attention)?;
            x = residual;
            add_inplace(&mut x, &projected);

            let residual = x.clone();
            let ffn_norm_weight = self.load_vector(&layer.ffn_norm)?;
            let mut normed = vec![0.0f32; self.config.embedding_length];
            apply_rmsnorm(&x, &ffn_norm_weight, self.config.rms_norm_eps, &mut normed);

            let mut gate = self.linear(&layer.ffn_gate, &normed)?;
            let up = self.linear(&layer.ffn_up, &normed)?;
            swiglu(&mut gate, &up);
            let down = self.linear(&layer.ffn_down, &gate)?;
            x = residual;
            add_inplace(&mut x, &down);
        }

        let output_norm_weight = self.load_vector(&self.output_norm)?;
        let mut hidden = vec![0.0f32; self.config.embedding_length];
        apply_rmsnorm(
            &x,
            &output_norm_weight,
            self.config.rms_norm_eps,
            &mut hidden,
        );
        self.linear(&self.output, &hidden)
    }

    fn attention<C: KvCache>(
        &self,
        layer_index: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        cache: &mut C,
    ) -> Result<Vec<f32>> {
        cache.append(layer_index, k, v)?;
        let seq_len = cache.len(layer_index);
        let head_dim = self.config.head_dim();
        let head_group = self.config.attention_head_count / self.config.attention_head_count_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; self.config.embedding_length];
        let mut scores = vec![0.0f32; seq_len];

        for head in 0..self.config.attention_head_count {
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            let kv_head = head / head_group;
            for (position, score) in scores.iter_mut().enumerate().take(seq_len) {
                let key = cache
                    .key(layer_index, position)
                    .ok_or_else(|| XrtError::Runtime("missing key cache entry".to_string()))?;
                let key_head = &key[kv_head * head_dim..(kv_head + 1) * head_dim];
                *score = dot(q_head, key_head) * scale;
            }
            softmax_inplace(&mut scores);

            let out_head = &mut output[head * head_dim..(head + 1) * head_dim];
            out_head.fill(0.0);
            for (position, &weight) in scores.iter().enumerate().take(seq_len) {
                let value = cache
                    .value(layer_index, position)
                    .ok_or_else(|| XrtError::Runtime("missing value cache entry".to_string()))?;
                let value_head = &value[kv_head * head_dim..(kv_head + 1) * head_dim];
                for (dst, src) in out_head.iter_mut().zip(value_head.iter()) {
                    *dst += weight * src;
                }
            }
        }

        Ok(output)
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
